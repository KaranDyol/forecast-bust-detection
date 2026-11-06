"""
dashboard.py
============
Phase 4: Operational Intelligence Hub for AI-Based Forecast Bust Early Warning.

A Streamlit-based mission-critical dashboard for meteorological agencies
(IMD, ECMWF, NOAA) to monitor medium-range forecast reliability, visualize regional
risk maps across India, and diagnose root causes using Explainable AI (SHAP).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# Resolve directory and model paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bust_model.pkl"
FASTAPI_URL = "http://127.0.0.1:8000/predict_bust_risk"

# Streamlit Page Setup
st.set_page_config(
    page_title="AI NWP Bust Early Warning System",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Professional Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-val-crit {
        color: #f85149;
        font-size: 26px;
        font-weight: 700;
    }
    .metric-val-good {
        color: #3fb950;
        font-size: 26px;
        font-weight: 700;
    }
    .metric-val-neutral {
        color: #58a6ff;
        font-size: 26px;
        font-weight: 700;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# MODEL & BACKEND CLIENT (WITH IN-PROCESS FALLBACK)
# ==============================================================================

@st.cache_resource
def get_in_memory_model_bundle():
    """Load model artifact in memory for fast local fallback if FastAPI server is offline."""
    if MODEL_PATH.is_file():
        return joblib.load(MODEL_PATH)
    return None


def query_bust_detection_service(station_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Communicates with the FastAPI microservice created in Phase 3.
    If the API server is not running on port 8000, seamlessly falls back to
    the exact same model and SHAP logic in-process to guarantee uninterrupted UX.
    """
    # 1. Attempt Live FastAPI HTTP Call
    try:
        response = requests.post(FASTAPI_URL, json=station_payload, timeout=0.8)
        if response.status_code == 200:
            res_json = response.json()
            res_json["_source"] = "FastAPI Service (Live HTTP)"
            return res_json
    except Exception:
        pass  # Server offline or not reachable, proceed to local fallback

    # 2. Local Fallback Execution (matches Phase 3 API logic exactly)
    bundle = get_in_memory_model_bundle()
    if bundle is None:
        raise RuntimeError("Model bundle not found. Please run train_model.py first.")

    model = bundle["model"]
    explainer = bundle["explainer"]
    feature_names = bundle["feature_names"]

    df_sample = pd.DataFrame([{
        "predicted_temp": station_payload["predicted_temp"],
        "predicted_rainfall": station_payload["predicted_rainfall"],
        "wind_shear": station_payload["wind_shear"],
        "pressure_drop": station_payload["pressure_drop"],
    }])[feature_names].astype(float)

    prob = float(model.predict_proba(df_sample)[0, 1])
    risk_pct = round(prob * 100.0, 2)
    is_warning = risk_pct >= 50.0

    raw_shap = explainer.shap_values(df_sample)
    if isinstance(raw_shap, list):
        shap_vec = raw_shap[1][0]
    elif len(raw_shap.shape) == 3:
        shap_vec = raw_shap[0, :, 1]
    else:
        shap_vec = raw_shap[0]

    narratives = {
        "pressure_drop": {
            "pos": "Abnormal barometric pressure drop detected: high frontal boundary timing uncertainty.",
            "neg": "Barometric pressure stable: synoptically quiescent atmospheric column."
        },
        "wind_shear": {
            "pos": "Elevated vertical wind shear: promotes convective cloud organization prone to spatial misforecast.",
            "neg": "Weak wind shear: laminar airflow conforms well to hydrostatic NWP parameterizations."
        },
        "predicted_rainfall": {
            "pos": "High convective rainfall volume forecast: microphysics boundary uncertainty elevated.",
            "neg": "Minimal rainfall forecasted: predictable dry-line dynamics."
        },
        "predicted_temp": {
            "pos": "Surface temperature pushes extreme boundary, stressing model boundary-layer physics.",
            "neg": "Near-climatology temperature maintains high model fidelity."
        }
    }

    factors = []
    for feat, s_val in zip(feature_names, shap_vec):
        val = float(df_sample[feat].iloc[0])
        s_float = float(s_val)
        is_pos = s_float > 0
        impact_dir = "increases_bust_risk" if is_pos else "decreases_bust_risk"
        desc = narratives.get(feat, {}).get("pos" if is_pos else "neg", "Atmospheric factor modulating forecast skill.")
        factors.append({
            "feature": feat,
            "feature_value": round(val, 2),
            "shap_value": round(s_float, 4),
            "impact": impact_dir,
            "description": desc,
            "abs_shap": abs(s_float)
        })

    factors.sort(key=lambda x: x["abs_shap"], reverse=True)

    tier = "CRITICAL" if risk_pct >= 75.0 else ("HIGH" if risk_pct >= 50.0 else ("ELEVATED" if risk_pct >= 30.0 else "LOW"))

    return {
        "coordinates": {"latitude": station_payload["latitude"], "longitude": station_payload["longitude"]},
        "risk_probability_percentage": risk_pct,
        "is_high_risk_warning": is_warning,
        "risk_level": tier,
        "explainability_factors": factors[:3],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "_source": "In-Process Engine (Local Fallback)"
    }


# ==============================================================================
# SYNTHETIC METEOROLOGICAL STATIONS ACROSS INDIA
# ==============================================================================

INDIAN_MET_STATIONS = [
    {"name": "New Delhi (Safdarjung)", "state": "Delhi", "lat": 28.5847, "lon": 77.2066, "base_temp": 31.0, "zone": "North Plains"},
    {"name": "Mumbai (Colaba / Santacruz)", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "base_temp": 32.5, "zone": "West Coast"},
    {"name": "Kolkata (Alipore)", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "base_temp": 33.0, "zone": "East Coast / BoB"},
    {"name": "Chennai (Meenambakkam)", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "base_temp": 34.0, "zone": "South Coast"},
    {"name": "Bengaluru (HAL)", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "base_temp": 27.5, "zone": "Deccan Plateau"},
    {"name": "Hyderabad (Begumpet)", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "base_temp": 32.0, "zone": "Central Peninsula"},
    {"name": "Bhubaneswar (IMD Met)", "state": "Odisha", "lat": 20.2961, "lon": 85.8245, "base_temp": 33.5, "zone": "East Coast / BoB"},
    {"name": "Srinagar (Rambagh)", "state": "J&K", "lat": 34.0837, "lon": 74.7973, "base_temp": 18.0, "zone": "Western Himalaya"},
    {"name": "Guwahati (Borjhar)", "state": "Assam", "lat": 26.1445, "lon": 91.7362, "base_temp": 29.5, "zone": "Northeast Basin"},
    {"name": "Ahmedabad (Airport)", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "base_temp": 35.0, "zone": "West Arid"},
    {"name": "Jaipur (Sanganer)", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "base_temp": 34.5, "zone": "Northwest Semi-Arid"},
    {"name": "Nagpur (Sonegaon)", "state": "Maharashtra", "lat": 21.1458, "lon": 79.0882, "base_temp": 33.0, "zone": "Central India"},
    {"name": "Visakhapatnam (Cyclone Warning)", "state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185, "base_temp": 32.8, "zone": "East Coast / BoB"},
    {"name": "Kochi (CIAL)", "state": "Kerala", "lat": 9.9312, "lon": 76.2673, "base_temp": 30.5, "zone": "Southwest Coast"},
    {"name": "Patna (Civil Aerodrome)", "state": "Bihar", "lat": 25.5941, "lon": 85.1376, "base_temp": 32.0, "zone": "Gangetic Plain"},
    {"name": "Bhopal (Bairagarh)", "state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126, "base_temp": 31.5, "zone": "Central India"},
    {"name": "Pune (Shivajinagar)", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "base_temp": 30.0, "zone": "Western Ghats Rain Shadow"},
    {"name": "Dehradun (Forest Research)", "state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322, "base_temp": 24.5, "zone": "Sub-Himalayan"}
]


def generate_station_forecasts(lead_day: int, random_seed: int = 101) -> pd.DataFrame:
    """
    Generate realistic synoptic forecast states across India stations modulated by forecast lead day (1 to 10).
    Higher lead days introduce greater atmospheric dispersion, shear shocks, and pressure tendency drops.
    """
    rng = np.random.default_rng(random_seed + lead_day * 7)
    records = []

    for station in INDIAN_MET_STATIONS:
        # Base atmospheric stability: Lead time factor amplifies forecast divergence
        lead_dispersion = 1.0 + (lead_day - 1) * 0.14

        # Regional synoptic weather regimes:
        is_bob_storm_zone = "BoB" in station["zone"] or "East Coast" in station["zone"]
        is_himalayan = "Himalaya" in station["zone"]

        if is_bob_storm_zone and lead_day >= 3:
            # Simulate approaching tropical depression / low-pressure disturbance in Bay of Bengal
            pressure_drop = round(float(rng.uniform(14.0, 24.0) * (lead_day / 5.0)), 2)
            wind_shear = round(float(rng.uniform(28.0, 44.0)), 2)
            predicted_rain = round(float(rng.uniform(22.0, 52.0) * lead_dispersion), 1)
        elif is_himalayan and lead_day >= 4:
            # Western Disturbance active over northern mountains
            pressure_drop = round(float(rng.uniform(10.0, 19.0)), 2)
            wind_shear = round(float(rng.uniform(32.0, 48.0)), 2)
            predicted_rain = round(float(rng.uniform(8.0, 26.0)), 1)
        else:
            # Normal or moderate convective activity
            pressure_drop = round(float(rng.gamma(shape=2.0, scale=3.0) * (1.0 + lead_day * 0.08)), 2)
            wind_shear = round(float(rng.normal(loc=16.0, scale=7.0)), 2)
            predicted_rain = round(float(rng.exponential(scale=6.0) if rng.random() > 0.4 else 0.0), 1)

        pressure_drop = max(0.5, min(pressure_drop, 28.0))
        wind_shear = max(4.0, min(wind_shear, 50.0))

        predicted_temp = round(station["base_temp"] + float(rng.normal(0, 1.2 * lead_dispersion)), 1)

        # Build payload matching FastAPI NWPForecastInput
        payload = {
            "latitude": station["lat"],
            "longitude": station["lon"],
            "predicted_temp": predicted_temp,
            "predicted_rainfall": predicted_rain,
            "wind_shear": wind_shear,
            "pressure_drop": pressure_drop,
            "lead_time_hours": lead_day * 24
        }

        # Query decision support service (FastAPI / local fallback)
        eval_result = query_bust_detection_service(payload)

        records.append({
            "station_name": station["name"],
            "state": station["state"],
            "zone": station["zone"],
            "lat": station["lat"],
            "lon": station["lon"],
            "predicted_temp": predicted_temp,
            "predicted_rainfall": predicted_rain,
            "wind_shear": wind_shear,
            "pressure_drop": pressure_drop,
            "risk_percentage": eval_result["risk_probability_percentage"],
            "is_high_risk": eval_result["is_high_risk_warning"],
            "risk_level": eval_result["risk_level"],
            "explainability_factors": eval_result["explainability_factors"],
            "eval_source": eval_result.get("_source", "Engine")
        })

    return pd.DataFrame(records)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/partly-cloudy-rain.png", width=68)
    st.title("IMD / NWP Operational Console")
    st.markdown("---")

    st.subheader("⏱️ Forecast Horizon")
    lead_day = st.slider(
        "Day 1 to Day 10 Forecast Slider",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Select Medium-Range NWP lead time (Day 1 = 24h, Day 10 = 240h)"
    )
    st.info(f"**Lead Time:** Day +{lead_day} ({lead_day * 24} Hours Horizon)")

    st.markdown("---")
    st.subheader("🌐 Global NWP Model Core")
    nwp_model = st.selectbox(
        "Ensemble Target System",
        ["ECMWF IFS (0.1° High-Res)", "NCMRWF UM (Global 12km)", "NOAA GFS (0.25°)", "IMD GFS Ensemble (GEFS)"],
        index=0
    )

    st.markdown("---")
    st.subheader("⚠️ Alert Configuration")
    alert_threshold = st.slider(
        "Critical Bust Threshold (%)",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
        help="Stations above this threshold trigger Red Alerts on the regional map"
    )

    st.markdown("---")
    st.caption("AI Decision Support Layer: Antigravity Met Engine v1.0")


# ==============================================================================
# MAIN DASHBOARD CONTENT
# ==============================================================================

# 1. Header Banner
st.title("⛈️ AI-Based Forecast Bust Early Warning Hub")
st.markdown(
    "**Operational Decision Support Layer** over Numerical Weather Prediction (NWP) systems. "
    "Detects synoptically driven forecast failures (busts) before dissemination, backed by Explainable AI (SHAP)."
)

# Ingest and Score Station Data
df_stations = generate_station_forecasts(lead_day=lead_day)

# 2. Key Metrics Row
high_risk_count = int((df_stations["risk_percentage"] >= alert_threshold).sum())
elevated_count = int(((df_stations["risk_percentage"] >= 50.0) & (df_stations["risk_percentage"] < alert_threshold)).sum())
mean_risk = float(df_stations["risk_percentage"].mean())
most_vulnerable = df_stations.loc[df_stations["risk_percentage"].idxmax()]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #8b949e; font-size: 13px;">STATIONS MONITORED</div>
        <div class="metric-val-neutral">{len(df_stations)} Synoptic Nodes</div>
        <div style="color: #8b949e; font-size: 12px;">Active All-India Grid</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #8b949e; font-size: 13px;">CRITICAL BUST ZONES (> {alert_threshold}%)</div>
        <div class="metric-val-crit">{high_risk_count} Alert Nodes</div>
        <div style="color: #f85149; font-size: 12px;">Immediate Forecaster Review</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #8b949e; font-size: 13px;">ELEVATED RISK ZONES (50-{alert_threshold}%)</div>
        <div class="metric-val-neutral">{elevated_count} Stations</div>
        <div style="color: #8b949e; font-size: 12px;">Medium Vulnerability</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #8b949e; font-size: 13px;">TOP VULNERABLE STATION</div>
        <div class="metric-val-crit">{most_vulnerable['station_name'].split()[0]} ({most_vulnerable['risk_percentage']}%)</div>
        <div style="color: #8b949e; font-size: 12px;">{most_vulnerable['zone']}</div>
    </div>
    """, unsafe_allow_html=True)

if high_risk_count > 0:
    st.error(
        f"🚨 **CRITICAL SYNOPTIC BUST WARNING (Day +{lead_day}):** {high_risk_count} meteorological stations exceed the "
        f"{alert_threshold}% forecast failure threshold. Severe error likely in rainfall timing or temperature anomaly."
    )
else:
    st.success(
        f"✅ **NOMINAL FORECAST RELIABILITY (Day +{lead_day}):** All 18 synoptic stations exhibit stable NWP characteristics below {alert_threshold}% bust risk."
    )

# 3. Interactive Geographical Map (Folium)
st.markdown("### 🗺️ All-India Regional Forecast Bust Risk Map")
st.caption(
    f"Color Legend: 🔴 **High-Risk Bust Zone** (Risk > {alert_threshold}%) | "
    "🟠 **Elevated Risk** (50% - 70%) | 🟢 **Reliable NWP Zone** (< 50%)"
)

# Center folium map on India
map_india = folium.Map(
    location=[22.8, 79.5],
    zoom_start=5,
    tiles="OpenStreetMap",
    control_scale=True
)

for _, row in df_stations.iterrows():
    risk = row["risk_percentage"]
    # Folium Color coding as requested: High-Risk (> 70%) in RED, Reliable in GREEN
    if risk >= alert_threshold:
        color = "#e63946"  # Red
        fill_color = "#e63946"
        icon_name = "exclamation-triangle"
        status_label = "CRITICAL BUST RISK"
        radius = 11
    elif risk >= 50.0:
        color = "#f4a261"  # Orange/Amber
        fill_color = "#f4a261"
        icon_name = "warning"
        status_label = "ELEVATED RISK"
        radius = 8
    else:
        color = "#2a9d8f"  # Green
        fill_color = "#2a9d8f"
        icon_name = "check-circle"
        status_label = "RELIABLE"
        radius = 7

    popup_html = f"""
    <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
        <h4 style="margin: 0 0 6px 0; color: {color};">{row['station_name']}</h4>
        <b>Status:</b> {status_label}<br>
        <b>Bust Risk:</b> <span style="font-size: 14px; font-weight: bold; color: {color};">{risk}%</span><br>
        <b>Forecasted Temp:</b> {row['predicted_temp']} °C<br>
        <b>Forecasted Rain:</b> {row['predicted_rainfall']} mm<br>
        <b>Pressure Drop (24h):</b> {row['pressure_drop']} hPa<br>
        <b>Wind Shear:</b> {row['wind_shear']} m/s<br>
        <hr style="margin: 6px 0;">
        <small style="color: #666;">Source: {row['eval_source']}</small>
    </div>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"{row['station_name']} — Bust Risk: {risk}% ({status_label})",
        color=color,
        fill=True,
        fill_color=fill_color,
        fill_opacity=0.85,
        weight=2
    ).add_to(map_india)

# Render map in Streamlit
st_folium(map_india, width="100%", height=480, returned_objects=[])

# ==============================================================================
# 4. EXPLAINABLE AI (SHAP) INSIGHTS PANEL
# ==============================================================================

st.markdown("---")
st.header("🧠 Explainable AI Insights (SHAP Diagnostic Engine)")
st.markdown(
    "Click or select a synoptic observation station below to analyze the **meteorological root cause drivers** "
    "quantified via Shapley Additive Explanations (SHAP)."
)

# Station Selector for simulated click / inspection
station_names = df_stations["station_name"].tolist()
# Default to the most high-risk station to immediately show value
default_idx = int(df_stations["risk_percentage"].idxmax())

selected_station_name = st.selectbox(
    "🎯 Select Observation Node / High-Risk Bust Zone for Deep SHAP Diagnostic:",
    options=station_names,
    index=default_idx
)

selected_record = df_stations[df_stations["station_name"] == selected_station_name].iloc[0]
sel_risk = selected_record["risk_percentage"]
sel_factors = selected_record["explainability_factors"]

# Display Station Profile & Status Alert
col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.subheader(f"📍 Station: {selected_record['station_name']}")
    st.write(f"**State / Zone:** {selected_record['state']} ({selected_record['zone']})")
    st.write(f"**Geographical Coordinates:** `{selected_record['lat']}° N, {selected_record['lon']}° E`")
    st.write(f"**Lead Time Horizon:** `Day +{lead_day} ({lead_day * 24}h)` | Model: `{nwp_model}`")

    if sel_risk >= alert_threshold:
        st.error(
            f"🚨 **HIGH-RISK BUST PREDICTED ({sel_risk}%)**\n\n"
            f"Numerical model output has a high probability of exceeding acceptable operational error "
            f"(> 3°C temp error OR > 15 mm rain error). Manual synoptic intervention advised."
        )
    elif sel_risk >= 50.0:
        st.warning(
            f"⚠️ **ELEVATED FORECAST RISK ({sel_risk}%)**\n\n"
            f"Atmospheric column is unstable. Monitor incoming satellite and radar imagery."
        )
    else:
        st.success(
            f"✅ **FORECAST RELIABLE ({sel_risk}%)**\n\n"
            f"Numerical forecast is robust. Synoptic conditions fall within low-uncertainty parameters."
        )

    # Synoptic Parameters Table
    st.markdown("**Observed Synoptic Inputs:**")
    st.dataframe(
        pd.DataFrame([{
            "Forecast Temp": f"{selected_record['predicted_temp']} °C",
            "Forecast Rain": f"{selected_record['predicted_rainfall']} mm",
            "Wind Shear (0-6km)": f"{selected_record['wind_shear']} m/s",
            "Pressure Tendency": f"-{selected_record['pressure_drop']} hPa / 24h"
        }]),
        hide_index=True,
        width="stretch"
    )

with col_right:
    st.subheader("📊 Root Cause Attribution (SHAP Waterfall / Contribution)")

    # Prepare data for Plotly Horizontal Bar / Waterfall Chart
    feat_names = [f["feature"].replace("_", " ").title() for f in sel_factors]
    shap_vals = [f["shap_value"] for f in sel_factors]
    colors = ["#e63946" if v > 0 else "#2a9d8f" for v in shap_vals]
    hover_texts = [
        f"Value: {f['feature_value']}<br>SHAP Impact: {f['shap_value']:+.4f}<br>{f['description']}"
        for f in sel_factors
    ]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=feat_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in shap_vals],
        textposition="outside",
        hoverinfo="text",
        hovertext=hover_texts
    ))

    fig.update_layout(
        title=f"SHAP Root Cause Attribution for {selected_record['station_name'].split()[0]}",
        xaxis_title="SHAP Attribution (Log-Odds Impact on Bust Risk)",
        yaxis_title="Atmospheric Synoptic Driver",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        margin=dict(l=10, r=30, t=40, b=30),
        height=280
    )

    st.plotly_chart(fig, width="stretch")

# 5. Detailed Root Cause Drivers Explanation Cards
st.markdown("#### 🔬 Detailed Meteorological Driver Analysis")
cards_cols = st.columns(len(sel_factors))

for idx, factor in enumerate(sel_factors):
    is_bust_driver = factor["impact"] == "increases_bust_risk"
    badge_color = "#f85149" if is_bust_driver else "#3fb950"
    badge_text = "RISK ACCELERATOR" if is_bust_driver else "STABILIZING FACTOR"

    with cards_cols[idx]:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {badge_color};">
            <span style="background-color: {badge_color}22; color: {badge_color}; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                {badge_text}
            </span>
            <h4 style="margin: 8px 0 4px 0;">{factor['feature'].replace('_', ' ').title()}</h4>
            <div style="font-size: 13px; color: #8b949e;">Observed Value: <b>{factor['feature_value']}</b></div>
            <div style="font-size: 13px; color: #8b949e;">SHAP Score: <b>{factor['shap_value']:+.4f}</b></div>
            <p style="font-size: 12px; margin-top: 8px; color: #c9d1d9;">{factor['description']}</p>
        </div>
        """, unsafe_allow_html=True)

# 6. Synoptic Risk Summary Table
with st.expander("📋 View Full Regional Synoptic Station Summary Table"):
    st.dataframe(
        df_stations[[
            "station_name", "zone", "risk_percentage", "risk_level",
            "predicted_temp", "predicted_rainfall", "pressure_drop", "wind_shear"
        ]].sort_values(by="risk_percentage", ascending=False),
        width="stretch",
        hide_index=True
    )

st.markdown("---")
st.caption("AI-Based Forecast Bust Detection System | Operational Met Intelligence Hub")
