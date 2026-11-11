"""
Medium-range weather forecast bust monitor and SHAP diagnostics dashboard.
"""

from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bust_model.pkl"
API_BASE_URL = "http://127.0.0.1:8000"

# Warm earthy palette: Cream (#FEF3E2), Amber (#F4AE52), Terracotta (#D4621A), Espresso (#2A1A0E)
PALETTE = {
    "cream": "#FEF3E2",
    "surface": "#FFF9F0",
    "sand": "#F8E5C8",
    "border": "#E3CCA8",
    "amber": "#F4AE52",
    "terracotta": "#D4621A",
    "critical_red": "#C23B16",
    "espresso": "#2A1A0E",
    "muted_brown": "#6E5038",
    "sage_green": "#3B6E4C",
}

st.set_page_config(
    page_title="Forecast Bust Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: {PALETTE['espresso']};
}}

[data-testid="stSidebar"] {{
    border-right: 1px solid {PALETTE['border']};
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1240px;
}}

.eyebrow-tag {{
    display: inline-block;
    background-color: {PALETTE['amber']}40;
    color: {PALETTE['espresso']};
    border: 1px solid {PALETTE['amber']};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 8px;
}}

.page-title {{
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: {PALETTE['espresso']};
    margin: 0 0 4px 0;
    line-height: 1.15;
}}

.page-subtitle {{
    font-size: 15px;
    color: {PALETTE['muted_brown']};
    margin: 0 0 22px 0;
}}

.sidebar-brand {{
    background-color: {PALETTE['espresso']};
    color: {PALETTE['cream']};
    padding: 14px 16px;
    border-radius: 8px;
    border-bottom: 3px solid {PALETTE['amber']};
    margin-bottom: 18px;
}}

.sidebar-brand-kicker {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['amber']};
    margin-bottom: 4px;
}}

.sidebar-brand-title {{
    font-size: 17px;
    font-weight: 700;
    color: {PALETTE['cream']};
    margin: 0;
}}

.kpi-card {{
    background-color: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}}

.kpi-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {PALETTE['muted_brown']};
    margin-bottom: 6px;
}}

.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}}

.kpi-sub {{
    font-size: 12px;
    color: {PALETTE['muted_brown']};
    margin-top: 4px;
}}

.status-banner {{
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0 20px 0;
    font-size: 14px;
    line-height: 1.45;
    color: {PALETTE['espresso']};
}}

.banner-crit {{
    background-color: rgba(212, 98, 26, 0.12);
    border: 1px solid rgba(212, 98, 26, 0.35);
    border-left: 4px solid {PALETTE['terracotta']};
}}

.banner-warn {{
    background-color: rgba(244, 174, 82, 0.18);
    border: 1px solid rgba(244, 174, 82, 0.55);
    border-left: 4px solid {PALETTE['amber']};
}}

.banner-ok {{
    background-color: rgba(59, 110, 76, 0.11);
    border: 1px solid rgba(59, 110, 76, 0.32);
    border-left: 4px solid {PALETTE['sage_green']};
}}

.map-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    font-size: 13px;
    color: {PALETTE['muted_brown']};
    margin-bottom: 10px;
}}

.legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}

.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}

.station-panel {{
    background-color: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 12px;
}}

.stat-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-top: 12px;
}}

.stat-cell {{
    background-color: {PALETTE['cream']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 10px 12px;
}}

.stat-cell-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {PALETTE['muted_brown']};
}}

.stat-cell-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: {PALETTE['espresso']};
    margin-top: 2px;
}}

.driver-card {{
    background-color: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 14px 16px;
    height: 100%;
    margin-bottom: 10px;
}}

.driver-pill {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
}}

.mono-meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: {PALETTE['muted_brown']};
}}

@media (max-width: 768px) {{
    .page-title {{ font-size: 24px; }}
    .kpi-value {{ font-size: 20px; }}
    .stat-grid {{ grid-template-columns: 1fr; }}
}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> Dict[str, Any]:
    """Load the serialized model and SHAP TreeExplainer once per worker."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing model file at {MODEL_PATH.resolve()}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=60, show_spinner=False)
def is_api_reachable() -> bool:
    """Probe the FastAPI service once every 60s instead of retrying inside per-station loops."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=0.3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


INDIAN_MET_STATIONS = [
    {"name": "New Delhi (Safdarjung)", "state": "Delhi", "lat": 28.5847, "lon": 77.2066, "base_temp": 31.0, "zone": "North Plains"},
    {"name": "Mumbai (Santacruz)", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "base_temp": 32.5, "zone": "West Coast"},
    {"name": "Kolkata (Alipore)", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "base_temp": 33.0, "zone": "East Coast / BoB"},
    {"name": "Chennai (Meenambakkam)", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "base_temp": 34.0, "zone": "South Coast"},
    {"name": "Bengaluru (HAL)", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "base_temp": 27.5, "zone": "Deccan Plateau"},
    {"name": "Hyderabad (Begumpet)", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "base_temp": 32.0, "zone": "Central Peninsula"},
    {"name": "Bhubaneswar (Airport)", "state": "Odisha", "lat": 20.2961, "lon": 85.8245, "base_temp": 33.5, "zone": "East Coast / BoB"},
    {"name": "Srinagar (Rambagh)", "state": "J&K", "lat": 34.0837, "lon": 74.7973, "base_temp": 18.0, "zone": "Western Himalaya"},
    {"name": "Guwahati (Borjhar)", "state": "Assam", "lat": 26.1445, "lon": 91.7362, "base_temp": 29.5, "zone": "Northeast Basin"},
    {"name": "Ahmedabad (Airport)", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "base_temp": 35.0, "zone": "West Arid"},
    {"name": "Jaipur (Sanganer)", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "base_temp": 34.5, "zone": "Northwest Semi-Arid"},
    {"name": "Nagpur (Sonegaon)", "state": "Maharashtra", "lat": 21.1458, "lon": 79.0882, "base_temp": 33.0, "zone": "Central India"},
    {"name": "Visakhapatnam (CWC)", "state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185, "base_temp": 32.8, "zone": "East Coast / BoB"},
    {"name": "Kochi (CIAL)", "state": "Kerala", "lat": 9.9312, "lon": 76.2673, "base_temp": 30.5, "zone": "Southwest Coast"},
    {"name": "Patna (Aerodrome)", "state": "Bihar", "lat": 25.5941, "lon": 85.1376, "base_temp": 32.0, "zone": "Gangetic Plain"},
    {"name": "Bhopal (Bairagarh)", "state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126, "base_temp": 31.5, "zone": "Central India"},
    {"name": "Pune (Shivajinagar)", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "base_temp": 30.0, "zone": "Western Ghats"},
    {"name": "Dehradun (FRI)", "state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322, "base_temp": 24.5, "zone": "Sub-Himalayan"},
]

DRIVER_NOTES = {
    "pressure_drop": {
        "pos": "Steep 24h barometric fall indicates frontal cyclogenesis, increasing timing and intensity error.",
        "neg": "Steady surface pressure reflects calm synoptic conditions with low forecast drift.",
    },
    "wind_shear": {
        "pos": "Strong vertical shear favors organized convection that medium-range grids often misplace.",
        "neg": "Weak shear keeps flow mostly laminar, which NWP parameterizations capture well.",
    },
    "predicted_rainfall": {
        "pos": "Heavy convective rainfall totals carry high microphysics and spatial placement uncertainty.",
        "neg": "Minimal forecast rainfall avoids convective initiation errors.",
    },
    "predicted_temp": {
        "pos": "Forecasted temperature sits near seasonal extremes where boundary-layer errors widen.",
        "neg": "Temperature remains close to seasonal norms with low model bias.",
    },
}


@st.cache_data(show_spinner=False)
def get_station_forecasts(lead_day: int, seed: int = 101) -> pd.DataFrame:
    """
    Generate synthetic synoptic state for all stations at `lead_day` and run
    a single vectorized batch prediction instead of per-station model calls.
    """
    rng = np.random.default_rng(seed + lead_day * 7)
    lead_dispersion = 1.0 + (lead_day - 1) * 0.14
    rows = []

    for station in INDIAN_MET_STATIONS:
        is_bob = "BoB" in station["zone"] or "East Coast" in station["zone"]
        is_himalayan = "Himalaya" in station["zone"]

        if is_bob and lead_day >= 3:
            pressure_drop = float(rng.uniform(14.0, 24.0) * (lead_day / 5.0))
            wind_shear = float(rng.uniform(28.0, 44.0))
            predicted_rain = float(rng.uniform(22.0, 52.0) * lead_dispersion)
        elif is_himalayan and lead_day >= 4:
            pressure_drop = float(rng.uniform(10.0, 19.0))
            wind_shear = float(rng.uniform(32.0, 48.0))
            predicted_rain = float(rng.uniform(8.0, 26.0))
        else:
            pressure_drop = float(rng.gamma(shape=2.0, scale=3.0) * (1.0 + lead_day * 0.08))
            wind_shear = float(rng.normal(loc=16.0, scale=7.0))
            predicted_rain = float(rng.exponential(scale=6.0) if rng.random() > 0.4 else 0.0)

        rows.append({
            "station_name": station["name"],
            "state": station["state"],
            "zone": station["zone"],
            "lat": station["lat"],
            "lon": station["lon"],
            "predicted_temp": round(station["base_temp"] + float(rng.normal(0, 1.2 * lead_dispersion)), 1),
            "predicted_rainfall": round(max(0.0, predicted_rain), 1),
            "wind_shear": round(float(np.clip(wind_shear, 4.0, 50.0)), 2),
            "pressure_drop": round(float(np.clip(pressure_drop, 0.5, 28.0)), 2),
        })

    df = pd.DataFrame(rows)

    # Vectorized inference over the whole grid in one pass
    bundle = load_model_bundle()
    feature_cols = bundle["feature_names"]
    probs = bundle["model"].predict_proba(df[feature_cols].astype(float))[:, 1]

    df["risk_percentage"] = np.round(probs * 100.0, 2)
    df["risk_level"] = np.where(
        df["risk_percentage"] >= 75.0, "Critical",
        np.where(df["risk_percentage"] >= 50.0, "Elevated",
        np.where(df["risk_percentage"] >= 30.0, "Watch", "Low"))
    )
    return df


@st.cache_data(show_spinner=False)
def get_station_shap(
    lat: float,
    lon: float,
    predicted_temp: float,
    predicted_rainfall: float,
    wind_shear: float,
    pressure_drop: float,
    lead_day: int,
    use_api: bool = False,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Compute SHAP attribution only for the currently inspected station.
    Avoids running TreeExplainer 18 times when only one station is viewed.
    """
    if use_api:
        try:
            payload = {
                "latitude": lat,
                "longitude": lon,
                "predicted_temp": predicted_temp,
                "predicted_rainfall": predicted_rainfall,
                "wind_shear": wind_shear,
                "pressure_drop": pressure_drop,
                "lead_time_hours": lead_day * 24,
            }
            resp = requests.post(f"{API_BASE_URL}/predict_bust_risk", json=payload, timeout=0.6)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "factors": data["explainability_factors"][:top_k],
                    "source": "FastAPI (/predict_bust_risk)",
                }
        except requests.RequestException:
            pass

    bundle = load_model_bundle()
    explainer = bundle["explainer"]
    feature_cols = bundle["feature_names"]

    sample_df = pd.DataFrame([{
        "predicted_temp": predicted_temp,
        "predicted_rainfall": predicted_rainfall,
        "wind_shear": wind_shear,
        "pressure_drop": pressure_drop,
    }])[feature_cols].astype(float)

    raw_shap = explainer.shap_values(sample_df)
    if isinstance(raw_shap, list):
        shap_vec = raw_shap[1][0]
    elif raw_shap.ndim == 3:
        shap_vec = raw_shap[0, :, 1]
    else:
        shap_vec = raw_shap[0]

    factors = []
    for feat, s_val in zip(feature_cols, shap_vec):
        s_float = float(s_val)
        is_pos = s_float > 0
        factors.append({
            "feature": feat,
            "feature_value": round(float(sample_df[feat].iloc[0]), 2),
            "shap_value": round(s_float, 4),
            "impact": "increases_bust_risk" if is_pos else "decreases_bust_risk",
            "description": DRIVER_NOTES.get(feat, {}).get(
                "pos" if is_pos else "neg",
                "Synoptic variable influencing forecast skill.",
            ),
            "_abs": abs(s_float),
        })

    factors.sort(key=lambda item: item["_abs"], reverse=True)
    return {
        "factors": [{k: v for k, v in f.items() if k != "_abs"} for f in factors[:top_k]],
        "source": "In-Process Model Bundle",
    }


# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-kicker">NWP Verification · India</div>
            <div class="sidebar-brand-title">Forecast Bust Monitor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lead_day = st.slider(
        "Day 1 to Day 10 Forecast Horizon",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Medium-range forecast lead time (Day 1 = 24h, Day 10 = 240h)",
    )
    st.caption(f"Active horizon: **Day +{lead_day}** ({lead_day * 24} hours)")

    st.markdown("---")
    nwp_model = st.selectbox(
        "Target NWP System",
        ["ECMWF IFS (0.1°)", "NCMRWF UM (12km)", "NOAA GFS (0.25°)", "IMD GEFS Ensemble"],
        index=0,
    )

    alert_threshold = st.slider(
        "Red Alert Bust Threshold (%)",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
        help="Stations at or above this risk score are marked red on the map.",
    )

    api_live = is_api_reachable()
    st.markdown("---")
    dot_color = PALETTE["sage_green"] if api_live else PALETTE["amber"]
    mode_label = "FastAPI backend online" if api_live else "Standalone model bundle"
    st.markdown(
        f"<span class='mono-meta'><span class='legend-dot' style='background:{dot_color}; margin-right:6px;'></span>{mode_label}</span>",
        unsafe_allow_html=True,
    )


# Header
st.markdown('<div class="eyebrow-tag">Medium-Range NWP Screening</div>', unsafe_allow_html=True)
st.markdown('<h1 class="page-title">Forecast Bust Early Warning</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Pre-dissemination failure screening (&gt;3°C temp or &gt;15mm rain error) across 18 synoptic stations, with SHAP attribution.</p>',
    unsafe_allow_html=True,
)

df_stations = get_station_forecasts(lead_day=lead_day)

# Alert calculations (explicitly exclude high-risk count from elevated count even if threshold < 50)
high_risk_mask = df_stations["risk_percentage"] >= alert_threshold
elevated_mask = (df_stations["risk_percentage"] >= 50.0) & (~high_risk_mask)

high_risk_count = int(high_risk_mask.sum())
elevated_count = int(elevated_mask.sum())
mean_risk = float(df_stations["risk_percentage"].mean())

# Use positional argmax rather than DataFrame label idxmax() so st.selectbox index never breaks
top_pos = int(np.argmax(df_stations["risk_percentage"].values))
top_station = df_stations.iloc[top_pos]

# KPI Strip
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 3px solid {PALETTE['espresso']};">
            <div class="kpi-label">Synoptic Grid</div>
            <div class="kpi-value" style="color:{PALETTE['espresso']};">{len(df_stations)} Stations</div>
            <div class="kpi-sub">Mean bust risk: {mean_risk:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    crit_color = PALETTE["critical_red"] if high_risk_count > 0 else PALETTE["sage_green"]
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 3px solid {crit_color};">
            <div class="kpi-label">High-Risk Bust Zones (&ge;{alert_threshold}%)</div>
            <div class="kpi-value" style="color:{crit_color};">{high_risk_count} Active</div>
            <div class="kpi-sub">Requires forecaster review</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 3px solid {PALETTE['amber']};">
            <div class="kpi-label">Elevated Watch (50&ndash;{alert_threshold}%)</div>
            <div class="kpi-value" style="color:{PALETTE['terracotta']};">{elevated_count} Stations</div>
            <div class="kpi-sub">Monitor convective shifts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    short_name = top_station["station_name"].split(" (")[0]
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 3px solid {PALETTE['terracotta']};">
            <div class="kpi-label">Peak Vulnerability</div>
            <div class="kpi-value" style="color:{PALETTE['critical_red']};">{top_station['risk_percentage']:.1f}%</div>
            <div class="kpi-sub">{short_name} &middot; {top_station['zone']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if high_risk_count > 0:
    st.markdown(
        f"""
        <div class="status-banner banner-crit">
            <strong>Bust Warning (Day +{lead_day}):</strong> {high_risk_count} station(s) exceed the {alert_threshold}% bust threshold under <strong>{nwp_model}</strong>. Elevated chance of large rainfall placement or temperature error.
        </div>
        """,
        unsafe_allow_html=True,
    )
elif elevated_count > 0:
    st.markdown(
        f"""
        <div class="status-banner banner-warn">
            <strong>Watch Status (Day +{lead_day}):</strong> No stations exceed {alert_threshold}%, though {elevated_count} station(s) sit in the 50&ndash;{alert_threshold}% watch band.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="status-banner banner-ok">
            <strong>Nominal Reliability (Day +{lead_day}):</strong> All {len(df_stations)} stations sit below 50% bust probability under <strong>{nwp_model}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# Regional Risk Map
st.subheader("Regional Bust Risk Map")
st.markdown(
    f"""
    <div class="map-legend">
        <span class="legend-item"><span class="legend-dot" style="background:{PALETTE['critical_red']};"></span> High-Risk Bust (&ge;{alert_threshold}%)</span>
        <span class="legend-item"><span class="legend-dot" style="background:{PALETTE['amber']};"></span> Elevated Watch (50&ndash;{alert_threshold}%)</span>
        <span class="legend-item"><span class="legend-dot" style="background:{PALETTE['sage_green']};"></span> Reliable (&lt;50%)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

india_map = folium.Map(
    location=[22.6, 79.2],
    zoom_start=5,
    tiles="OpenStreetMap",
    control_scale=True,
)

# Subtle warm parchment tint on map tiles so OSM blends naturally with #FEF3E2
india_map.get_root().header.add_child(
    folium.Element(
        "<style>.leaflet-tile-pane { filter: sepia(0.24) saturate(0.88) hue-rotate(-6deg) brightness(1.01); }</style>"
    )
)

for _, row in df_stations.iterrows():
    risk = float(row["risk_percentage"])
    if risk >= alert_threshold:
        marker_color = PALETTE["critical_red"]
        status_text = "High Bust Risk"
        radius = 10
    elif risk >= 50.0:
        marker_color = PALETTE["amber"]
        status_text = "Elevated Watch"
        radius = 8
    else:
        marker_color = PALETTE["sage_green"]
        status_text = "Reliable"
        radius = 7

    popup_html = f"""
    <div style="font-family: -apple-system, sans-serif; font-size: 12.5px; width: 225px; background: {PALETTE['cream']}; color: {PALETTE['espresso']}; padding: 8px 10px; border-radius: 6px; border: 1px solid {PALETTE['border']};">
        <div style="font-weight: 700; font-size: 13.5px; color: {PALETTE['espresso']}; margin-bottom: 4px;">{row['station_name']}</div>
        <div style="margin-bottom: 6px;">
            <span style="display:inline-block; background:{marker_color}; color:#FFF9F0; font-size:10px; font-weight:600; padding:2px 7px; border-radius:3px;">{status_text}</span>
            <strong style="float:right; color:{marker_color}; font-size:14px;">{risk:.1f}%</strong>
        </div>
        <div style="border-top: 1px solid {PALETTE['border']}; padding-top: 6px; line-height: 1.55;">
            <b>Forecast Temp:</b> {row['predicted_temp']} °C<br>
            <b>Forecast Rain:</b> {row['predicted_rainfall']} mm<br>
            <b>24h Pressure Drop:</b> {row['pressure_drop']} hPa<br>
            <b>0-6km Wind Shear:</b> {row['wind_shear']} m/s
        </div>
    </div>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"{row['station_name']} — {risk:.1f}% ({status_text})",
        color=PALETTE["espresso"] if risk >= alert_threshold else marker_color,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.88,
        weight=1.5,
    ).add_to(india_map)

# returned_objects=[] avoids triggering a full script rerun on map pan/zoom
st_folium(india_map, use_container_width=True, height=470, returned_objects=[])


# Explainable AI Insights
st.markdown("---")
st.subheader("Explainable AI Insights")
st.caption("Select a station below to inspect the SHAP attribution behind its bust probability.")

station_options = df_stations["station_name"].tolist()
selected_name = st.selectbox(
    "Inspect Station",
    options=station_options,
    index=top_pos,
)

sel = df_stations[df_stations["station_name"] == selected_name].iloc[0]
sel_risk = float(sel["risk_percentage"])

shap_result = get_station_shap(
    lat=float(sel["lat"]),
    lon=float(sel["lon"]),
    predicted_temp=float(sel["predicted_temp"]),
    predicted_rainfall=float(sel["predicted_rainfall"]),
    wind_shear=float(sel["wind_shear"]),
    pressure_drop=float(sel["pressure_drop"]),
    lead_day=lead_day,
    use_api=api_live,
    top_k=3,
)
factors = shap_result["factors"]

col_info, col_chart = st.columns([1, 1.35])

with col_info:
    if sel_risk >= alert_threshold:
        badge_bg = PALETTE["critical_red"]
        verdict = f"High Bust Risk ({sel_risk:.1f}%)"
        advice = "High chance of >3°C temp or >15mm rain error. Manual forecaster override advised."
    elif sel_risk >= 50.0:
        badge_bg = PALETTE["amber"]
        verdict = f"Elevated Watch ({sel_risk:.1f}%)"
        advice = "Column stability is marginal. Cross-check against latest satellite and radar loops."
    else:
        badge_bg = PALETTE["sage_green"]
        verdict = f"Reliable Forecast ({sel_risk:.1f}%)"
        advice = "Synoptic inputs lie within well-behaved regimes with low error variance."

    st.markdown(
        f"""
        <div class="station-panel">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div style="font-size:18px; font-weight:700; color:{PALETTE['espresso']};">{sel['station_name']}</div>
                <span style="background:{badge_bg}; color:#FFF9F0; font-family:'IBM Plex Mono', monospace; font-size:11px; font-weight:600; padding:3px 9px; border-radius:4px;">{verdict}</span>
            </div>
            <div class="mono-meta" style="margin-top:6px;">
                {sel['state']} &middot; {sel['zone']} &middot; {sel['lat']:.2f}°N, {sel['lon']:.2f}°E
            </div>
            <p style="font-size:13.5px; color:{PALETTE['espresso']}; margin:10px 0 4px 0;">{advice}</p>
            <div class="stat-grid">
                <div class="stat-cell">
                    <div class="stat-cell-label">Forecast Temp</div>
                    <div class="stat-cell-val">{sel['predicted_temp']} °C</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-cell-label">Forecast Rain (24h)</div>
                    <div class="stat-cell-val">{sel['predicted_rainfall']} mm</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-cell-label">Wind Shear (0-6km)</div>
                    <div class="stat-cell-val">{sel['wind_shear']} m/s</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-cell-label">Pressure Drop (24h)</div>
                    <div class="stat-cell-val">-{sel['pressure_drop']} hPa</div>
                </div>
            </div>
            <div class="mono-meta" style="margin-top:10px; font-size:11px;">
                Horizon: Day +{lead_day} ({lead_day * 24}h) &middot; Attribution via {shap_result['source']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_chart:
    feat_labels = [f["feature"].replace("_", " ").title() for f in factors]
    shap_vals = [f["shap_value"] for f in factors]
    bar_colors = [PALETTE["terracotta"] if v > 0 else PALETTE["sage_green"] for v in shap_vals]
    hover_text = [
        f"<b>{f['feature'].replace('_', ' ').title()}</b><br>"
        f"Value: {f['feature_value']}<br>"
        f"SHAP: {f['shap_value']:+.4f}<br>"
        f"{f['description']}"
        for f in factors
    ]

    fig = go.Figure(
        go.Bar(
            x=shap_vals,
            y=feat_labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color=PALETTE["espresso"], width=0.8)),
            text=[f"{v:+.3f}" for v in shap_vals],
            textposition="outside",
            textfont=dict(color=PALETTE["espresso"], family="IBM Plex Mono, monospace", size=12),
            hoverinfo="text",
            hovertext=hover_text,
            cliponaxis=False,
        )
    )

    max_abs = max([abs(v) for v in shap_vals] + [0.05]) * 1.32
    fig.update_layout(
        title=dict(
            text=f"Top Root Cause Drivers — {sel['station_name'].split(' (')[0]}",
            font=dict(size=15, color=PALETTE["espresso"], family="DM Sans, sans-serif"),
        ),
        xaxis=dict(
            title=dict(text="SHAP Contribution (Log-Odds)", font=dict(size=12, color=PALETTE["muted_brown"])),
            range=[-max_abs, max_abs],
            zeroline=True,
            zerolinecolor=PALETTE["espresso"],
            zerolinewidth=1.4,
            gridcolor="#EDE0C9",
            tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=11),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color=PALETTE["espresso"], size=12),
        ),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["cream"],
        margin=dict(l=12, r=36, t=46, b=36),
        height=315,
    )

    st.plotly_chart(fig, width="stretch")

# Driver Breakdown Cards
driver_cols = st.columns(len(factors))
for col, factor in zip(driver_cols, factors):
    pushes_risk = factor["impact"] == "increases_bust_risk"
    accent = PALETTE["terracotta"] if pushes_risk else PALETTE["sage_green"]
    pill_bg = f"{PALETTE['terracotta']}1F" if pushes_risk else f"{PALETTE['sage_green']}1F"
    pill_label = "Raises Bust Risk" if pushes_risk else "Stabilizes Forecast"

    with col:
        st.markdown(
            f"""
            <div class="driver-card" style="border-left: 4px solid {accent};">
                <span class="driver-pill" style="background:{pill_bg}; color:{accent};">{pill_label}</span>
                <div style="font-size:16px; font-weight:700; color:{PALETTE['espresso']}; margin:8px 0 4px 0;">
                    {factor['feature'].replace('_', ' ').title()}
                </div>
                <div class="mono-meta">
                    Input: <strong>{factor['feature_value']}</strong> &middot; SHAP: <strong>{factor['shap_value']:+.4f}</strong>
                </div>
                <p style="font-size:13px; color:{PALETTE['muted_brown']}; margin:8px 0 0 0; line-height:1.45;">
                    {factor['description']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Station Table
with st.expander("View All 18 Synoptic Stations"):
    table_df = df_stations[
        [
            "station_name",
            "zone",
            "risk_percentage",
            "risk_level",
            "predicted_temp",
            "predicted_rainfall",
            "pressure_drop",
            "wind_shear",
        ]
    ].sort_values(by="risk_percentage", ascending=False)

    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        column_config={
            "station_name": "Station",
            "zone": "Zone",
            "risk_percentage": st.column_config.NumberColumn("Bust Risk", format="%.1f%%"),
            "risk_level": "Tier",
            "predicted_temp": st.column_config.NumberColumn("Temp", format="%.1f °C"),
            "predicted_rainfall": st.column_config.NumberColumn("Rain", format="%.1f mm"),
            "pressure_drop": st.column_config.NumberColumn("Pressure Drop", format="%.2f hPa"),
            "wind_shear": st.column_config.NumberColumn("Wind Shear", format="%.2f m/s"),
        },
    )
