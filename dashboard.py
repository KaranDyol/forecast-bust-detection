"""
Medium-range weather forecast bust monitor and SHAP diagnostics dashboard.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

from stations_catalog import INDIAN_MET_STATIONS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bust_model.pkl"


def _resolve_api_base_url() -> str:
    env_url = os.getenv("BUST_API_URL")
    if env_url:
        return env_url.rstrip("/")
    try:
        secret_url = st.secrets.get("BUST_API_URL")
        if secret_url:
            return str(secret_url).rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:8000"


API_BASE_URL = _resolve_api_base_url()
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

NWP_MODELS: Dict[str, Dict[str, Any]] = {
    "ECMWF IFS (0.25°)": {"open_meteo_model": "ecmwf_ifs025", "sim_scale": 1.00},
    "NOAA GFS (0.25°)": {"open_meteo_model": "gfs_seamless", "sim_scale": 1.07},
    "Multi-Model Blend (Best Match)": {"open_meteo_model": "best_match", "sim_scale": 0.96},
}

CHART_LABELS: Dict[str, str] = {
    "pressure_drop": "Pressure Drop",
    "wind_shear": "Wind Shear",
    "predicted_rainfall": "Forecast Rain",
    "predicted_temp": "Forecast Temp",
    "lead_day": "Lead Horizon",
    "relative_humidity": "Rel. Humidity",
    "cape": "CAPE Instability",
    "model_temp_spread": "ECMWF-GFS Temp Δ",
    "model_rain_spread": "ECMWF-GFS Rain Δ",
}

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

@keyframes fadeSlideUp {{
    from {{
        opacity: 0;
        transform: translateY(10px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes pulseRing {{
    0% {{
        box-shadow: 0 0 0 0 rgba(194, 59, 22, 0.55);
    }}
    70% {{
        box-shadow: 0 0 0 8px rgba(194, 59, 22, 0);
    }}
    100% {{
        box-shadow: 0 0 0 0 rgba(194, 59, 22, 0);
    }}
}}

@keyframes barGrow {{
    from {{ width: 0%; }}
}}

html, body, [class*="css"] {{
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: {PALETTE['espresso']};
}}

header[data-testid="stHeader"] {{
    background: rgba(254, 243, 226, 0.88);
    backdrop-filter: blur(6px);
}}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    overflow-x: clip;
}}

[data-testid="stSidebar"] {{
    border-right: 1px solid {PALETTE['border']};
}}

.block-container,
[data-testid="stMainBlockContainer"] {{
    padding: clamp(3.9rem, 5vw, 4.6rem) clamp(0.85rem, 2.8vw, 2.2rem) 4.5rem !important;
    max-width: 1260px;
}}

[data-testid="stHorizontalBlock"] {{
    flex-wrap: wrap !important;
    gap: 1rem !important;
}}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
    min-width: 0 !important;
}}

[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stCustomComponentV1"] {{
    max-width: 100% !important;
}}

.section-divider {{
    height: 1px;
    border: none;
    background-color: {PALETTE['border']};
    margin: 0.75rem 0 1.15rem 0;
}}

.header-wrap {{
    animation: fadeSlideUp 0.4s cubic-bezier(0.22, 1, 0.36, 1) both;
}}

.eyebrow-tag {{
    display: inline-block;
    background-color: {PALETTE['amber']}3D;
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
    font-size: clamp(1.55rem, 2.8vw, 2.1rem);
    font-weight: 700;
    letter-spacing: -0.025em;
    color: {PALETTE['espresso']};
    margin: 0 0 4px 0;
    line-height: 1.15;
}}

.page-subtitle {{
    font-size: clamp(0.88rem, 1.3vw, 0.96rem);
    color: {PALETTE['muted_brown']};
    margin: 0 0 14px 0;
    line-height: 1.45;
}}

.header-rule {{
    height: 2px;
    width: 100%;
    background: linear-gradient(90deg, {PALETTE['espresso']} 0%, {PALETTE['terracotta']} 45%, {PALETTE['amber']} 80%, transparent 100%);
    border-radius: 2px;
    margin-bottom: 20px;
    opacity: 0.75;
}}

.sidebar-brand {{
    background-color: {PALETTE['espresso']};
    color: {PALETTE['cream']};
    padding: 14px 16px;
    border-radius: 8px;
    border-bottom: 3px solid {PALETTE['amber']};
    margin-bottom: 18px;
    transition: transform 0.2s ease;
}}

.sidebar-brand:hover {{
    transform: translateY(-1px);
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

/* Responsive 4 -> 2 -> 1 KPI grid */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 14px;
}}

.kpi-card {{
    background-color: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 14px 16px;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    animation: fadeSlideUp 0.42s cubic-bezier(0.22, 1, 0.36, 1) both;
}}

.kpi-card:nth-child(1) {{ animation-delay: 0.03s; }}
.kpi-card:nth-child(2) {{ animation-delay: 0.08s; }}
.kpi-card:nth-child(3) {{ animation-delay: 0.13s; }}
.kpi-card:nth-child(4) {{ animation-delay: 0.18s; }}

.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 22px -8px rgba(42, 26, 14, 0.14);
    border-color: {PALETTE['amber']};
}}

.kpi-label {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.055em;
    text-transform: uppercase;
    color: {PALETTE['muted_brown']};
    margin-bottom: 6px;
}}

.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: clamp(1.2rem, 2vw, 1.52rem);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}}

.kpi-sub {{
    font-size: 12px;
    color: {PALETTE['muted_brown']};
    margin-top: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.pulse-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: {PALETTE['critical_red']};
    display: inline-block;
    animation: pulseRing 1.9s infinite;
    flex-shrink: 0;
}}

.status-banner {{
    border-radius: 10px;
    padding: 12px 16px;
    margin: 4px 0 20px 0;
    font-size: 14px;
    line-height: 1.48;
    color: {PALETTE['espresso']};
    animation: fadeSlideUp 0.46s cubic-bezier(0.22, 1, 0.36, 1) 0.14s both;
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
    gap: 10px 18px;
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
    flex-shrink: 0;
}}

iframe {{
    display: block;
    max-width: 100% !important;
    border-radius: 10px !important;
    border: 1px solid {PALETTE['border']} !important;
    box-shadow: 0 6px 18px -8px rgba(42, 26, 14, 0.12);
}}

.station-panel {{
    background-color: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
    animation: fadeSlideUp 0.42s cubic-bezier(0.22, 1, 0.36, 1) both;
}}

.risk-meter-track {{
    width: 100%;
    height: 6px;
    background-color: {PALETTE['sand']};
    border-radius: 99px;
    overflow: hidden;
    margin: 10px 0 6px 0;
}}

.risk-meter-fill {{
    height: 100%;
    border-radius: 99px;
    animation: barGrow 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
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
    border-radius: 8px;
    padding: 10px 12px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}}

.stat-cell:hover {{
    border-color: {PALETTE['amber']};
    transform: translateY(-1px);
}}

.stat-cell-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {PALETTE['muted_brown']};
}}

.stat-cell-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15.5px;
    font-weight: 600;
    color: {PALETTE['espresso']};
    margin-top: 2px;
}}

/* Responsive 3 -> 2 -> 1 Driver cards grid */
.driver-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 10px;
    margin-bottom: 14px;
}}

.driver-card {{
    background-color: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 14px 16px;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    animation: fadeSlideUp 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
}}

.driver-card:nth-child(1) {{ animation-delay: 0.06s; }}
.driver-card:nth-child(2) {{ animation-delay: 0.12s; }}
.driver-card:nth-child(3) {{ animation-delay: 0.18s; }}

.driver-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 22px -8px rgba(42, 26, 14, 0.14);
    border-color: {PALETTE['amber']};
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
    font-size: 12px;
    color: {PALETTE['muted_brown']};
}}

.kpi-card, .driver-card, .station-panel {{
    min-width: 0;
    overflow-wrap: break-word;
}}

/* Tablet breakpoint: keep 2x2 KPIs, stack columns & driver cards cleanly (avoid 2+1 orphan row) */
@media (max-width: 1024px) {{
    .kpi-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
}}

@media (max-width: 900px) {{
    [data-testid="stHorizontalBlock"] {{
        flex-direction: column !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
        max-width: 100% !important;
    }}
    .driver-grid {{
        grid-template-columns: 1fr;
    }}
}}

/* Mobile breakpoint */
@media (max-width: 640px) {{
    .kpi-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }}
    .kpi-card {{
        padding: 12px;
    }}
    .kpi-sub {{
        white-space: normal;
    }}
    .driver-grid {{
        grid-template-columns: 1fr;
    }}
    [data-baseweb="tab-list"] {{
        flex-wrap: wrap !important;
        overflow-x: visible !important;
    }}
    button[data-baseweb="tab"] {{
        font-size: 12.5px !important;
        padding: 6px 10px !important;
    }}
}}

/* Very small phones */
@media (max-width: 400px) {{
    .kpi-grid, .stat-grid {{
        grid-template-columns: 1fr;
    }}
}}

/* Warm workspace tabs & interactive buttons */
[data-baseweb="tab-list"] {{
    gap: 6px;
    border-bottom: 1px solid {PALETTE['border']};
    overflow-x: auto;
    scrollbar-width: none;
}}

[data-baseweb="tab-list"]::-webkit-scrollbar {{
    display: none;
}}

button[data-baseweb="tab"] {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: {PALETTE['muted_brown']} !important;
    padding: 8px 14px !important;
    border-radius: 8px 8px 0 0 !important;
    white-space: nowrap !important;
    transition: color 0.18s ease, background-color 0.18s ease;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: {PALETTE['espresso']} !important;
    background-color: {PALETTE['surface']} !important;
}}

[data-baseweb="tab-highlight"] {{
    background-color: {PALETTE['terracotta']} !important;
}}

[data-testid="stButton"] > button,
[data-testid="stDownloadButton"] > button {{
    background-color: {PALETTE['surface']};
    color: {PALETTE['espresso']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    font-weight: 600;
    transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
}}

[data-testid="stButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover {{
    border-color: {PALETTE['terracotta']};
    color: {PALETTE['terracotta']};
    transform: translateY(-1px);
}}

.btn-spacer {{
    height: 28px;
}}

@media (max-width: 900px) {{
    .btn-spacer {{
        display: none;
    }}
}}

/* Respect OS accessibility setting */
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_bundle(target_version: str = "v3-multihorizon-spread-10728") -> Dict[str, Any]:
    """Load the serialized model and SHAP TreeExplainer once per worker."""
    import importlib

    data_csv = BASE_DIR / "processed_weather_data.csv"
    bundle: Any = None

    if MODEL_PATH.is_file():
        bundle = joblib.load(MODEL_PATH)
        meta = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
        metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics"), dict) else {}
        if (
            meta.get("pipeline_version") == target_version
            and len(bundle.get("feature_names", [])) >= 9
            and metrics.get("brier_score") is not None
        ):
            return bundle

    if data_csv.is_file():
        import train_model

        importlib.reload(train_model)
        return train_model.train_and_build_bundle(
            csv_path=data_csv,
            save_to_disk=True,
            model_destination=MODEL_PATH,
        )

    if bundle is not None:
        return bundle

    raise FileNotFoundError(
        f"Missing model file at {MODEL_PATH.resolve()}. Run train_model.py first."
    )


@st.cache_data(ttl=60, show_spinner=False)
def is_api_reachable() -> bool:
    """Probe the FastAPI service once every 60s instead of retrying inside per-station loops."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=0.3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner=False)
def ensure_db_initialized(schema_version: str = "v3-multihorizon-spread-10728-r2") -> Optional[Dict[str, Any]]:
    """Initialize SQL tables, hydrate multi-horizon archive rows if needed, and register active model."""
    try:
        import importlib
        import data_pipeline
        from db import bootstrap_database, register_model_run

        summary = bootstrap_database()
        data_csv = BASE_DIR / "processed_weather_data.csv"
        hydrate_error: Optional[str] = None
        if summary.get("forecast_runs", 0) < 5000 and data_csv.is_file():
            try:
                importlib.reload(data_pipeline)
                df_csv = pd.read_csv(data_csv)
                if "station_name" in df_csv.columns and "valid_date" in df_csv.columns:
                    data_pipeline.persist_archive_df_to_db(df_csv)
                    summary = bootstrap_database()
            except Exception as exc:
                hydrate_error = str(exc)

        bundle = load_model_bundle()
        meta = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
        if meta.get("pipeline_version"):
            register_model_run(
                model_version=str(meta["pipeline_version"]),
                champion_name=str(meta.get("champion_name", "RandomForest")),
                training_source=str(meta.get("training_source", "Open-Meteo NWP Archive")),
                train_samples=int(meta.get("sample_count", 0) or 0),
                bust_rate=float(meta.get("bust_rate", 0.0) or 0.0),
                metrics=meta.get("metrics", {}) if isinstance(meta.get("metrics"), dict) else {},
                global_drivers=meta.get("global_drivers", []) if isinstance(meta.get("global_drivers"), list) else [],
                artifact_uri=MODEL_PATH.name,
                set_active=True,
            )
            summary = bootstrap_database()
        if hydrate_error:
            summary["hydration_error"] = hydrate_error
        return summary
    except Exception:
        return None


def read_db_summary_safe() -> Optional[Dict[str, Any]]:
    """Read live DB row counts without blocking on transient connection hiccups."""
    try:
        from db import get_database_summary

        return get_database_summary()
    except Exception:
        return None

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
    "lead_day": {
        "pos": "Extended multi-day forecast lead increases planetary wave and frontal timing drift.",
        "neg": "Short forecast lead anchors the run closely to recent synoptic assimilation.",
    },
    "relative_humidity": {
        "pos": "High boundary-layer moisture increases convective trigger sensitivity and latent heat uncertainty.",
        "neg": "Drier boundary-layer air suppresses moist convective surprise.",
    },
    "cape": {
        "pos": "Elevated CAPE signals explosive convective instability where NWP rain timing degrades sharply.",
        "neg": "Low CAPE indicates stable atmospheric stratification with limited convective surprise.",
    },
    "model_temp_spread": {
        "pos": "Large ECMWF-IFS vs GFS-Seamless temperature divergence reflects competing air-mass boundaries.",
        "neg": "Strong ECMWF-IFS vs GFS-Seamless temperature agreement confirms high synoptic confidence.",
    },
    "model_rain_spread": {
        "pos": "Wide ECMWF-IFS vs GFS-Seamless precipitation divergence highlights uncertain storm track and intensity.",
        "neg": "Close inter-model rainfall agreement supports reliable quantitative precipitation guidance.",
    },
}

DEFAULT_FEATURE_VALUES: Dict[str, float] = {
    "predicted_temp": 30.0,
    "predicted_rainfall": 2.0,
    "wind_shear": 12.0,
    "pressure_drop": 1.0,
    "lead_day": 3.0,
    "relative_humidity": 65.0,
    "cape": 250.0,
    "model_temp_spread": 1.0,
    "model_rain_spread": 2.0,
}


def _align_feature_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Ensure all model feature columns exist in exact order with safe numeric defaults."""
    aligned = df.copy()
    for col in feature_cols:
        default_val = DEFAULT_FEATURE_VALUES.get(col, 0.0)
        if col not in aligned.columns:
            aligned[col] = default_val
        else:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(default_val)
    return aligned[feature_cols].astype(float)


def _score_station_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Run vectorized batch predict_proba passes across overall + temp/rain heads."""
    df = pd.DataFrame(rows)
    bundle = load_model_bundle()
    feature_cols = bundle["feature_names"]
    X = _align_feature_frame(df, feature_cols)
    probs = bundle["model"].predict_proba(X)[:, 1]

    df["risk_percentage"] = np.round(probs * 100.0, 2)
    temp_model = bundle.get("temp_model")
    rain_model = bundle.get("rain_model")
    df["temp_bust_prob"] = (
        np.round(temp_model.predict_proba(X)[:, 1] * 100.0, 1)
        if temp_model is not None
        else np.nan
    )
    df["rain_bust_prob"] = (
        np.round(rain_model.predict_proba(X)[:, 1] * 100.0, 1)
        if rain_model is not None
        else np.nan
    )
    df["risk_level"] = np.where(
        df["risk_percentage"] >= 75.0, "Critical",
        np.where(df["risk_percentage"] >= 50.0, "Elevated",
        np.where(df["risk_percentage"] >= 35.0, "Watch", "Low"))
    )
    return df


def _vector_wind_shear(w10: float, d10: float, w500: float, d500: float) -> float:
    """Compute magnitude of 500 hPa vs 10m bulk wind vector difference (m/s)."""
    u10 = -w10 * math.sin(math.radians(d10))
    v10 = -w10 * math.cos(math.radians(d10))
    u500 = -w500 * math.sin(math.radians(d500))
    v500 = -w500 * math.cos(math.radians(d500))
    return math.hypot(u500 - u10, v500 - v10)


def _first_series(hourly: Dict[str, Any], *keys: str) -> List[Any]:
    """Return the first non-empty list found across candidate Open-Meteo series keys."""
    for key in keys:
        arr = hourly.get(key)
        if isinstance(arr, list) and arr:
            return arr
    return []


def _blend_hourly_pair(arr_a: List[Any], arr_b: List[Any]) -> List[Any]:
    """Element-wise average of two Open-Meteo hourly arrays where both have values."""
    if not arr_a:
        return arr_b
    if not arr_b:
        return arr_a
    out: List[Any] = []
    for a, b in zip(arr_a, arr_b):
        if a is not None and b is not None:
            out.append((float(a) + float(b)) / 2.0)
        else:
            out.append(a if a is not None else b)
    return out


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle surface distance in kilometers between two coordinates."""
    r_earth = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


def _find_nearest_station(lat: float, lon: float) -> Dict[str, Any]:
    """Return the nearest canonical Indian synoptic station and surface distance (km)."""
    best_st = INDIAN_MET_STATIONS[0]
    best_dist = float("inf")
    for st_info in INDIAN_MET_STATIONS:
        dist = _haversine_km(lat, lon, float(st_info["lat"]), float(st_info["lon"]))
        if dist < best_dist:
            best_dist = dist
            best_st = st_info
    return {**best_st, "dist_km": round(best_dist, 1)}


def _extract_location_horizon_rows(
    station_meta: Dict[str, Any],
    loc_data: Dict[str, Any],
    om_model: str,
) -> List[Dict[str, Any]]:
    """Transform a single Open-Meteo multi-model hourly location payload into 10 daily feature rows."""
    hourly = loc_data.get("hourly", {})
    times = hourly.get("time", [])

    ec_t_arr = _first_series(hourly, "temperature_2m_ecmwf_ifs025", "temperature_2m")
    gf_t_arr = _first_series(hourly, "temperature_2m_gfs_seamless", "temperature_2m")
    ec_r_arr = _first_series(hourly, "precipitation_ecmwf_ifs025", "precipitation")
    gf_r_arr = _first_series(hourly, "precipitation_gfs_seamless", "precipitation")

    if om_model == "gfs_seamless":
        temps = gf_t_arr or ec_t_arr
        rains = gf_r_arr or ec_r_arr
        press = _first_series(hourly, "pressure_msl_gfs_seamless", "pressure_msl_ecmwf_ifs025", "pressure_msl")
        w10_arr = _first_series(hourly, "wind_speed_10m_gfs_seamless", "wind_speed_10m_ecmwf_ifs025", "wind_speed_10m")
        d10_arr = _first_series(hourly, "wind_direction_10m_gfs_seamless", "wind_direction_10m_ecmwf_ifs025", "wind_direction_10m")
        w5_arr = _first_series(hourly, "wind_speed_500hPa_gfs_seamless", "wind_speed_500hPa_ecmwf_ifs025", "wind_speed_500hPa")
        d5_arr = _first_series(hourly, "wind_direction_500hPa_gfs_seamless", "wind_direction_500hPa_ecmwf_ifs025", "wind_direction_500hPa")
        rh_arr = _first_series(hourly, "relative_humidity_2m_gfs_seamless", "relative_humidity_2m_ecmwf_ifs025", "relative_humidity_2m")
    elif om_model == "best_match":
        temps = _blend_hourly_pair(ec_t_arr, gf_t_arr)
        rains = _blend_hourly_pair(ec_r_arr, gf_r_arr)
        press = _blend_hourly_pair(
            _first_series(hourly, "pressure_msl_ecmwf_ifs025", "pressure_msl"),
            _first_series(hourly, "pressure_msl_gfs_seamless", "pressure_msl"),
        )
        w10_arr = _first_series(hourly, "wind_speed_10m_ecmwf_ifs025", "wind_speed_10m_gfs_seamless", "wind_speed_10m")
        d10_arr = _first_series(hourly, "wind_direction_10m_ecmwf_ifs025", "wind_direction_10m_gfs_seamless", "wind_direction_10m")
        w5_arr = _first_series(hourly, "wind_speed_500hPa_ecmwf_ifs025", "wind_speed_500hPa_gfs_seamless", "wind_speed_500hPa")
        d5_arr = _first_series(hourly, "wind_direction_500hPa_ecmwf_ifs025", "wind_direction_500hPa_gfs_seamless", "wind_direction_500hPa")
        rh_arr = _blend_hourly_pair(
            _first_series(hourly, "relative_humidity_2m_ecmwf_ifs025", "relative_humidity_2m"),
            _first_series(hourly, "relative_humidity_2m_gfs_seamless", "relative_humidity_2m"),
        )
    else:
        temps = ec_t_arr or gf_t_arr
        rains = ec_r_arr or gf_r_arr
        press = _first_series(hourly, "pressure_msl_ecmwf_ifs025", "pressure_msl_gfs_seamless", "pressure_msl")
        w10_arr = _first_series(hourly, "wind_speed_10m_ecmwf_ifs025", "wind_speed_10m_gfs_seamless", "wind_speed_10m")
        d10_arr = _first_series(hourly, "wind_direction_10m_ecmwf_ifs025", "wind_direction_10m_gfs_seamless", "wind_direction_10m")
        w5_arr = _first_series(hourly, "wind_speed_500hPa_ecmwf_ifs025", "wind_speed_500hPa_gfs_seamless", "wind_speed_500hPa")
        d5_arr = _first_series(hourly, "wind_direction_500hPa_ecmwf_ifs025", "wind_direction_500hPa_gfs_seamless", "wind_direction_500hPa")
        rh_arr = _first_series(hourly, "relative_humidity_2m_ecmwf_ifs025", "relative_humidity_2m_gfs_seamless", "relative_humidity_2m")

    # Keep GFS CAPE primary to stay distribution-aligned with Open-Meteo archive training data
    cape_arr = _first_series(hourly, "cape_gfs_seamless", "cape_ecmwf_ifs025", "cape")
    fallback_temp = float(station_meta.get("base_temp", 30.0))

    rows: List[Dict[str, Any]] = []
    for lead_day in range(1, 11):
        start = lead_day * 24
        end = (lead_day + 1) * 24
        if len(times) < end:
            raise RuntimeError("Incomplete Open-Meteo horizon array")

        day_temps = [float(x) for x in temps[start:end] if x is not None]
        day_rains = [float(x) for x in rains[start:end] if x is not None]
        ec_t_day = [float(x) for x in ec_t_arr[start:end] if x is not None]
        gf_t_day = [float(x) for x in gf_t_arr[start:end] if x is not None]
        ec_r_day = [float(x) for x in ec_r_arr[start:end] if x is not None]
        gf_r_day = [float(x) for x in gf_r_arr[start:end] if x is not None]
        rh_day = [float(x) for x in rh_arr[start:end] if x is not None]
        cape_day = [float(x) for x in cape_arr[start:end] if x is not None]

        # Same-hour 24h pressure tendency removes the ~4 hPa tropical diurnal tide
        drops_24h = [
            float(press[idx - 24]) - float(press[idx])
            for idx in range(start, end)
            if idx < len(press) and press[idx - 24] is not None and press[idx] is not None
        ]
        shears = [
            _vector_wind_shear(
                float(w10_arr[idx]),
                float(d10_arr[idx]),
                float(w5_arr[idx]),
                float(d5_arr[idx]),
            )
            for idx in range(start, end)
            if idx < len(w10_arr) and idx < len(d10_arr) and idx < len(w5_arr) and idx < len(d5_arr)
            and None not in (w10_arr[idx], d10_arr[idx], w5_arr[idx], d5_arr[idx])
        ]

        pred_temp = max(day_temps) if day_temps else fallback_temp
        pred_rain = sum(day_rains) if day_rains else 0.0
        p_drop = max([0.5] + drops_24h)
        shear_85 = float(np.percentile(shears, 85)) if shears else 12.0
        rh_mean = float(np.mean(rh_day)) if rh_day else 65.0
        cape_max = max(cape_day) if cape_day else 0.0
        t_spread = abs(max(ec_t_day) - max(gf_t_day)) if (ec_t_day and gf_t_day) else 0.8
        r_spread = abs(sum(ec_r_day) - sum(gf_r_day)) if (ec_r_day and gf_r_day) else 1.0

        rows.append({
            "lead_day": lead_day,
            "forecast_date": str(times[start])[:10],
            "station_name": station_meta["name"],
            "state": station_meta["state"],
            "zone": station_meta["zone"],
            "lat": float(station_meta["lat"]),
            "lon": float(station_meta["lon"]),
            "predicted_temp": round(pred_temp, 1),
            "predicted_rainfall": round(max(0.0, pred_rain), 1),
            "wind_shear": round(float(np.clip(shear_85, 4.0, 50.0)), 2),
            "pressure_drop": round(float(np.clip(p_drop, 0.5, 30.0)), 2),
            "relative_humidity": round(float(np.clip(rh_mean, 5.0, 100.0)), 1),
            "cape": round(float(np.clip(cape_max, 0.0, 5000.0)), 0),
            "model_temp_spread": round(t_spread, 2),
            "model_rain_spread": round(r_spread, 2),
        })
    return rows


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_open_meteo_grid() -> List[Dict[str, Any]]:
    """
    Fetch an 11-day multi-model (ECMWF-IFS025 + GFS-Seamless) IST forecast batch
    across all 18 Indian stations in a single GET request.
    """
    lats = ",".join(f"{s['lat']:.4f}" for s in INDIAN_MET_STATIONS)
    lons = ",".join(f"{s['lon']:.4f}" for s in INDIAN_MET_STATIONS)
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": (
            "temperature_2m,precipitation,pressure_msl,"
            "wind_speed_10m,wind_direction_10m,wind_speed_500hPa,wind_direction_500hPa,"
            "relative_humidity_2m,cape"
        ),
        "wind_speed_unit": "ms",
        "forecast_days": 11,
        "timezone": "Asia/Kolkata",
        "models": "ecmwf_ifs025,gfs_seamless",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=15.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo returned HTTP {resp.status_code}")
    payload = resp.json()
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or len(payload) != len(INDIAN_MET_STATIONS):
        raise RuntimeError("Unexpected Open-Meteo multi-location payload shape")
    return payload


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_open_meteo_point(lat_round: float, lon_round: float) -> Dict[str, Any]:
    """Fetch an 11-day multi-model forecast series for any clicked coordinate or searched city."""
    params = {
        "latitude": lat_round,
        "longitude": lon_round,
        "hourly": (
            "temperature_2m,precipitation,pressure_msl,"
            "wind_speed_10m,wind_direction_10m,wind_speed_500hPa,wind_direction_500hPa,"
            "relative_humidity_2m,cape"
        ),
        "wind_speed_unit": "ms",
        "forecast_days": 11,
        "timezone": "Asia/Kolkata",
        "models": "ecmwf_ifs025,gfs_seamless",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=12.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo returned HTTP {resp.status_code}")
    payload = resp.json()
    if isinstance(payload, list):
        payload = payload[0]
    if not isinstance(payload, dict) or "hourly" not in payload:
        raise RuntimeError("Unexpected single-point Open-Meteo response")
    return payload


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city_candidates(query: str) -> List[Dict[str, Any]]:
    """Resolve city/town names into coordinates via Open-Meteo Geocoding API (prioritizing India)."""
    clean_q = query.strip()
    if not clean_q:
        return []
    resp = requests.get(
        GEOCODING_URL,
        params={"name": clean_q, "count": 6, "language": "en", "format": "json"},
        timeout=8.0,
    )
    if resp.status_code != 200:
        return []
    results = resp.json().get("results") or []
    candidates: List[Dict[str, Any]] = []
    for item in results:
        name = str(item.get("name", clean_q))
        admin1 = str(item.get("admin1") or "").strip()
        country = str(item.get("country") or "").strip()
        parts = [p for p in [name, admin1, country] if p]
        candidates.append({
            "name": name,
            "admin1": admin1,
            "country": country,
            "country_code": str(item.get("country_code") or "").upper(),
            "label": ", ".join(parts),
            "lat": round(float(item["latitude"]), 4),
            "lon": round(float(item["longitude"]), 4),
            "elevation": item.get("elevation"),
        })
    candidates.sort(key=lambda c: 0 if c["country_code"] == "IN" else 1)
    return candidates


def _build_idw_point_horizon_df(
    lat: float,
    lon: float,
    label: str,
    state: str,
    zone: str,
    grid_df: pd.DataFrame,
) -> pd.DataFrame:
    """Spatially interpolate 18 synoptic station forecasts onto an arbitrary coordinate via IDW."""
    feature_cols = [
        "predicted_temp",
        "predicted_rainfall",
        "wind_shear",
        "pressure_drop",
        "relative_humidity",
        "cape",
        "model_temp_spread",
        "model_rain_spread",
    ]
    rows: List[Dict[str, Any]] = []
    for ld in range(1, 11):
        sub = grid_df[grid_df["lead_day"] == ld]
        if sub.empty:
            continue
        dists = np.array([
            _haversine_km(lat, lon, float(r.lat), float(r.lon))
            for r in sub.itertuples()
        ])
        weights = 1.0 / (dists ** 2 + 900.0)
        weights /= weights.sum()
        blended = {col: float(np.dot(weights, sub[col].astype(float).values)) for col in feature_cols}
        rows.append({
            "lead_day": ld,
            "forecast_date": str(sub["forecast_date"].iloc[0]),
            "station_name": label,
            "state": state,
            "zone": zone,
            "lat": lat,
            "lon": lon,
            "predicted_temp": round(blended["predicted_temp"], 1),
            "predicted_rainfall": round(max(0.0, blended["predicted_rainfall"]), 1),
            "wind_shear": round(float(np.clip(blended["wind_shear"], 4.0, 50.0)), 2),
            "pressure_drop": round(float(np.clip(blended["pressure_drop"], 0.5, 30.0)), 2),
            "relative_humidity": round(float(np.clip(blended["relative_humidity"], 5.0, 100.0)), 1),
            "cape": round(float(np.clip(blended["cape"], 0.0, 5000.0)), 0),
            "model_temp_spread": round(blended["model_temp_spread"], 2),
            "model_rain_spread": round(blended["model_rain_spread"], 2),
        })
    return _score_station_rows(rows)


def get_custom_point_horizon(
    lat: float,
    lon: float,
    label: str,
    data_mode: str,
    nwp_model: str,
    grid_df: pd.DataFrame,
    state_override: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, Optional[float], Dict[str, Any]]:
    """Score a 10-day forecast horizon for any map click or custom city point."""
    nearest = _find_nearest_station(lat, lon)
    state_txt = state_override or f"Near {nearest['state']}"
    zone_txt = f"{nearest['dist_km']:.0f} km from {nearest['name'].split(' (')[0]}"
    om_model = NWP_MODELS.get(nwp_model, {}).get("open_meteo_model", "ecmwf_ifs025")

    if data_mode.startswith("Live"):
        try:
            loc_data = _fetch_open_meteo_point(round(lat, 4), round(lon, 4))
            elev = float(loc_data["elevation"]) if loc_data.get("elevation") is not None else None
            station_meta = {
                "name": label,
                "state": state_txt,
                "zone": zone_txt,
                "lat": lat,
                "lon": lon,
                "base_temp": nearest["base_temp"],
            }
            rows = _extract_location_horizon_rows(station_meta, loc_data, om_model=om_model)
            return _score_station_rows(rows), f"Live Open-Meteo Point ({nwp_model})", elev, nearest
        except Exception:
            fallback_df = _build_idw_point_horizon_df(lat, lon, label, state_txt, zone_txt, grid_df)
            return fallback_df, "Synoptic IDW Spatial Interpolation (Live fallback)", None, nearest

    sim_df = _build_idw_point_horizon_df(lat, lon, label, state_txt, zone_txt, grid_df)
    return sim_df, f"Synoptic IDW Spatial Interpolation ({nwp_model} Sim)", None, nearest


@st.cache_data(ttl=900, show_spinner=False)
def _build_live_horizon_df(om_model: str) -> pd.DataFrame:
    """Transform 11-day multi-model Open-Meteo hourly series into 18x10 ML features + ECMWF/GFS spread."""
    payload = _fetch_open_meteo_grid()
    rows: List[Dict[str, Any]] = []
    elevations: Dict[str, float] = {}

    for station, loc_data in zip(INDIAN_MET_STATIONS, payload):
        if loc_data.get("elevation") is not None:
            elevations[station["name"]] = float(loc_data["elevation"])
        rows.extend(_extract_location_horizon_rows(station, loc_data, om_model=om_model))

    scored_df = _score_station_rows(rows)
    try:
        from db import seed_stations, upsert_forecast_snapshots

        if elevations:
            seed_stations(elevations=elevations)
        bundle_meta = load_model_bundle().get("metadata", {})
        ist_today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
        upsert_forecast_snapshots(
            scored_df.to_dict(orient="records"),
            nwp_model=om_model,
            issued_date=ist_today,
            model_version=str(bundle_meta.get("pipeline_version", "v3-multihorizon-spread-10728")),
        )
    except Exception:
        pass
    return scored_df


@st.cache_data(show_spinner=False)
def _build_sim_horizon_df(nwp_model: str, seed: int = 101) -> pd.DataFrame:
    """Generate deterministic Day+1..Day+10 storm-stress simulation matrix."""
    model_cfg = NWP_MODELS.get(nwp_model, {"sim_scale": 1.0})
    sim_scale = float(model_cfg.get("sim_scale", 1.0))
    model_seed_offset = list(NWP_MODELS.keys()).index(nwp_model) * 19 if nwp_model in NWP_MODELS else 0
    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30)))

    rows: List[Dict[str, Any]] = []
    for lead_day in range(1, 11):
        rng = np.random.default_rng(seed + lead_day * 7 + model_seed_offset)
        lead_dispersion = (1.0 + (lead_day - 1) * 0.14) * sim_scale
        date_str = (ist_now + timedelta(days=lead_day)).strftime("%Y-%m-%d")

        for station in INDIAN_MET_STATIONS:
            is_bob = "BoB" in station["zone"] or "East Coast" in station["zone"]
            is_himalayan = "Himalaya" in station["zone"]

            if is_bob and lead_day >= 3:
                pressure_drop = float(rng.uniform(14.0, 24.0) * (lead_day / 5.0) * sim_scale)
                wind_shear = float(rng.uniform(28.0, 44.0) * sim_scale)
                predicted_rain = float(rng.uniform(22.0, 52.0) * lead_dispersion)
                rel_hum = float(rng.uniform(82.0, 96.0))
                cape_val = float(rng.uniform(1350.0, 2600.0) * sim_scale)
                t_spread = float(rng.uniform(1.8, 3.8) * (1.0 + (lead_day - 3) * 0.08))
                r_spread = float(rng.uniform(14.0, 34.0) * (1.0 + (lead_day - 3) * 0.10))
            elif is_himalayan and lead_day >= 4:
                pressure_drop = float(rng.uniform(10.0, 19.0) * sim_scale)
                wind_shear = float(rng.uniform(32.0, 48.0) * sim_scale)
                predicted_rain = float(rng.uniform(8.0, 26.0) * sim_scale)
                rel_hum = float(rng.uniform(68.0, 88.0))
                cape_val = float(rng.uniform(450.0, 1100.0))
                t_spread = float(rng.uniform(2.2, 4.4) * (1.0 + (lead_day - 4) * 0.07))
                r_spread = float(rng.uniform(8.0, 22.0))
            else:
                pressure_drop = float(rng.gamma(shape=2.0, scale=3.0) * (1.0 + lead_day * 0.08) * sim_scale)
                wind_shear = float(rng.normal(loc=16.0, scale=7.0) * sim_scale)
                predicted_rain = float(rng.exponential(scale=6.0) * sim_scale if rng.random() > 0.4 else 0.0)
                rel_hum = float(rng.uniform(42.0, 76.0))
                cape_val = float(rng.uniform(80.0, 750.0))
                t_spread = float(rng.uniform(0.4, 1.9) * (1.0 + lead_day * 0.08))
                r_spread = float(rng.uniform(0.2, 6.5) * (1.0 + lead_day * 0.10))

            rows.append({
                "lead_day": lead_day,
                "forecast_date": date_str,
                "station_name": station["name"],
                "state": station["state"],
                "zone": station["zone"],
                "lat": station["lat"],
                "lon": station["lon"],
                "predicted_temp": round(station["base_temp"] + float(rng.normal(0, 1.2 * lead_dispersion)), 1),
                "predicted_rainfall": round(max(0.0, predicted_rain), 1),
                "wind_shear": round(float(np.clip(wind_shear, 4.0, 50.0)), 2),
                "pressure_drop": round(float(np.clip(pressure_drop, 0.5, 28.0)), 2),
                "relative_humidity": round(float(np.clip(rel_hum, 15.0, 100.0)), 1),
                "cape": round(float(np.clip(cape_val, 0.0, 4000.0)), 0),
                "model_temp_spread": round(t_spread, 2),
                "model_rain_spread": round(r_spread, 2),
            })

    return _score_station_rows(rows)


def get_horizon_matrix(data_mode: str, nwp_model: str) -> Tuple[pd.DataFrame, str, bool]:
    """Return 18x10 forecast matrix along with effective feed metadata."""
    if data_mode.startswith("Live"):
        om_model = NWP_MODELS.get(nwp_model, {}).get("open_meteo_model", "ecmwf_ifs025")
        try:
            df = _build_live_horizon_df(om_model)
            return df, f"Live Open-Meteo ({nwp_model})", True
        except Exception:
            df = _build_sim_horizon_df(nwp_model)
            return df, "Stress Simulation Fallback (Live feed unavailable)", False

    df = _build_sim_horizon_df(nwp_model)
    return df, f"Synoptic Stress Simulation ({nwp_model})", False


def _compute_local_shap(sample_df: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
    """Run SHAP TreeExplainer for a single-row DataFrame."""
    bundle = load_model_bundle()
    explainer = bundle["explainer"]
    feature_cols = bundle["feature_names"]
    ordered_df = _align_feature_frame(sample_df, feature_cols)

    raw_shap = explainer.shap_values(ordered_df)
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
            "feature_value": round(float(ordered_df[feat].iloc[0]), 2),
            "shap_value": round(s_float, 4),
            "impact": "increases_bust_risk" if is_pos else "decreases_bust_risk",
            "description": DRIVER_NOTES.get(feat, {}).get(
                "pos" if is_pos else "neg",
                "Synoptic variable influencing forecast skill.",
            ),
            "_abs": abs(s_float),
        })

    factors.sort(key=lambda item: item["_abs"], reverse=True)
    return [{k: v for k, v in f.items() if k != "_abs"} for f in factors[:top_k]]


@st.cache_data(show_spinner=False)
def get_station_shap(
    lat: float,
    lon: float,
    predicted_temp: float,
    predicted_rainfall: float,
    wind_shear: float,
    pressure_drop: float,
    lead_day: int,
    relative_humidity: float = 65.0,
    cape: float = 250.0,
    model_temp_spread: float = 1.0,
    model_rain_spread: float = 2.0,
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
                "lead_day": lead_day,
                "lead_time_hours": lead_day * 24,
                "relative_humidity": relative_humidity,
                "cape": cape,
                "model_temp_spread": model_temp_spread,
                "model_rain_spread": model_rain_spread,
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

    sample_df = pd.DataFrame([{
        "predicted_temp": predicted_temp,
        "predicted_rainfall": predicted_rainfall,
        "wind_shear": wind_shear,
        "pressure_drop": pressure_drop,
        "lead_day": lead_day,
        "relative_humidity": relative_humidity,
        "cape": cape,
        "model_temp_spread": model_temp_spread,
        "model_rain_spread": model_rain_spread,
    }])
    return {
        "factors": _compute_local_shap(sample_df, top_k=top_k),
        "source": "In-Process Model Bundle",
    }


@st.cache_data(show_spinner=False)
def evaluate_custom_scenario(
    predicted_temp: float,
    predicted_rainfall: float,
    wind_shear: float,
    pressure_drop: float,
    lead_day: int = 3,
    relative_humidity: float = 65.0,
    cape: float = 250.0,
    model_temp_spread: float = 1.0,
    model_rain_spread: float = 2.0,
) -> Dict[str, Any]:
    """Predict calibrated bust probability + temp/rain split + SHAP attribution for What-If inputs."""
    bundle = load_model_bundle()
    feature_cols = bundle["feature_names"]
    sample_df = pd.DataFrame([{
        "predicted_temp": predicted_temp,
        "predicted_rainfall": predicted_rainfall,
        "wind_shear": wind_shear,
        "pressure_drop": pressure_drop,
        "lead_day": lead_day,
        "relative_humidity": relative_humidity,
        "cape": cape,
        "model_temp_spread": model_temp_spread,
        "model_rain_spread": model_rain_spread,
    }])
    X = _align_feature_frame(sample_df, feature_cols)
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    temp_model = bundle.get("temp_model")
    rain_model = bundle.get("rain_model")
    temp_pct = round(float(temp_model.predict_proba(X)[0, 1]) * 100.0, 1) if temp_model is not None else None
    rain_pct = round(float(rain_model.predict_proba(X)[0, 1]) * 100.0, 1) if rain_model is not None else None
    return {
        "risk_percentage": round(prob * 100.0, 2),
        "temp_bust_prob": temp_pct,
        "rain_bust_prob": rain_pct,
        "factors": _compute_local_shap(sample_df, top_k=3),
    }


def build_shap_figure(factors: List[Dict[str, Any]], title_text: str) -> go.Figure:
    """Create collision-safe horizontal Plotly bar figure for SHAP drivers."""
    feat_labels = [
        CHART_LABELS.get(f["feature"], f["feature"].replace("_", " ").title())
        for f in factors
    ]
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
            textposition="auto",
            insidetextanchor="middle",
            insidetextfont=dict(color="#FFF9F0", family="IBM Plex Mono, monospace", size=11),
            outsidetextfont=dict(color=PALETTE["espresso"], family="IBM Plex Mono, monospace", size=11),
            hoverinfo="text",
            hovertext=hover_text,
        )
    )

    max_abs = max([abs(v) for v in shap_vals] + [0.06]) * 1.48
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.02,
            font=dict(size=14, color=PALETTE["espresso"], family="DM Sans, sans-serif"),
        ),
        xaxis=dict(
            title=dict(text="SHAP Contribution (Log-Odds)", font=dict(size=11.5, color=PALETTE["muted_brown"])),
            range=[-max_abs, max_abs],
            zeroline=True,
            zerolinecolor=PALETTE["espresso"],
            zerolinewidth=1.4,
            gridcolor="#EDE0C9",
            tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=10.5),
        ),
        yaxis=dict(
            autorange="reversed",
            automargin=True,
            tickfont=dict(color=PALETTE["espresso"], size=11.5),
        ),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["cream"],
        margin=dict(l=10, r=20, t=46, b=38),
        height=315,
        transition=dict(duration=320, easing="cubic-in-out"),
    )
    return fig


def build_horizon_figure(
    station_df: pd.DataFrame,
    station_short: str,
    alert_threshold: int,
    active_day: int,
) -> go.Figure:
    """Create Day+1..Day+10 forecast bust trajectory figure for the selected station."""
    ordered = station_df.sort_values("lead_day")
    days = [f"Day +{int(d)}" for d in ordered["lead_day"]]
    risks = ordered["risk_percentage"].tolist()
    hover_text = [
        f"<b>Day +{int(row.lead_day)} ({row.forecast_date})</b><br>"
        f"Bust Risk: <b>{row.risk_percentage:.1f}%</b><br>"
        f"Temp: {row.predicted_temp} °C | Rain: {row.predicted_rainfall} mm<br>"
        f"Shear: {row.wind_shear} m/s | ΔPress: -{row.pressure_drop} hPa<br>"
        f"CAPE: {getattr(row, 'cape', 0.0):.0f} J/kg | "
        f"ECMWF-GFS Δ: {getattr(row, 'model_temp_spread', 0.0):.1f}°C / {getattr(row, 'model_rain_spread', 0.0):.1f}mm"
        for row in ordered.itertuples()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=risks,
            mode="lines+markers",
            line=dict(color=PALETTE["terracotta"], width=2.8, shape="spline"),
            marker=dict(
                size=[11 if int(d) == active_day else 7 for d in ordered["lead_day"]],
                color=[
                    PALETTE["critical_red"] if r >= alert_threshold else PALETTE["amber"]
                    for r in risks
                ],
                line=dict(color=PALETTE["espresso"], width=1.2),
            ),
            fill="tozeroy",
            fillcolor="rgba(212, 98, 26, 0.12)",
            hoverinfo="text",
            hovertext=hover_text,
            name="Bust Probability",
        )
    )
    fig.add_hline(
        y=float(alert_threshold),
        line_dash="dash",
        line_color=PALETTE["critical_red"],
        line_width=1.4,
        annotation_text=f"Alert Threshold ({alert_threshold}%)",
        annotation_position="top right",
        annotation_font=dict(color=PALETTE["critical_red"], size=11, family="IBM Plex Mono, monospace"),
    )
    fig.update_layout(
        title=dict(
            text=f"10-Day Predictability Horizon — {station_short}",
            x=0.01,
            font=dict(size=14, color=PALETTE["espresso"], family="DM Sans, sans-serif"),
        ),
        xaxis=dict(
            gridcolor="#EDE0C9",
            tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=11),
        ),
        yaxis=dict(
            title=dict(text="Bust Risk (%)", font=dict(size=11.5, color=PALETTE["muted_brown"])),
            range=[0, 105],
            gridcolor="#EDE0C9",
            tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=11),
        ),
        showlegend=False,
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["cream"],
        margin=dict(l=12, r=18, t=46, b=32),
        height=275,
        transition=dict(duration=320, easing="cubic-in-out"),
    )
    return fig


def render_station_panel(
    title: str,
    meta_line: str,
    risk_pct: float,
    alert_threshold: int,
    temp: float,
    rainfall: float,
    shear: float,
    pdrop: float,
    footer_meta: str,
    temp_bust_pct: Optional[float] = None,
    rain_bust_pct: Optional[float] = None,
    spread_t: Optional[float] = None,
    spread_r: Optional[float] = None,
    cape: Optional[float] = None,
    rh: Optional[float] = None,
) -> None:
    """Render shared Station / Scenario diagnostic panel with animated risk gauge and param split."""
    if risk_pct >= alert_threshold:
        badge_bg = PALETTE["critical_red"]
        verdict = f"High Bust Risk ({risk_pct:.1f}%)"
        advice = "High chance of >3°C temp or >15mm rain error. Manual forecaster override advised."
    elif risk_pct >= 35.0:
        badge_bg = PALETTE["amber"]
        verdict = f"Elevated Watch ({risk_pct:.1f}%)"
        advice = "Column stability or multi-model agreement is marginal. Cross-check satellite and radar loops."
    else:
        badge_bg = PALETTE["sage_green"]
        verdict = f"Reliable Forecast ({risk_pct:.1f}%)"
        advice = "Synoptic inputs and multi-model consensus lie within well-behaved regimes."

    clamped = min(max(risk_pct, 0.0), 100.0)
    split_badges = ""
    if temp_bust_pct is not None and rain_bust_pct is not None and not (pd.isna(temp_bust_pct) or pd.isna(rain_bust_pct)):
        split_badges = (
            f'<div class="mono-meta" style="display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 2px 0; font-size:11.5px;">'
            f'<span style="background:{PALETTE["sand"]}; padding:3px 8px; border-radius:5px; color:{PALETTE["espresso"]};">'
            f'Temp Bust (&gt;3°C): <strong>{float(temp_bust_pct):.1f}%</strong></span>'
            f'<span style="background:{PALETTE["sand"]}; padding:3px 8px; border-radius:5px; color:{PALETTE["espresso"]};">'
            f'Rain Bust (&gt;15mm): <strong>{float(rain_bust_pct):.1f}%</strong></span>'
            f'</div>'
        )

    extra_cells = ""
    if cape is not None and rh is not None:
        extra_cells += (
            f'<div class="stat-cell">'
            f'<div class="stat-cell-label">CAPE / Rel. Humidity</div>'
            f'<div class="stat-cell-val">{float(cape):.0f} J/kg &middot; {float(rh):.0f}%</div>'
            f'</div>'
        )
    if spread_t is not None and spread_r is not None:
        extra_cells += (
            f'<div class="stat-cell">'
            f'<div class="stat-cell-label">ECMWF-GFS Spread</div>'
            f'<div class="stat-cell-val">&Delta;{float(spread_t):.1f}°C &middot; &Delta;{float(spread_r):.1f}mm</div>'
            f'</div>'
        )

    st.markdown(
        f"""
<div class="station-panel">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
<div style="font-size:18px; font-weight:700; color:{PALETTE['espresso']};">{title}</div>
<span style="background:{badge_bg}; color:#FFF9F0; font-family:'IBM Plex Mono', monospace; font-size:11px; font-weight:600; padding:3px 9px; border-radius:4px;">{verdict}</span>
</div>
<div class="mono-meta" style="margin-top:6px;">{meta_line}</div>
<div class="risk-meter-track" role="progressbar" aria-valuenow="{clamped:.1f}" aria-valuemin="0" aria-valuemax="100">
<div class="risk-meter-fill" style="width: {clamped:.1f}%; background: {badge_bg};"></div>
</div>
<p style="font-size:13.5px; color:{PALETTE['espresso']}; margin:10px 0 4px 0;">{advice}</p>
{split_badges}
<div class="stat-grid">
<div class="stat-cell">
<div class="stat-cell-label">Forecast Temp</div>
<div class="stat-cell-val">{temp} °C</div>
</div>
<div class="stat-cell">
<div class="stat-cell-label">Forecast Rain (24h)</div>
<div class="stat-cell-val">{rainfall} mm</div>
</div>
<div class="stat-cell">
<div class="stat-cell-label">Wind Shear (0-6km)</div>
<div class="stat-cell-val">{shear} m/s</div>
</div>
<div class="stat-cell">
<div class="stat-cell-label">Pressure Drop (24h)</div>
<div class="stat-cell-val">-{pdrop} hPa</div>
</div>
{extra_cells}
</div>
<div class="mono-meta" style="margin-top:10px; font-size:11px;">{footer_meta}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_driver_cards(factors: List[Dict[str, Any]]) -> None:
    """Render responsive 3 -> 1 SHAP root-cause breakdown cards."""
    cards_html: List[str] = []
    for factor in factors:
        pushes_risk = factor["impact"] == "increases_bust_risk"
        accent = PALETTE["terracotta"] if pushes_risk else PALETTE["sage_green"]
        pill_bg = f"{PALETTE['terracotta']}1F" if pushes_risk else f"{PALETTE['sage_green']}1F"
        pill_label = "Raises Bust Risk" if pushes_risk else "Stabilizes Forecast"
        feat_title = factor["feature"].replace("_", " ").title()
        cards_html.append(
            f'<div class="driver-card" style="border-left: 4px solid {accent};">'
            f'<span class="driver-pill" style="background:{pill_bg}; color:{accent};">{pill_label}</span>'
            f'<div style="font-size:16px; font-weight:700; color:{PALETTE["espresso"]}; margin:8px 0 4px 0;">{feat_title}</div>'
            f'<div class="mono-meta">Input: <strong>{factor["feature_value"]}</strong> &middot; SHAP: <strong>{factor["shap_value"]:+.4f}</strong></div>'
            f'<p style="font-size:13px; color:{PALETTE["muted_brown"]}; margin:8px 0 0 0; line-height:1.45;">{factor["description"]}</p>'
            f"</div>"
        )
    st.markdown(f'<div class="driver-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)


@st.cache_data(ttl=120, show_spinner=False)
def load_verified_scorecard_df(cache_nonce: int = 0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load joined forecast-vs-observation pairs from SQL DB and align evaluation splits."""
    import db as db_mod

    _ = cache_nonce
    rows = db_mod.fetch_verified_training_records(limit=25000)
    if hasattr(db_mod, "get_pending_verification_summary"):
        pending_info = db_mod.get_pending_verification_summary()
    else:
        pending_info = {
            "pending_past_pairs": 0,
            "live_forward_runs": 0,
            "live_verified_pairs": 0,
            "min_verified_date": None,
            "max_verified_date": None,
        }
    if not rows:
        return pd.DataFrame(), pending_info

    df = pd.DataFrame(rows)
    numeric_cols = [
        "lead_day",
        "predicted_temp",
        "actual_temp",
        "predicted_rainfall",
        "actual_rainfall",
        "wind_shear",
        "pressure_drop",
        "cape",
        "relative_humidity",
        "model_temp_spread",
        "model_rain_spread",
        "predicted_bust_prob",
        "temp_error",
        "rainfall_error",
        "is_bust",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["valid_date"] = df["valid_date"].astype(str).str.slice(0, 10)
    df["issued_date"] = df["issued_date"].astype(str).str.slice(0, 10)
    df["station_name"] = df["station_name"].astype(str)
    df["climatic_zone"] = df["climatic_zone"].astype(str)
    df["nwp_model"] = df["nwp_model"].astype(str)
    df["is_bust"] = df["is_bust"].fillna(0).astype(int)

    has_saved_prob = df["predicted_bust_prob"].notna()
    df["prob_origin"] = np.where(has_saved_prob, "Issuance Snapshot", "Retrospective Model Score")
    missing_mask = ~has_saved_prob
    if missing_mask.any():
        bundle = load_model_bundle()
        feature_cols = bundle["feature_names"]
        X_missing = _align_feature_frame(df.loc[missing_mask], feature_cols)
        df.loc[missing_mask, "predicted_bust_prob"] = bundle["model"].predict_proba(X_missing)[:, 1]

    df["predicted_bust_prob"] = df["predicted_bust_prob"].clip(0.0, 1.0)
    df["predicted_bust_pct"] = np.round(df["predicted_bust_prob"] * 100.0, 2)
    df["temp_bust_actual"] = (df["temp_error"] > 3.0).astype(int)
    df["rain_bust_actual"] = (df["rainfall_error"] > 15.0).astype(int)
    df["actual_outcome"] = np.where(df["is_bust"] == 1, "BUST", "Reliable")

    bundle_meta = load_model_bundle().get("metadata", {})
    test_range = str(bundle_meta.get("test_date_range") or "")
    holdout_start = test_range.split("..")[0].strip() if ".." in test_range else ""

    is_live_run = df["nwp_model"] != "open_meteo_multihorizon"
    is_holdout = (~is_live_run) & bool(holdout_start) & (df["valid_date"] >= holdout_start)
    df["eval_split"] = np.where(
        is_live_run,
        "Live Forward Run",
        np.where(is_holdout, "Chronological Hold-Out", "Archive Train Window"),
    )
    return df, pending_info


@st.cache_data(ttl=1800, show_spinner=False)
def maybe_auto_verify_past_snapshots(pending_past_pairs: int) -> int:
    """Best-effort Day-0 actuals backfill when past live forecast runs await verification."""
    if pending_past_pairs <= 0:
        return 0
    try:
        import importlib
        import data_pipeline

        importlib.reload(data_pipeline)
        res = data_pipeline.sync_recent_observations_to_db(past_days=14)
        return int(res.get("observation_rows", 0) or 0)
    except Exception:
        return 0


def _compute_scorecard_metrics(fdf: pd.DataFrame, alert_threshold: int) -> Dict[str, Any]:
    """Compute discrimination, calibration, error, and alert-capture metrics on verified rows."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    n_rows = int(len(fdf))
    if n_rows == 0:
        return {"n_rows": 0}

    y_true = fdf["is_bust"].astype(int).values
    y_prob = np.clip(fdf["predicted_bust_prob"].astype(float).values, 0.0, 1.0)
    n_busts = int(y_true.sum())
    obs_bust_rate = (n_busts / n_rows) * 100.0
    temp_mae = float(fdf["temp_error"].mean())
    rain_mae = float(fdf["rainfall_error"].mean())
    temp_busts = int(fdf["temp_bust_actual"].sum())
    rain_busts = int(fdf["rain_bust_actual"].sum())

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else None
    brier = float(brier_score_loss(y_true, y_prob))

    alerted = (y_prob * 100.0) >= float(alert_threshold)
    flagged_n = int(alerted.sum())
    tp_n = int((alerted & (y_true == 1)).sum())
    precision = (tp_n / flagged_n * 100.0) if flagged_n > 0 else 0.0
    recall = (tp_n / n_busts * 100.0) if n_busts > 0 else 0.0

    return {
        "n_rows": n_rows,
        "n_busts": n_busts,
        "obs_bust_rate": obs_bust_rate,
        "temp_mae": temp_mae,
        "rain_mae": rain_mae,
        "temp_busts": temp_busts,
        "rain_busts": rain_busts,
        "roc_auc": roc_auc,
        "brier": brier,
        "flagged_n": flagged_n,
        "tp_n": tp_n,
        "precision": precision,
        "recall": recall,
    }


def build_lead_decay_figure(fdf: pd.DataFrame) -> go.Figure:
    """Compare Observed Bust Rate vs Model Predicted Risk across forecast lead horizons."""
    grouped = (
        fdf.groupby("lead_day", as_index=False)
        .agg(
            n=("is_bust", "count"),
            obs_rate=("is_bust", lambda s: float(s.mean()) * 100.0),
            pred_rate=("predicted_bust_pct", "mean"),
            temp_mae=("temp_error", "mean"),
            rain_mae=("rainfall_error", "mean"),
        )
        .sort_values("lead_day")
    )
    x_labels = [f"Day +{int(d)}" for d in grouped["lead_day"]]
    hover_obs = [
        f"<b>Day +{int(r.lead_day)}</b><br>"
        f"Observed Bust Rate: <b>{r.obs_rate:.1f}%</b> ({int(r.n):,} pairs)<br>"
        f"Temp MAE: {r.temp_mae:.2f} °C · Rain MAE: {r.rain_mae:.1f} mm"
        for r in grouped.itertuples()
    ]
    hover_pred = [
        f"<b>Day +{int(r.lead_day)}</b><br>"
        f"Mean Predicted Risk: <b>{r.pred_rate:.1f}%</b> ({int(r.n):,} pairs)"
        for r in grouped.itertuples()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=grouped["obs_rate"],
            name="Observed Bust Rate (%)",
            marker=dict(color=PALETTE["terracotta"], line=dict(color=PALETTE["espresso"], width=0.8)),
            text=[f"{v:.1f}%" for v in grouped["obs_rate"]],
            textposition="outside",
            hoverinfo="text",
            hovertext=hover_obs,
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=grouped["pred_rate"],
            name="Mean Predicted Risk (%)",
            marker=dict(color=PALETTE["amber"], line=dict(color=PALETTE["espresso"], width=0.8)),
            text=[f"{v:.1f}%" for v in grouped["pred_rate"]],
            textposition="outside",
            hoverinfo="text",
            hovertext=hover_pred,
        )
    )
    y_max = max(list(grouped["obs_rate"]) + list(grouped["pred_rate"]) + [25.0]) * 1.28
    fig.update_layout(
        title=dict(
            text="Lead-Time Error & Bust Decay by Horizon",
            x=0.01,
            font=dict(size=14, color=PALETTE["espresso"], family="DM Sans, sans-serif"),
        ),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, font=dict(size=11.5, color=PALETTE["espresso"])),
        xaxis=dict(gridcolor="#EDE0C9", tickfont=dict(color=PALETTE["espresso"], family="IBM Plex Mono, monospace", size=11)),
        yaxis=dict(
            title=dict(text="Rate (%)", font=dict(size=11.5, color=PALETTE["muted_brown"])),
            range=[0, min(105.0, y_max)],
            gridcolor="#EDE0C9",
            tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=10.5),
        ),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["cream"],
        margin=dict(l=12, r=16, t=68, b=34),
        height=310,
    )
    return fig


def build_calibration_figure(fdf: pd.DataFrame) -> go.Figure:
    """Plot reliability diagram comparing predicted bust risk bins against observed bust frequency."""
    bins = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.1]
    work = fdf[["predicted_bust_pct", "is_bust"]].copy()
    work["bin"] = pd.cut(work["predicted_bust_pct"], bins=bins, include_lowest=True, right=False)
    cal = (
        work.groupby("bin", observed=True, as_index=False)
        .agg(
            mean_pred=("predicted_bust_pct", "mean"),
            obs_freq=("is_bust", lambda s: float(s.mean()) * 100.0),
            n=("is_bust", "count"),
        )
        .dropna()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            line=dict(color=PALETTE["muted_brown"], dash="dash", width=1.4),
            name="Perfect Calibration",
            hoverinfo="skip",
        )
    )
    if not cal.empty:
        sizes = [float(np.clip(8.0 + math.sqrt(n) * 0.65, 9.0, 24.0)) for n in cal["n"]]
        hover_cal = [
            f"<b>Predicted Mean: {r.mean_pred:.1f}%</b><br>"
            f"Observed Bust Rate: <b>{r.obs_freq:.1f}%</b><br>"
            f"Sample Count: {int(r.n):,}"
            for r in cal.itertuples()
        ]
        fig.add_trace(
            go.Scatter(
                x=cal["mean_pred"],
                y=cal["obs_freq"],
                mode="lines+markers",
                line=dict(color=PALETTE["terracotta"], width=2.8),
                marker=dict(size=sizes, color=PALETTE["terracotta"], line=dict(color=PALETTE["espresso"], width=1.2)),
                name="Observed vs Predicted",
                hoverinfo="text",
                hovertext=hover_cal,
            )
        )

    fig.update_layout(
        title=dict(
            text="Probability Calibration Curve (Reliability)",
            x=0.01,
            font=dict(size=14, color=PALETTE["espresso"], family="DM Sans, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, font=dict(size=11.5, color=PALETTE["espresso"])),
        xaxis=dict(
            title=dict(text="Mean Predicted Risk (%)", font=dict(size=11.5, color=PALETTE["muted_brown"])),
            range=[-2, 102],
            gridcolor="#EDE0C9",
            tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=10.5),
        ),
        yaxis=dict(
            title=dict(text="Observed Bust Rate (%)", font=dict(size=11.5, color=PALETTE["muted_brown"])),
            range=[-2, 102],
            gridcolor="#EDE0C9",
            tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=10.5),
        ),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["cream"],
        margin=dict(l=12, r=16, t=68, b=36),
        height=310,
    )
    return fig


def build_station_scorecard_figure(fdf: pd.DataFrame) -> go.Figure:
    """Rank top synoptic stations or climatic regimes by observed bust frequency."""
    work = fdf.copy()
    work["short_station"] = work["station_name"].str.split(" (").str[0]
    grp = (
        work.groupby("short_station", as_index=False)
        .agg(
            n=("is_bust", "count"),
            obs_rate=("is_bust", lambda s: float(s.mean()) * 100.0),
            pred_rate=("predicted_bust_pct", "mean"),
            temp_mae=("temp_error", "mean"),
            rain_mae=("rainfall_error", "mean"),
        )
        .sort_values("obs_rate", ascending=False)
        .head(10)
    )
    hover_txt = [
        f"<b>{r.short_station}</b><br>"
        f"Observed Busts: <b>{r.obs_rate:.1f}%</b> ({int(r.n):,} runs)<br>"
        f"Predicted Mean: {r.pred_rate:.1f}%<br>"
        f"MAE: {r.temp_mae:.2f}°C · {r.rain_mae:.1f}mm"
        for r in grp.itertuples()
    ]
    x_max = max(list(grp["obs_rate"]) + list(grp["pred_rate"]) + [25.0]) * 1.24
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=grp["short_station"],
            x=grp["obs_rate"],
            orientation="h",
            name="Observed Bust %",
            marker=dict(color=PALETTE["critical_red"], line=dict(color=PALETTE["espresso"], width=0.8)),
            text=[f"{v:.1f}%" for v in grp["obs_rate"]],
            textposition="outside",
            hoverinfo="text",
            hovertext=hover_txt,
        )
    )
    fig.add_trace(
        go.Bar(
            y=grp["short_station"],
            x=grp["pred_rate"],
            orientation="h",
            name="Predicted Mean %",
            marker=dict(color=PALETTE["amber"], line=dict(color=PALETTE["espresso"], width=0.7)),
            hoverinfo="text",
            hovertext=hover_txt,
        )
    )
    fig.update_layout(
        title=dict(
            text="Highest-Vulnerability Stations (Top 10)",
            x=0.01,
            font=dict(size=14, color=PALETTE["espresso"], family="DM Sans, sans-serif"),
        ),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, font=dict(size=11, color=PALETTE["espresso"])),
        xaxis=dict(
            title=dict(text="Bust Rate (%)", font=dict(size=11, color=PALETTE["muted_brown"])),
            range=[0, min(105.0, x_max)],
            gridcolor="#EDE0C9",
            tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=10),
        ),
        yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(color=PALETTE["espresso"], size=11)),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["cream"],
        margin=dict(l=10, r=20, t=68, b=34),
        height=340,
    )
    return fig


def build_verification_timeline_figure(fdf: pd.DataFrame) -> go.Figure:
    """Plot chronological daily observed bust rates vs model predicted bust risk."""
    daily = (
        fdf.groupby("valid_date", as_index=False)
        .agg(
            n=("is_bust", "count"),
            busts=("is_bust", "sum"),
            obs_rate=("is_bust", lambda s: float(s.mean()) * 100.0),
            pred_rate=("predicted_bust_pct", "mean"),
        )
        .sort_values("valid_date")
    )
    window = 7 if len(daily) >= 14 else 1
    daily["obs_roll"] = daily["obs_rate"].rolling(window, min_periods=1).mean()
    daily["pred_roll"] = daily["pred_rate"].rolling(window, min_periods=1).mean()
    roll_label = f"{window}d Rolling" if window > 1 else "Daily"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["valid_date"],
            y=daily["obs_rate"],
            name="Daily Bust Rate (%)",
            marker=dict(color="rgba(212, 98, 26, 0.22)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["valid_date"],
            y=daily["obs_roll"],
            mode="lines",
            name=f"Observed ({roll_label} %)",
            line=dict(color=PALETTE["critical_red"], width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["valid_date"],
            y=daily["pred_roll"],
            mode="lines",
            name=f"Predicted Risk ({roll_label} %)",
            line=dict(color=PALETTE["sage_green"], width=2.2, dash="dot"),
        )
    )
    fig.update_layout(
        title=dict(
            text="Verification Timeline: Predicted Risk vs Observed Busts",
            x=0.01,
            font=dict(size=14, color=PALETTE["espresso"], family="DM Sans, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, font=dict(size=11, color=PALETTE["espresso"])),
        xaxis=dict(gridcolor="#EDE0C9", tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=10)),
        yaxis=dict(
            title=dict(text="Bust Rate (%)", font=dict(size=11, color=PALETTE["muted_brown"])),
            range=[0, 105],
            gridcolor="#EDE0C9",
            tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=10),
        ),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["cream"],
        margin=dict(l=12, r=16, t=68, b=34),
        height=340,
    )
    return fig


def render_scorecard_tab(alert_threshold: int) -> None:
    """Render the live & archive Forecast-vs-Observed Verification Scorecard."""
    st.session_state.setdefault("scorecard_nonce", 0)
    score_df, pending_info = load_verified_scorecard_df(cache_nonce=int(st.session_state["scorecard_nonce"]))

    if pending_info.get("pending_past_pairs", 0) > 0:
        auto_synced = maybe_auto_verify_past_snapshots(int(pending_info["pending_past_pairs"]))
        if auto_synced > 0:
            load_verified_scorecard_df.clear()
            st.session_state["scorecard_nonce"] += 1
            score_df, pending_info = load_verified_scorecard_df(cache_nonce=int(st.session_state["scorecard_nonce"]))

    if score_df.empty:
        st.info(
            "No paired verification records found in the SQL database yet. Click 'Sync 150-Day Paired Archive to DB' "
            "in the Bulletin & Model tab or trigger a Day-0 observation sync below."
        )
        if st.button("Fetch Latest Day-0 Observations Now", key="sc_empty_sync"):
            import importlib
            import data_pipeline

            importlib.reload(data_pipeline)
            res = data_pipeline.sync_recent_observations_to_db(past_days=14)
            load_verified_scorecard_df.clear()
            st.session_state["scorecard_nonce"] += 1
            st.success(f"Synced {res['observation_rows']} Day-0 observation rows.")
            st.rerun()
        return

    date_min = pending_info.get("min_verified_date") or score_df["valid_date"].min()
    date_max = pending_info.get("max_verified_date") or score_df["valid_date"].max()
    live_ver_n = int(pending_info.get("live_verified_pairs", 0) or 0)
    pending_n = int(pending_info.get("pending_past_pairs", 0) or 0)
    st.caption(
        f"Verified window: **{date_min}** to **{date_max}** · Ground-truth bust threshold: **>3.0 °C Temp Error** or "
        f"**>15.0 mm 24h Rain Error** · Live forward verified: **{live_ver_n:,}** (pending past runs: **{pending_n:,}**)."
    )

    has_holdout = bool((score_df["eval_split"] == "Chronological Hold-Out").any())
    has_live = bool((score_df["eval_split"] == "Live Forward Run").any())
    scope_options = ["All Verified Records (Archive + Live)"]
    if has_holdout:
        scope_options.insert(0, "Chronological Hold-Out (Out-of-Sample)")
    if has_live:
        scope_options.append("Live Forward NWP Runs Only")

    sc_f1, sc_f2, sc_f3, sc_f4 = st.columns([1.45, 1.25, 0.9, 0.9])
    with sc_f1:
        scope_choice = st.selectbox("Evaluation Scope", options=scope_options, index=0, key="sc_scope")
    with sc_f2:
        station_opts = ["All 18 Stations"] + sorted(score_df["station_name"].unique().tolist())
        station_choice = st.selectbox("Station Filter", options=station_opts, index=0, key="sc_station")
    with sc_f3:
        lead_vals = sorted({int(x) for x in score_df["lead_day"].dropna().unique()})
        lead_opts = ["All Leads"] + [f"Day +{d}" for d in lead_vals]
        lead_choice = st.selectbox("Lead Horizon", options=lead_opts, index=0, key="sc_lead")
    with sc_f4:
        st.markdown("<div class='btn-spacer'></div>", unsafe_allow_html=True)
        if st.button("Sync Day-0 Actuals", width="stretch", key="sc_sync_btn"):
            try:
                with st.spinner("Fetching Day-0 verified actuals from Open-Meteo..."):
                    import importlib
                    import data_pipeline

                    importlib.reload(data_pipeline)
                    sync_res = data_pipeline.sync_recent_observations_to_db(past_days=14)
                load_verified_scorecard_df.clear()
                st.session_state["scorecard_nonce"] += 1
                st.success(
                    f"Synced {sync_res['observation_rows']:,} Day-0 rows through {sync_res.get('latest_obs_date') or 'today'}."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Day-0 observation sync failed: {exc}")

    fdf = score_df.copy()
    if scope_choice.startswith("Chronological"):
        fdf = fdf[fdf["eval_split"] == "Chronological Hold-Out"]
    elif scope_choice.startswith("Live Forward"):
        fdf = fdf[fdf["eval_split"] == "Live Forward Run"]
    if station_choice != "All 18 Stations":
        fdf = fdf[fdf["station_name"] == station_choice]
    if lead_choice != "All Leads":
        target_ld = int(lead_choice.replace("Day +", "").strip())
        fdf = fdf[fdf["lead_day"] == target_ld]

    if fdf.empty:
        st.warning("No verified records match the active filters. Try widening the scope or lead filter.")
        return

    m = _compute_scorecard_metrics(fdf, alert_threshold=alert_threshold)
    auc_str = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "N/A"
    scope_short = "Out-of-Sample" if scope_choice.startswith("Chronological") else (
        "Live Runs" if scope_choice.startswith("Live") else "Full Verified DB"
    )

    st.markdown(
        f"""
<div class="kpi-grid" style="margin-top:6px;">
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['espresso']};">
    <div class="kpi-label">Verified Forecasts</div>
    <div class="kpi-value" style="color:{PALETTE['espresso']};">{m['n_rows']:,}</div>
    <div class="kpi-sub">{m['n_busts']:,} actual busts ({m['temp_busts']:,} temp &middot; {m['rain_busts']:,} rain)</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['terracotta']};">
    <div class="kpi-label">Observed Bust Rate</div>
    <div class="kpi-value" style="color:{PALETTE['terracotta']};">{m['obs_bust_rate']:.1f}%</div>
    <div class="kpi-sub">MAE: {m['temp_mae']:.2f}°C temp &middot; {m['rain_mae']:.1f}mm rain</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['amber']};">
    <div class="kpi-label">ROC-AUC &middot; Brier Score</div>
    <div class="kpi-value" style="color:{PALETTE['espresso']};">{auc_str}</div>
    <div class="kpi-sub">Brier {m['brier']:.4f} &middot; {scope_short}</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['sage_green']};">
    <div class="kpi-label">Alert Capture (&ge;{alert_threshold}%)</div>
    <div class="kpi-value" style="color:{PALETTE['sage_green']};">{m['precision']:.0f}% / {m['recall']:.0f}%</div>
    <div class="kpi-sub">Prec/Rec &middot; {m['flagged_n']:,} alerts &rarr; {m['tp_n']:,} busts caught</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    sc_c1, sc_c2 = st.columns(2)
    with sc_c1:
        st.plotly_chart(build_lead_decay_figure(fdf), width="stretch", config={"displayModeBar": False})
    with sc_c2:
        st.plotly_chart(build_calibration_figure(fdf), width="stretch", config={"displayModeBar": False})

    sc_c3, sc_c4 = st.columns(2)
    with sc_c3:
        st.plotly_chart(build_station_scorecard_figure(fdf), width="stretch", config={"displayModeBar": False})
    with sc_c4:
        st.plotly_chart(build_verification_timeline_figure(fdf), width="stretch", config={"displayModeBar": False})

    st.markdown("**Verified Forecast-vs-Observation Audit Log (Most Recent)**")
    audit_cols = [
        "valid_date",
        "station_name",
        "nwp_model",
        "lead_day",
        "eval_split",
        "predicted_temp",
        "actual_temp",
        "temp_error",
        "predicted_rainfall",
        "actual_rainfall",
        "rainfall_error",
        "predicted_bust_pct",
        "actual_outcome",
    ]
    audit_df = fdf[[c for c in audit_cols if c in fdf.columns]].head(400)
    sc_dl1, sc_dl2 = st.columns([2.2, 1])
    with sc_dl1:
        st.caption(f"Showing top {len(audit_df):,} most recent pairs of {len(fdf):,} filtered verified forecasts.")
    with sc_dl2:
        sc_csv = fdf[[c for c in audit_cols if c in fdf.columns]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scorecard (CSV)",
            data=sc_csv,
            file_name=f"verification_scorecard_{scope_short.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            width="stretch",
            key="sc_csv_dl",
        )

    st.dataframe(
        audit_df,
        width="stretch",
        hide_index=True,
        column_config={
            "valid_date": st.column_config.TextColumn("Valid Date", width="small"),
            "station_name": st.column_config.TextColumn("Station", width="medium"),
            "nwp_model": st.column_config.TextColumn("Feed", width="small"),
            "lead_day": st.column_config.NumberColumn("Lead", format="Day +%d", width="small"),
            "eval_split": st.column_config.TextColumn("Split", width="small"),
            "predicted_temp": st.column_config.NumberColumn("Pred T", format="%.1f °C", width="small"),
            "actual_temp": st.column_config.NumberColumn("Obs T", format="%.1f °C", width="small"),
            "temp_error": st.column_config.NumberColumn("|ΔT|", format="%.2f °C", width="small"),
            "predicted_rainfall": st.column_config.NumberColumn("Pred R", format="%.1f mm", width="small"),
            "actual_rainfall": st.column_config.NumberColumn("Obs R", format="%.1f mm", width="small"),
            "rainfall_error": st.column_config.NumberColumn("|ΔR|", format="%.1f mm", width="small"),
            "predicted_bust_pct": st.column_config.NumberColumn("Bust Risk", format="%.1f%%", width="small"),
            "actual_outcome": st.column_config.TextColumn("Outcome", width="small"),
        },
    )


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

    data_mode = st.radio(
        "Forecast Feed",
        ["Live NWP (Open-Meteo)", "Synoptic Stress Simulation"],
        index=0,
        help="Live mode fetches real 10-day global NWP data for India. Simulation stresses coastal and Himalayan stations for bust drills.",
    )

    nwp_model = st.selectbox(
        "Target NWP System",
        list(NWP_MODELS.keys()),
        index=0,
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

    alert_threshold = st.slider(
        "Red Alert Bust Threshold (%)",
        min_value=40,
        max_value=90,
        value=60,
        step=5,
        help="Stations at or above this calibrated probability are marked red on the map.",
    )

    db_summary = ensure_db_initialized()
    horizon_df, feed_label, is_live_feed = get_horizon_matrix(data_mode=data_mode, nwp_model=nwp_model)
    api_live = is_api_reachable()
    st.markdown("---")
    feed_dot = PALETTE["sage_green"] if is_live_feed else PALETTE["amber"]
    api_dot = PALETTE["sage_green"] if api_live else PALETTE["amber"]
    api_label = "FastAPI backend online" if api_live else "In-process ML bundle"
    db_dot = PALETTE["sage_green"] if db_summary else PALETTE["amber"]
    db_label = f"{db_summary['backend']} · {db_summary['location']}" if db_summary else "DB unavailable"
    st.markdown(
        f"<div class='mono-meta' style='line-height:1.7;'>"
        f"<span class='legend-dot' style='background:{feed_dot}; margin-right:6px;'></span>{feed_label}<br>"
        f"<span class='legend-dot' style='background:{api_dot}; margin-right:6px;'></span>{api_label}<br>"
        f"<span class='legend-dot' style='background:{db_dot}; margin-right:6px;'></span>{db_label}"
        f"</div>",
        unsafe_allow_html=True,
    )


# Header
st.markdown(
    """
<div class="header-wrap">
  <div class="eyebrow-tag">Medium-Range NWP Screening</div>
  <h1 class="page-title">Forecast Bust Early Warning</h1>
  <p class="page-subtitle">Pre-dissemination failure screening (&gt;3°C temp or &gt;15mm rain error) across 18 synoptic stations, with SHAP attribution.</p>
  <div class="header-rule"></div>
</div>
""",
    unsafe_allow_html=True,
)

df_stations = horizon_df[horizon_df["lead_day"] == lead_day].reset_index(drop=True)
active_date = str(df_stations["forecast_date"].iloc[0])

# Alert calculations (explicitly exclude high-risk count from elevated count even if threshold <= 35)
high_risk_mask = df_stations["risk_percentage"] >= alert_threshold
elevated_mask = (df_stations["risk_percentage"] >= 35.0) & (~high_risk_mask)

high_risk_count = int(high_risk_mask.sum())
elevated_count = int(elevated_mask.sum())
mean_risk = float(df_stations["risk_percentage"].mean())

# Use positional argmax rather than DataFrame label idxmax() so st.selectbox index never breaks
top_pos = int(np.argmax(df_stations["risk_percentage"].values))
top_station = df_stations.iloc[top_pos]

# Responsive CSS Grid KPI Strip (4 cols desktop -> 2x2 tablet/mobile -> 1 col micro)
crit_color = PALETTE["critical_red"] if high_risk_count > 0 else PALETTE["sage_green"]
pulse_html = '<span class="pulse-dot"></span>' if high_risk_count > 0 else ""
short_name = top_station["station_name"].split(" (")[0]

st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['espresso']};">
    <div class="kpi-label">Synoptic Grid</div>
    <div class="kpi-value" style="color:{PALETTE['espresso']};">{len(df_stations)} Stations</div>
    <div class="kpi-sub">Mean bust risk: {mean_risk:.1f}%</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {crit_color};">
    <div class="kpi-label">{pulse_html}High-Risk Bust Zones (&ge;{alert_threshold}%)</div>
    <div class="kpi-value" style="color:{crit_color};">{high_risk_count} Active</div>
    <div class="kpi-sub">Requires forecaster review</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['amber']};">
    <div class="kpi-label">Elevated Watch (35&ndash;{alert_threshold}%)</div>
    <div class="kpi-value" style="color:{PALETTE['terracotta']};">{elevated_count} Stations</div>
    <div class="kpi-sub">Monitor convective shifts</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['terracotta']};">
    <div class="kpi-label">Peak Vulnerability</div>
    <div class="kpi-value" style="color:{PALETTE['critical_red']};">{top_station['risk_percentage']:.1f}%</div>
    <div class="kpi-sub">{short_name} &middot; {top_station['zone']}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if high_risk_count > 0:
    st.markdown(
        f"""
        <div class="status-banner banner-crit">
            <strong>Bust Warning (Day +{lead_day} &middot; {active_date}):</strong> {high_risk_count} station(s) exceed the {alert_threshold}% bust threshold via <strong>{feed_label}</strong>. Elevated chance of large rainfall placement or temperature error.
        </div>
        """,
        unsafe_allow_html=True,
    )
elif elevated_count > 0:
    st.markdown(
        f"""
        <div class="status-banner banner-warn">
            <strong>Watch Status (Day +{lead_day} &middot; {active_date}):</strong> No stations exceed {alert_threshold}% via <strong>{feed_label}</strong>, though {elevated_count} station(s) sit in the 35&ndash;{alert_threshold}% watch band.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="status-banner banner-ok">
            <strong>Nominal Reliability (Day +{lead_day} &middot; {active_date}):</strong> All {len(df_stations)} stations sit below 35% calibrated bust probability via <strong>{feed_label}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _set_custom_point(
    lat: float,
    lon: float,
    label: str,
    state: Optional[str] = None,
    switch_to_custom: bool = True,
) -> Dict[str, Any]:
    """Update active custom map/city inspection point and sync coordinate inputs."""
    cp = {
        "lat": round(float(lat), 4),
        "lon": round(float(lon), 4),
        "label": str(label),
        "state": state,
    }
    st.session_state["custom_point"] = cp
    st.session_state["custom_lat_box"] = cp["lat"]
    st.session_state["custom_lon_box"] = cp["lon"]
    if switch_to_custom:
        st.session_state["diag_target_mode"] = "Custom Map Pin / City"
    return cp


def _clear_custom_pin() -> None:
    """Clear active custom map pin and return diagnostics to canonical 18-station mode."""
    st.session_state.pop("custom_point", None)
    st.session_state.pop("city_candidates", None)
    st.session_state["diag_target_mode"] = "18 Synoptic Stations"


# Regional Risk Map
st.subheader("Regional Bust Risk Map")
st.markdown(
    f"""
    <div class="map-legend">
        <span class="legend-item"><span class="legend-dot" style="background:{PALETTE['critical_red']};"></span> High-Risk Bust (&ge;{alert_threshold}%)</span>
        <span class="legend-item"><span class="legend-dot" style="background:{PALETTE['amber']};"></span> Elevated Watch (35&ndash;{alert_threshold}%)</span>
        <span class="legend-item"><span class="legend-dot" style="background:{PALETTE['sage_green']};"></span> Reliable (&lt;35%)</span>
        <span class="legend-item"><span class="legend-dot" style="background:{PALETTE['espresso']}; border:2px solid {PALETTE['terracotta']};"></span> Custom Map Pin (Click anywhere on map)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

active_custom_pin = st.session_state.get("custom_point")
india_map = folium.Map(
    location=[22.2, 80.5],
    zoom_start=4,
    tiles="OpenStreetMap",
    control_scale=True,
)
if active_custom_pin:
    cp_lat = float(active_custom_pin["lat"])
    cp_lon = float(active_custom_pin["lon"])
    india_map.fit_bounds([
        [min(8.0, cp_lat - 1.5), min(68.5, cp_lon - 1.5)],
        [max(35.2, cp_lat + 1.5), max(95.5, cp_lon + 1.5)],
    ])
else:
    india_map.fit_bounds([[8.0, 68.5], [35.2, 95.5]])

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
    elif risk >= 35.0:
        marker_color = PALETTE["amber"]
        status_text = "Elevated Watch"
        radius = 8
    else:
        marker_color = PALETTE["sage_green"]
        status_text = "Reliable"
        radius = 7

    popup_html = f"""
    <div style="font-family: -apple-system, sans-serif; font-size: 12.5px; width: 235px; background: {PALETTE['cream']}; color: {PALETTE['espresso']}; padding: 8px 10px; border-radius: 6px; border: 1px solid {PALETTE['border']};">
        <div style="font-weight: 700; font-size: 13.5px; color: {PALETTE['espresso']}; margin-bottom: 4px;">{row['station_name']}</div>
        <div style="margin-bottom: 6px;">
            <span style="display:inline-block; background:{marker_color}; color:#FFF9F0; font-size:10px; font-weight:600; padding:2px 7px; border-radius:3px;">{status_text}</span>
            <strong style="float:right; color:{marker_color}; font-size:14px;">{risk:.1f}%</strong>
        </div>
        <div style="border-top: 1px solid {PALETTE['border']}; padding-top: 6px; line-height: 1.55;">
            <b>Forecast Temp:</b> {row['predicted_temp']} °C<br>
            <b>Forecast Rain:</b> {row['predicted_rainfall']} mm<br>
            <b>24h Pressure Drop:</b> {row['pressure_drop']} hPa<br>
            <b>0-6km Wind Shear:</b> {row['wind_shear']} m/s<br>
            <b>CAPE / RH:</b> {float(row.get('cape', 0.0)):.0f} J/kg · {float(row.get('relative_humidity', 65.0)):.0f}%<br>
            <b>ECMWF-GFS Δ:</b> {float(row.get('model_temp_spread', 0.0)):.1f}°C · {float(row.get('model_rain_spread', 0.0)):.1f}mm
        </div>
    </div>
    """

    if risk >= alert_threshold:
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=17,
            color=PALETTE["critical_red"],
            fill=True,
            fill_color=PALETTE["critical_red"],
            fill_opacity=0.18,
            weight=1,
            interactive=False,
        ).add_to(india_map)

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

if active_custom_pin:
    cp_lat = float(active_custom_pin["lat"])
    cp_lon = float(active_custom_pin["lon"])
    folium.CircleMarker(
        location=[cp_lat, cp_lon],
        radius=16,
        color=PALETTE["espresso"],
        fill=True,
        fill_color=PALETTE["terracotta"],
        fill_opacity=0.22,
        weight=2,
        interactive=False,
    ).add_to(india_map)
    crosshair_html = (
        f'<div style="width:18px;height:18px;border-radius:50%;background:{PALETTE["espresso"]};'
        f'border:3px solid {PALETTE["amber"]};box-shadow:0 2px 6px rgba(42,26,14,0.45);"></div>'
    )
    folium.Marker(
        location=[cp_lat, cp_lon],
        icon=folium.DivIcon(icon_size=(18, 18), icon_anchor=(9, 9), html=crosshair_html),
        tooltip=f"Custom Pin: {active_custom_pin['label']} ({cp_lat:.2f}°N, {cp_lon:.2f}°E)",
        popup=f"<b>Custom Inspection Point</b><br>{active_custom_pin['label']}<br>{cp_lat:.4f}°N, {cp_lon:.4f}°E",
    ).add_to(india_map)

# Capture only explicit map clicks (pan/zoom do not trigger reruns)
map_state = st_folium(
    india_map,
    use_container_width=True,
    height=430,
    returned_objects=["last_clicked"],
    key="india_risk_map",
)

clicked_coords = map_state.get("last_clicked") if isinstance(map_state, dict) else None
if clicked_coords and clicked_coords.get("lat") is not None and clicked_coords.get("lng") is not None:
    click_sig = (round(float(clicked_coords["lat"]), 4), round(float(clicked_coords["lng"]), 4))
    if st.session_state.get("_last_map_click_sig") != click_sig:
        st.session_state["_last_map_click_sig"] = click_sig
        st.session_state.pop("city_candidates", None)
        _set_custom_point(
            lat=click_sig[0],
            lon=click_sig[1],
            label=f"Map Pin ({click_sig[0]:.2f}°N, {click_sig[1]:.2f}°E)",
            switch_to_custom=True,
        )
        st.rerun()

if st.session_state.get("custom_point"):
    cp_bar = st.session_state["custom_point"]
    pin_c1, pin_c2 = st.columns([4.2, 1])
    with pin_c1:
        st.info(
            f"Active Custom Point: **{cp_bar['label']}** ({cp_bar['lat']:.4f}°N, {cp_bar['lon']:.4f}°E) — "
            "scored live in **Diagnostics & Horizon -> Custom Map Pin / City** below."
        )
    with pin_c2:
        st.button("Clear Map Pin", width="stretch", on_click=_clear_custom_pin, key="clear_pin_banner")
else:
    st.caption(
        "Tip: Click anywhere on the map above to run live NWP bust + SHAP diagnostics for any town, highway, or farm — "
        "or switch to **Custom Map Pin / City** inside Diagnostics & Horizon."
    )


# Operational Diagnostics Tabs
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
tab_diag, tab_whatif, tab_scorecard, tab_bulletin = st.tabs([
    "Diagnostics & Horizon",
    "What-If Simulator",
    "Verification Scorecard",
    "Bulletin & Model",
])

with tab_diag:
    st.session_state.setdefault("diag_target_mode", "18 Synoptic Stations")
    diag_target_mode = st.radio(
        "Inspection Scope",
        ["18 Synoptic Stations", "Custom Map Pin / City"],
        horizontal=True,
        key="diag_target_mode",
    )

    if diag_target_mode == "18 Synoptic Stations":
        st.caption("Inspect SHAP drivers for any synoptic station alongside its full Day +1 to Day +10 predictability curve.")
        station_options = df_stations["station_name"].tolist()
        selected_name = st.selectbox(
            "Inspect Station",
            options=station_options,
            index=top_pos,
        )
        sel = df_stations[df_stations["station_name"] == selected_name].iloc[0]
        target_horizon_df = horizon_df[horizon_df["station_name"] == selected_name]
        panel_title = str(sel["station_name"])
        sel_short = panel_title.split(" (")[0]
        meta_line = f"{sel['state']} &middot; {sel['zone']} &middot; {float(sel['lat']):.2f}°N, {float(sel['lon']):.2f}°E"
        footer_source_override: Optional[str] = None
    else:
        st.caption("Click any coordinate on the map above, search a city/town, or enter custom latitude/longitude for live multi-model NWP scoring.")
        cp_active = st.session_state.get("custom_point") or {
            "lat": 26.8393,
            "lon": 80.9231,
            "label": "Lucknow, Uttar Pradesh, India",
            "state": "Uttar Pradesh",
        }
        st.session_state.setdefault("custom_point", cp_active)
        st.session_state.setdefault("custom_lat_box", float(cp_active["lat"]))
        st.session_state.setdefault("custom_lon_box", float(cp_active["lon"]))

        cs_c1, cs_c2 = st.columns([3.2, 1])
        with cs_c1:
            city_query = st.text_input(
                "Search City or Town",
                placeholder="e.g., Lucknow, Varanasi, Darjeeling, Surat, Shillong",
                key="city_search_input",
            )
        with cs_c2:
            st.markdown("<div class='btn-spacer'></div>", unsafe_allow_html=True)
            if st.button("Find & Score City", width="stretch", key="find_city_btn"):
                matches = geocode_city_candidates(city_query)
                if matches:
                    st.session_state["city_candidates"] = matches
                    st.session_state.pop("city_candidate_picker", None)
                    top_match = matches[0]
                    _set_custom_point(
                        lat=top_match["lat"],
                        lon=top_match["lon"],
                        label=top_match["label"],
                        state=top_match["admin1"] or top_match["country"] or None,
                        switch_to_custom=True,
                    )
                    st.rerun()
                else:
                    st.warning("No matching city found in Open-Meteo Geocoder. Try another spelling or click the map.")

        candidates = st.session_state.get("city_candidates") or []
        if len(candidates) > 1:
            def _on_pick_city_candidate() -> None:
                picked = st.session_state.get("city_candidate_picker")
                if isinstance(picked, dict):
                    _set_custom_point(
                        lat=picked["lat"],
                        lon=picked["lon"],
                        label=picked["label"],
                        state=picked.get("admin1") or picked.get("country") or None,
                        switch_to_custom=True,
                    )

            st.selectbox(
                "Matching Locations",
                options=candidates,
                format_func=lambda item: f"{item['label']} ({item['lat']:.2f}°N, {item['lon']:.2f}°E)",
                key="city_candidate_picker",
                on_change=_on_pick_city_candidate,
            )

        cp_active = st.session_state["custom_point"]
        ll_c1, ll_c2, ll_c3, ll_c4 = st.columns([1, 1, 1, 1])
        with ll_c1:
            lat_in = st.number_input("Latitude (°N)", min_value=-60.0, max_value=60.0, step=0.05, format="%.4f", key="custom_lat_box")
        with ll_c2:
            lon_in = st.number_input("Longitude (°E)", min_value=-180.0, max_value=180.0, step=0.05, format="%.4f", key="custom_lon_box")
        with ll_c3:
            st.markdown("<div class='btn-spacer'></div>", unsafe_allow_html=True)
            if st.button("Score Coords", width="stretch", key="score_coords_btn"):
                st.session_state["custom_point"] = {
                    "lat": round(float(lat_in), 4),
                    "lon": round(float(lon_in), 4),
                    "label": f"Custom Coords ({float(lat_in):.2f}°N, {float(lon_in):.2f}°E)",
                    "state": None,
                }
                st.session_state.pop("city_candidates", None)
                st.session_state.pop("city_candidate_picker", None)
                st.rerun()
        with ll_c4:
            st.markdown("<div class='btn-spacer'></div>", unsafe_allow_html=True)
            st.button("Reset to Grid", width="stretch", on_click=_clear_custom_pin, key="reset_pin_btn")

        target_horizon_df, cp_source, cp_elev, nearest_info = get_custom_point_horizon(
            lat=float(cp_active["lat"]),
            lon=float(cp_active["lon"]),
            label=str(cp_active["label"]),
            data_mode=data_mode,
            nwp_model=nwp_model,
            grid_df=horizon_df,
            state_override=cp_active.get("state"),
        )
        sel = target_horizon_df[target_horizon_df["lead_day"] == lead_day].iloc[0]
        panel_title = str(cp_active["label"])
        sel_short = panel_title.split(",")[0].split(" (")[0]
        elev_str = f" &middot; Elev {cp_elev:.0f}m" if cp_elev is not None else ""
        meta_line = (
            f"{sel['state']} &middot; {nearest_info['dist_km']:.0f} km from {nearest_info['name'].split(' (')[0]}"
            f"{elev_str} &middot; {float(sel['lat']):.2f}°N, {float(sel['lon']):.2f}°E"
        )
        footer_source_override = cp_source

    sel_risk = float(sel["risk_percentage"])
    sel_rh = float(sel.get("relative_humidity", 65.0))
    sel_cape = float(sel.get("cape", 250.0))
    sel_tspread = float(sel.get("model_temp_spread", 1.0))
    sel_rspread = float(sel.get("model_rain_spread", 2.0))
    sel_tbust = float(sel["temp_bust_prob"]) if pd.notna(sel.get("temp_bust_prob")) else None
    sel_rbust = float(sel["rain_bust_prob"]) if pd.notna(sel.get("rain_bust_prob")) else None

    shap_result = get_station_shap(
        lat=float(sel["lat"]),
        lon=float(sel["lon"]),
        predicted_temp=float(sel["predicted_temp"]),
        predicted_rainfall=float(sel["predicted_rainfall"]),
        wind_shear=float(sel["wind_shear"]),
        pressure_drop=float(sel["pressure_drop"]),
        lead_day=lead_day,
        relative_humidity=sel_rh,
        cape=sel_cape,
        model_temp_spread=sel_tspread,
        model_rain_spread=sel_rspread,
        use_api=api_live,
        top_k=3,
    )
    factors = shap_result["factors"]
    effective_source = footer_source_override or shap_result["source"]

    col_info, col_chart = st.columns([1, 1.35])
    with col_info:
        render_station_panel(
            title=panel_title,
            meta_line=meta_line,
            risk_pct=sel_risk,
            alert_threshold=alert_threshold,
            temp=sel["predicted_temp"],
            rainfall=sel["predicted_rainfall"],
            shear=sel["wind_shear"],
            pdrop=sel["pressure_drop"],
            footer_meta=f"Day +{lead_day} ({sel.get('forecast_date', active_date)}, {lead_day * 24}h) &middot; {effective_source}",
            temp_bust_pct=sel_tbust,
            rain_bust_pct=sel_rbust,
            spread_t=sel_tspread,
            spread_r=sel_rspread,
            cape=sel_cape,
            rh=sel_rh,
        )

    with col_chart:
        shap_fig = build_shap_figure(factors, f"SHAP Drivers — {sel_short}")
        st.plotly_chart(shap_fig, width="stretch", config={"displayModeBar": False})

    render_driver_cards(factors)

    if diag_target_mode != "18 Synoptic Stations":
        if st.button("Log Custom Point Inspection to Audit DB", key="log_custom_point_db"):
            try:
                from db import log_prediction

                bundle_meta = load_model_bundle().get("metadata", {})
                model_ver = str(bundle_meta.get("pipeline_version", "v3-multihorizon-spread-10728"))
                log_prediction(
                    input_features={
                        "predicted_temp": float(sel["predicted_temp"]),
                        "predicted_rainfall": float(sel["predicted_rainfall"]),
                        "wind_shear": float(sel["wind_shear"]),
                        "pressure_drop": float(sel["pressure_drop"]),
                        "lead_day": int(lead_day),
                        "relative_humidity": sel_rh,
                        "cape": sel_cape,
                        "model_temp_spread": sel_tspread,
                        "model_rain_spread": sel_rspread,
                    },
                    risk_prob_pct=sel_risk,
                    risk_level=str(sel.get("risk_level", "Elevated")),
                    shap_factors=factors,
                    model_version=model_ver,
                    latitude=float(sel["lat"]),
                    longitude=float(sel["lon"]),
                    station_name=panel_title,
                    lead_time_hours=int(lead_day) * 24,
                    source="dashboard_custom_pin",
                )
                st.success(f"Saved inspection for **{panel_title}** (`{sel_risk:.1f}%`) to SQL audit log (`prediction_logs`).")
            except Exception as exc:
                st.warning(f"Could not write audit log entry: {exc}")

    horizon_fig = build_horizon_figure(target_horizon_df, sel_short, alert_threshold, lead_day)
    st.plotly_chart(horizon_fig, width="stretch", config={"displayModeBar": False})

with tab_whatif:
    st.caption("Stress-test extreme synoptic dynamics, CAPE, and ECMWF–GFS model disagreement live.")

    st.session_state.setdefault("whatif_temp", 31.0)
    st.session_state.setdefault("whatif_rain", 12.0)
    st.session_state.setdefault("whatif_shear", 18.0)
    st.session_state.setdefault("whatif_pdrop", 6.0)
    st.session_state.setdefault("whatif_lead", 3)
    st.session_state.setdefault("whatif_rh", 68.0)
    st.session_state.setdefault("whatif_cape", 350.0)
    st.session_state.setdefault("whatif_tspread", 1.2)
    st.session_state.setdefault("whatif_rspread", 4.0)

    def _set_whatif_preset(
        temp_val: float,
        rain_val: float,
        shear_val: float,
        pdrop_val: float,
        lead_val: int,
        rh_val: float,
        cape_val: float,
        tspread_val: float,
        rspread_val: float,
    ) -> None:
        st.session_state["whatif_temp"] = temp_val
        st.session_state["whatif_rain"] = rain_val
        st.session_state["whatif_shear"] = shear_val
        st.session_state["whatif_pdrop"] = pdrop_val
        st.session_state["whatif_lead"] = lead_val
        st.session_state["whatif_rh"] = rh_val
        st.session_state["whatif_cape"] = cape_val
        st.session_state["whatif_tspread"] = tspread_val
        st.session_state["whatif_rspread"] = rspread_val

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.button(
            "Preset: Calm Clear Day",
            width="stretch",
            on_click=_set_whatif_preset,
            args=(28.5, 0.0, 9.0, 2.0, 2, 48.0, 120.0, 0.5, 0.5),
        )
    with p_col2:
        st.button(
            "Preset: Western Disturbance",
            width="stretch",
            on_click=_set_whatif_preset,
            args=(19.0, 26.0, 36.0, 14.5, 5, 82.0, 750.0, 2.8, 16.0),
        )
    with p_col3:
        st.button(
            "Preset: Coastal Cyclone",
            width="stretch",
            on_click=_set_whatif_preset,
            args=(33.5, 68.0, 44.0, 23.0, 5, 92.0, 2200.0, 3.6, 28.0),
        )

    sim_lead = st.slider("Forecast Lead Horizon (Day)", min_value=1, max_value=10, step=1, key="whatif_lead")
    in_col1, in_col2 = st.columns(2)
    with in_col1:
        sim_temp = st.slider("Predicted Temperature (°C)", min_value=2.0, max_value=45.0, step=0.5, key="whatif_temp")
        sim_rain = st.slider("24h Forecast Rainfall (mm)", min_value=0.0, max_value=140.0, step=1.0, key="whatif_rain")
        sim_rh = st.slider("Relative Humidity (%)", min_value=10.0, max_value=100.0, step=1.0, key="whatif_rh")
        sim_tspread = st.slider("ECMWF–GFS Temp Spread (°C)", min_value=0.0, max_value=8.0, step=0.1, key="whatif_tspread")
    with in_col2:
        sim_shear = st.slider("0–6km Wind Shear (m/s)", min_value=2.0, max_value=50.0, step=0.5, key="whatif_shear")
        sim_pdrop = st.slider("24h Pressure Drop (hPa)", min_value=0.0, max_value=32.0, step=0.5, key="whatif_pdrop")
        sim_cape = st.slider("CAPE Instability (J/kg)", min_value=0.0, max_value=4000.0, step=50.0, key="whatif_cape")
        sim_rspread = st.slider("ECMWF–GFS Rain Spread (mm)", min_value=0.0, max_value=60.0, step=0.5, key="whatif_rspread")

    custom_res = evaluate_custom_scenario(
        predicted_temp=float(sim_temp),
        predicted_rainfall=float(sim_rain),
        wind_shear=float(sim_shear),
        pressure_drop=float(sim_pdrop),
        lead_day=int(sim_lead),
        relative_humidity=float(sim_rh),
        cape=float(sim_cape),
        model_temp_spread=float(sim_tspread),
        model_rain_spread=float(sim_rspread),
    )

    sim_col_info, sim_col_chart = st.columns([1, 1.35])
    with sim_col_info:
        render_station_panel(
            title="Custom Stress Scenario",
            meta_line=f"Day +{int(sim_lead)} Synthetic Profile &middot; Threshold {alert_threshold}%",
            risk_pct=custom_res["risk_percentage"],
            alert_threshold=alert_threshold,
            temp=round(float(sim_temp), 1),
            rainfall=round(float(sim_rain), 1),
            shear=round(float(sim_shear), 2),
            pdrop=round(float(sim_pdrop), 2),
            footer_meta="Evaluated live via calibrated multi-horizon ML + SHAP TreeExplainer",
            temp_bust_pct=custom_res.get("temp_bust_prob"),
            rain_bust_pct=custom_res.get("rain_bust_prob"),
            spread_t=float(sim_tspread),
            spread_r=float(sim_rspread),
            cape=float(sim_cape),
            rh=float(sim_rh),
        )
    with sim_col_chart:
        sim_fig = build_shap_figure(custom_res["factors"], "SHAP Drivers — Custom Scenario")
        st.plotly_chart(sim_fig, width="stretch", config={"displayModeBar": False})

    render_driver_cards(custom_res["factors"])

with tab_scorecard:
    render_scorecard_tab(alert_threshold=alert_threshold)

with tab_bulletin:
    sorted_df = df_stations.sort_values(by="risk_percentage", ascending=False).reset_index(drop=True)
    export_cols = [
        c for c in [
            "forecast_date",
            "lead_day",
            "station_name",
            "state",
            "zone",
            "risk_percentage",
            "temp_bust_prob",
            "rain_bust_prob",
            "risk_level",
            "predicted_temp",
            "predicted_rainfall",
            "pressure_drop",
            "wind_shear",
            "cape",
            "relative_humidity",
            "model_temp_spread",
            "model_rain_spread",
        ]
        if c in sorted_df.columns
    ]
    export_df = sorted_df[export_cols]
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        st.caption(
            f"Sorted highest-to-lowest bust risk for **Day +{lead_day}** ({active_date}) via **{feed_label}**. "
            "Swipe horizontally on compact screens to view ensemble spread & instability metrics."
        )
    with b_col2:
        st.download_button(
            label="Download Bust Bulletin (CSV)",
            data=csv_bytes,
            file_name=f"forecast_bust_bulletin_day{lead_day}_{active_date}.csv",
            mime="text/csv",
            width="stretch",
        )

    table_cols = [
        c for c in [
            "station_name",
            "zone",
            "risk_percentage",
            "temp_bust_prob",
            "rain_bust_prob",
            "risk_level",
            "predicted_temp",
            "predicted_rainfall",
            "pressure_drop",
            "wind_shear",
            "cape",
            "model_temp_spread",
            "model_rain_spread",
        ]
        if c in sorted_df.columns
    ]
    table_df = sorted_df[table_cols]
    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        column_config={
            "station_name": st.column_config.TextColumn("Station", width="medium"),
            "zone": st.column_config.TextColumn("Zone", width="small"),
            "risk_percentage": st.column_config.NumberColumn("Risk", format="%.1f%%", width="small"),
            "temp_bust_prob": st.column_config.NumberColumn("Temp Bust", format="%.1f%%", width="small"),
            "rain_bust_prob": st.column_config.NumberColumn("Rain Bust", format="%.1f%%", width="small"),
            "risk_level": st.column_config.TextColumn("Tier", width="small"),
            "predicted_temp": st.column_config.NumberColumn("Temp", format="%.1f °C", width="small"),
            "predicted_rainfall": st.column_config.NumberColumn("Rain", format="%.1f mm", width="small"),
            "pressure_drop": st.column_config.NumberColumn("ΔPress", format="%.1f hPa", width="small"),
            "wind_shear": st.column_config.NumberColumn("Shear", format="%.1f m/s", width="small"),
            "cape": st.column_config.NumberColumn("CAPE", format="%.0f J/kg", width="small"),
            "model_temp_spread": st.column_config.NumberColumn("ECMWF-GFS ΔT", format="%.1f °C", width="small"),
            "model_rain_spread": st.column_config.NumberColumn("ECMWF-GFS ΔR", format="%.1f mm", width="small"),
        },
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("**Model Validation & Global Feature Importance**")
    bundle_meta = load_model_bundle().get("metadata", {})
    champion_name = str(bundle_meta.get("champion_name", "RandomForest"))
    metrics = bundle_meta.get("metrics", {}) if isinstance(bundle_meta.get("metrics"), dict) else {}
    roc_auc = metrics.get("roc_auc")
    brier_val = metrics.get("brier_score")
    f1_val = metrics.get("f1")
    recall_val = metrics.get("recall")
    cal_method = str(bundle_meta.get("calibration") or metrics.get("calibration") or "none")
    val_scheme = str(bundle_meta.get("validation_scheme") or metrics.get("validation_scheme") or "holdout")
    test_range = bundle_meta.get("test_date_range")

    training_source = str(bundle_meta.get("training_source", "Historical NWP Benchmark Dataset"))
    sample_count = bundle_meta.get("sample_count")
    bust_rate = bundle_meta.get("bust_rate")
    pipeline_ver = bundle_meta.get("pipeline_version")

    dataset_badge_parts = [f"Source: **{training_source}**"]
    if sample_count is not None:
        dataset_badge_parts.append(f"Records: **{int(sample_count):,}**")
    if bust_rate is not None:
        dataset_badge_parts.append(f"Observed Bust Rate: **{float(bust_rate) * 100:.1f}%**")
    if test_range and test_range != "n/a":
        dataset_badge_parts.append(f"Hold-Out Window: **{test_range}**")
    st.caption(" · ".join(dataset_badge_parts))

    roc_txt = f"{float(roc_auc):.3f}" if roc_auc is not None else "N/A"
    brier_txt = f"{float(brier_val):.4f}" if brier_val is not None else "N/A"
    f1_txt = f"{float(f1_val):.3f}" if f1_val is not None else "N/A"
    recall_txt = f"{float(recall_val):.3f}" if recall_val is not None else "N/A"
    cal_tag = f"Calibrated ({cal_method})" if cal_method != "none" else "Uncalibrated"
    classifier_sub = (
        f"{int(sample_count):,} records · {pipeline_ver}"
        if sample_count is not None and pipeline_ver
        else "Serialized in bust_model.pkl"
    )
    auc_sub = (
        f"{val_scheme.replace('_', ' ')} ({float(bust_rate) * 100:.1f}% bust)"
        if bust_rate is not None
        else val_scheme.replace("_", " ")
    )
    t_head_auc = metrics.get("temp_head_auc")
    r_head_auc = metrics.get("rain_head_auc")
    brier_sub = (
        f"Temp AUC {float(t_head_auc):.2f} · Rain AUC {float(r_head_auc):.2f}"
        if t_head_auc is not None and r_head_auc is not None
        else cal_tag
    )

    st.markdown(
        f"""
<div class="kpi-grid" style="margin-top:8px;">
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['espresso']};">
    <div class="kpi-label">Active Classifier</div>
    <div class="kpi-value" style="color:{PALETTE['espresso']}; font-size:1.08rem;">{champion_name}</div>
    <div class="kpi-sub">{classifier_sub}</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['terracotta']};">
    <div class="kpi-label">Chronological ROC-AUC</div>
    <div class="kpi-value" style="color:{PALETTE['terracotta']};">{roc_txt}</div>
    <div class="kpi-sub">{auc_sub}</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['amber']};">
    <div class="kpi-label">Brier Score ({cal_method})</div>
    <div class="kpi-value" style="color:{PALETTE['espresso']};">{brier_txt}</div>
    <div class="kpi-sub">{brier_sub}</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['sage_green']};">
    <div class="kpi-label">Validation F1 &middot; Recall</div>
    <div class="kpi-value" style="color:{PALETTE['sage_green']};">{f1_txt} / {recall_txt}</div>
    <div class="kpi-sub">Chronological hold-out threshold 0.50</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    global_drivers = bundle_meta.get("global_drivers", [])
    if isinstance(global_drivers, list) and global_drivers:
        g_labels = [
            CHART_LABELS.get(str(d.get("Feature", "")), str(d.get("Feature", "")).replace("_", " ").title())
            for d in global_drivers
        ]
        g_vals = [float(d.get("Mean_Abs_SHAP", 0.0)) for d in global_drivers]
        g_max = max(g_vals + [0.02]) * 1.32
        g_fig = go.Figure(
            go.Bar(
                x=g_vals,
                y=g_labels,
                orientation="h",
                marker=dict(color=PALETTE["terracotta"], line=dict(color=PALETTE["espresso"], width=0.8)),
                text=[f"{v:.4f}" for v in g_vals],
                textposition="auto",
                insidetextanchor="middle",
                insidetextfont=dict(color="#FFF9F0", family="IBM Plex Mono, monospace", size=11),
                outsidetextfont=dict(color=PALETTE["espresso"], family="IBM Plex Mono, monospace", size=11),
            )
        )
        g_fig.update_layout(
            title=dict(
                text="Global Mean |SHAP| Driver Hierarchy",
                x=0.01,
                font=dict(size=14, color=PALETTE["espresso"], family="DM Sans, sans-serif"),
            ),
            xaxis=dict(
                title=dict(text="Mean |SHAP| Magnitude", font=dict(size=11.5, color=PALETTE["muted_brown"])),
                range=[0, g_max],
                gridcolor="#EDE0C9",
                tickfont=dict(color=PALETTE["muted_brown"], family="IBM Plex Mono, monospace", size=10.5),
            ),
            yaxis=dict(
                autorange="reversed",
                automargin=True,
                tickfont=dict(color=PALETTE["espresso"], size=11.5),
            ),
            paper_bgcolor=PALETTE["surface"],
            plot_bgcolor=PALETTE["cream"],
            margin=dict(l=10, r=20, t=44, b=34),
            height=max(260, 44 + 28 * len(g_labels)),
        )
        st.plotly_chart(g_fig, width="stretch", config={"displayModeBar": False})

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("**Database Persistence & Sync Status**")
    live_db = read_db_summary_safe() or ensure_db_initialized()
    if live_db:
        st.markdown(
            f"""
<div class="kpi-grid" style="margin-top:8px;">
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['espresso']};">
    <div class="kpi-label">Active Database</div>
    <div class="kpi-value" style="color:{PALETTE['espresso']}; font-size:1.1rem;">{live_db['backend']}</div>
    <div class="kpi-sub">{live_db['location']}</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['terracotta']};">
    <div class="kpi-label">Forecast Snapshots</div>
    <div class="kpi-value" style="color:{PALETTE['terracotta']};">{live_db['forecast_runs']:,}</div>
    <div class="kpi-sub">{live_db['stations']} synoptic stations seeded</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['amber']};">
    <div class="kpi-label">Verified Archive Pairs</div>
    <div class="kpi-value" style="color:{PALETTE['espresso']};">{live_db['verified_pairs']:,}</div>
    <div class="kpi-sub">{live_db['verification_outcomes']:,} Day-0 actual rows</div>
  </div>
  <div class="kpi-card" style="border-top: 3px solid {PALETTE['sage_green']};">
    <div class="kpi-label">Registry & Audit Logs</div>
    <div class="kpi-value" style="color:{PALETTE['sage_green']};">{live_db['model_runs']} / {live_db['prediction_logs']}</div>
    <div class="kpi-sub">Models tracked / API predictions</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        sync_col1, sync_col2 = st.columns([2, 1])
        with sync_col1:
            st.caption(
                "Connect Supabase or Neon PostgreSQL anytime by setting `DATABASE_URL` in environment or Streamlit Secrets; "
                "until then, records persist in local SQLite (`forecast_bust.db`)."
            )
        with sync_col2:
            if st.button("Sync 150-Day Paired Archive to DB", width="stretch"):
                try:
                    with st.spinner("Syncing 150-day Open-Meteo multi-horizon archive into SQL database..."):
                        import importlib
                        import data_pipeline

                        importlib.reload(data_pipeline)
                        synced = data_pipeline.sync_archive_to_database(past_days=150)
                    load_verified_scorecard_df.clear()
                    st.success(
                        f"Synced {synced['forecast_runs']:,} forecast runs and {synced['verification_outcomes']:,} verification rows."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Archive sync failed: {exc}")
        if isinstance(db_summary, dict) and db_summary.get("hydration_error"):
            st.warning(f"Startup archive hydration note: {db_summary['hydration_error']}")
    else:
        st.info("Database engine initializing or unavailable in this environment.")
