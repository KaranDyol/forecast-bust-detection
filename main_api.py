"""
main_api.py
===========
Phase 3: High-Performance FastAPI Microservice for AI Forecast Bust Detection.

Features:
- Microservice architecture with FastAPI and Pydantic schema validation.
- In-memory caching of the trained champion ML model and SHAP TreeExplainer.
- Real-time POST /predict_bust_risk endpoint returning:
  * Geographical coordinates
  * Risk Probability Percentage (0% to 100%)
  * High-risk warning threshold alert
  * Detailed Explainable AI (SHAP) root-cause factors
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Resolve model path relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bust_model.pkl"

# Global model state container
model_bundle: Dict[str, Any] = {}


def load_model_bundle() -> None:
    """Load model artifact into global memory if not already loaded."""
    if "model" in model_bundle:
        return
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH.resolve()}. "
            "Please run train_model.py to generate bust_model.pkl first."
        )
    print(f"[*] Loading model artifact bundle from: {MODEL_PATH.resolve()}...")
    bundle = joblib.load(MODEL_PATH)
    model_bundle["model"] = bundle["model"]
    model_bundle["explainer"] = bundle["explainer"]
    model_bundle["feature_names"] = bundle["feature_names"]
    model_bundle["metadata"] = bundle.get("metadata", {})
    champion = model_bundle["metadata"].get("champion_name", "RandomForest")
    print(f"[+] Model bundle loaded successfully. Champion: {champion}")


# Pre-load model at import time so endpoints & testclients are immediately ready
try:
    load_model_bundle()
except Exception as e:
    print(f"[!] Warning during initial model load: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for ASGI lifecycle management."""
    load_model_bundle()
    yield



# Initialize FastAPI Microservice
app = FastAPI(
    title="AI Forecast Bust Detection Microservice",
    description=(
        "Decision support layer over Numerical Weather Prediction (NWP) systems. "
        "Predicts the probability of medium-range forecast failure and diagnoses "
        "meteorological root causes via SHAP (Explainable AI)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients (e.g., Streamlit, React dashboards)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# PYDANTIC DATA SCHEMAS
# ==========================================

class CoordinateModel(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees", example=-33.8688)
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees", example=151.2093)


class NWPForecastInput(BaseModel):
    """
    Input schema capturing NWP model outputs and atmospheric synoptic pre-conditions.
    """
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Location latitude", example=-33.8688)
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Location longitude", example=151.2093)
    predicted_temp: float = Field(..., description="NWP forecasted surface temperature (°C)", example=32.5)
    predicted_rainfall: float = Field(..., ge=0.0, description="NWP forecasted 24h accumulated rainfall (mm)", example=24.0)
    wind_shear: float = Field(..., ge=0.0, description="0-6 km bulk vertical wind shear (m/s)", example=38.5)
    pressure_drop: float = Field(..., ge=0.0, description="24-hour barometric pressure drop (hPa)", example=18.2)
    relative_humidity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Optional surface relative humidity (%)", example=82.0)
    lead_time_hours: Optional[int] = Field(72, description="Forecast lead time in hours (medium-range)", example=72)


class ExplainabilityFactor(BaseModel):
    feature: str = Field(..., description="Atmospheric or NWP driver feature name")
    feature_value: float = Field(..., description="Observed input value")
    shap_value: float = Field(..., description="SHAP attribution value (log-odds impact)")
    impact: str = Field(..., description="'increases_bust_risk' or 'decreases_bust_risk'")
    description: str = Field(..., description="Meteorological interpretation of the driver's role")


class BustRiskResponse(BaseModel):
    coordinates: CoordinateModel
    risk_probability_percentage: float = Field(..., description="Calibrated forecast bust probability (0.0% to 100.0%)")
    is_high_risk_warning: bool = Field(..., description="True if risk exceeds alert threshold (50.0%)")
    risk_level: str = Field(..., description="Categorical risk tier: LOW, ELEVATED, HIGH, or CRITICAL")
    explainability_factors: List[ExplainabilityFactor] = Field(..., description="Top contributing root cause drivers from SHAP")
    evaluated_at: str = Field(..., description="ISO 8601 UTC evaluation timestamp")


# ==========================================
# METEOROLOGICAL EXPLANATION UTILITIES
# ==========================================

FEATURE_NARRATIVES = {
    "pressure_drop": {
        "positive": "Rapid barometric pressure fall signals strong frontal cyclogenesis or deep convective boundary, creating severe forecast timing/intensity uncertainty.",
        "negative": "Stable or rising barometric pressure provides strong synoptic stabilization, suppressing bust risk."
    },
    "wind_shear": {
        "positive": "Strong vertical wind shear organizes storm updrafts and baroclinic tilting, which NWP models frequently displace spatially.",
        "negative": "Weak vertical wind shear limits convective organization, keeping NWP dynamics within predictable laminar regimes."
    },
    "predicted_rainfall": {
        "positive": "High convective precipitation forecasts encounter non-linear microphysical uncertainties, elevating failure risk.",
        "negative": "Low to zero precipitation forecasts fall within well-constrained dry synoptic regimes."
    },
    "predicted_temp": {
        "positive": "Extreme forecasted temperature pushes boundaries toward climatological extremes where model parameterizations degrade.",
        "negative": "Moderate forecasted temperature aligns with typical seasonal climatology, maintaining forecast reliability."
    }
}


def explain_single_prediction(
    input_df: pd.DataFrame,
    top_k: int = 3
) -> List[ExplainabilityFactor]:
    """
    Computes local SHAP attribution values for a single forecast sample
    and enriches them with meteorological diagnostic explanations.
    """
    explainer = model_bundle["explainer"]
    feature_names = model_bundle["feature_names"]

    # Calculate SHAP values
    raw_shap = explainer.shap_values(input_df[feature_names])
    if isinstance(raw_shap, list):
        shap_vec = raw_shap[1][0]
    elif len(raw_shap.shape) == 3:
        shap_vec = raw_shap[0, :, 1]
    else:
        shap_vec = raw_shap[0]

    factors = []
    for feat_name, s_val in zip(feature_names, shap_vec):
        val = float(input_df[feat_name].iloc[0])
        s_val_float = float(s_val)
        is_pos = s_val_float > 0
        impact_label = "increases_bust_risk" if is_pos else "decreases_bust_risk"

        narrative_dict = FEATURE_NARRATIVES.get(feat_name, {})
        narrative = narrative_dict.get("positive" if is_pos else "negative", "Atmospheric factor modulating forecast reliability.")

        factors.append({
            "feature": feat_name,
            "feature_value": round(val, 2),
            "shap_value": round(s_val_float, 4),
            "impact": impact_label,
            "description": narrative,
            "abs_shap": abs(s_val_float)
        })

    # Sort descending by absolute SHAP impact and take top_k
    factors.sort(key=lambda x: x["abs_shap"], reverse=True)
    return [
        ExplainabilityFactor(
            feature=f["feature"],
            feature_value=f["feature_value"],
            shap_value=f["shap_value"],
            impact=f["impact"],
            description=f["description"]
        )
        for f in factors[:top_k]
    ]


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/", tags=["System"])
def root():
    """Service health check and metadata."""
    return {
        "service": "AI-Based Forecast Bust Detection Microservice",
        "status": "healthy",
        "version": "1.0.0",
        "champion_model": model_bundle.get("metadata", {}).get("champion_name", "Ready"),
        "features": model_bundle.get("feature_names", []),
        "endpoints": {
            "predict": "POST /predict_bust_risk",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["System"])
def health():
    """Liveness probe."""
    is_ready = "model" in model_bundle and "explainer" in model_bundle
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet loaded into memory"
        )
    return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post(
    "/predict_bust_risk",
    response_model=BustRiskResponse,
    status_code=status.HTTP_200_OK,
    tags=["Forecast Diagnostics"]
)
def predict_bust_risk(forecast_data: NWPForecastInput):
    """
    Main Bust Prediction and Explainability Endpoint.
    
    1. Ingests NWP forecast and synoptic pre-conditions.
    2. Runs inference through the champion ensemble model.
    3. Calculates confidence/risk probability (0% to 100%).
    4. Computes SHAP values to extract Top-3 Root Cause Drivers.
    5. Returns coordinates, risk percentage, high-risk flag, and explainability factors.
    """
    if "model" not in model_bundle:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle not initialized. Ensure bust_model.pkl exists."
        )

    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    # Prepare DataFrame matching exact feature schema
    sample_dict = {
        "predicted_temp": forecast_data.predicted_temp,
        "predicted_rainfall": forecast_data.predicted_rainfall,
        "wind_shear": forecast_data.wind_shear,
        "pressure_drop": forecast_data.pressure_drop,
    }
    input_df = pd.DataFrame([sample_dict])[feature_names].astype(float)

    # Run Model Inference
    try:
        probabilities = model.predict_proba(input_df)[0]
        # Class 1 is 'is_bust'
        bust_prob = float(probabilities[1])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}"
        )

    # Convert to Percentage (0 to 100%)
    risk_percentage = round(bust_prob * 100.0, 2)

    # Risk Warning Decision Threshold (50%)
    is_high_risk = risk_percentage >= 50.0

    # Categorical Risk Tiering
    if risk_percentage >= 75.0:
        risk_level = "CRITICAL"
    elif risk_percentage >= 50.0:
        risk_level = "HIGH"
    elif risk_percentage >= 30.0:
        risk_level = "ELEVATED"
    else:
        risk_level = "LOW"

    # Compute Root Cause Explanation using SHAP
    try:
        explainability_factors = explain_single_prediction(input_df, top_k=3)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation generation failed: {str(exc)}"
        )

    return BustRiskResponse(
        coordinates=CoordinateModel(
            latitude=forecast_data.latitude,
            longitude=forecast_data.longitude
        ),
        risk_probability_percentage=risk_percentage,
        is_high_risk_warning=is_high_risk,
        risk_level=risk_level,
        explainability_factors=explainability_factors,
        evaluated_at=datetime.now(timezone.utc).isoformat()
    )


# Standard Uvicorn Startup
if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
