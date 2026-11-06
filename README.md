# AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts

An operational AI decision support layer over Numerical Weather Prediction (NWP) systems. This system identifies when a medium-range weather forecast has a high risk of failure (a "bust") using historical error dynamics and synoptic pre-conditions, backed by Explainable AI (SHAP) root-cause diagnostics.

---

## 🏛️ System Architecture

```text
[ NWP Model Outputs / Climatology ] (GFS / ECMWF / NCMRWF)
                │
                ▼
┌─────────────────────────────────┐
│ Phase 1: Data Pipeline          │ ──► processed_weather_data.csv
│ (data_pipeline.py)              │     - Synoptic error dynamics & bust thresholding
└─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ Phase 2: AI Training & SHAP     │ ──► bust_model.pkl
│ (train_model.py)                │     - Imbalance-weighted Champion Random Forest / XGBoost
└─────────────────────────────────┘     - Global & Local SHAP TreeExplainer
                │
                ▼
┌─────────────────────────────────┐
│ Phase 3: FastAPI Microservice   │ ──► http://127.0.0.1:8000
│ (main_api.py)                   │     - POST /predict_bust_risk
└─────────────────────────────────┘     - Sub-millisecond in-memory inference & SHAP
                │
                ▼
┌─────────────────────────────────┐
│ Phase 4: Operational Dashboard  │ ──► http://localhost:8501
│ (dashboard.py)                  │     - All-India Folium Risk Map (Red/Green alert zones)
└─────────────────────────────────┘     - Day 1 - Day 10 Lead Time Slider
                                        - Interactive SHAP Root Cause Attribution Charts
```

---

## 🚀 Quickstart Commands

All commands can be executed via the master orchestrator [`run_project.py`](file:///C:/Users/kdeol/.gemini/antigravity/scratch/forecast-bust-detection/run_project.py):

### 1. Run the Entire Full-Stack Solution Concurrently
```bash
python run_project.py --all
```
This boots the FastAPI microservice on `http://127.0.0.1:8000` and opens the Streamlit early-warning dashboard on `http://localhost:8501`.

### 2. Run Individual Components
- **Phase 1: Data Pipeline & Error Labeling:**
  ```bash
  python data_pipeline.py
  ```
- **Phase 2: Model Training & SHAP Explainability:**
  ```bash
  python train_model.py
  ```
- **Phase 3: FastAPI Backend Microservice:**
  ```bash
  uvicorn main_api:app --host 127.0.0.1 --port 8000 --reload
  # API Docs available at: http://127.0.0.1:8000/docs
  ```
- **Phase 4: Streamlit Early Warning Console:**
  ```bash
  streamlit run dashboard.py
  ```

---

## 📊 Phase-by-Phase Highlights

### Phase 1: Feature Engineering & Ground Truth
- Synoptic parameters: `predicted_temp`, `actual_temp`, `predicted_rainfall`, `actual_rainfall`, `wind_shear`, `pressure_drop`.
- Operational Bust Definition:
  $$\text{is\_bust} = 1 \iff (|\Delta T| > 3.0\,^\circ\text{C} \lor |\Delta \text{Rain}| > 15.0\,\text{mm})$$

### Phase 2: AI Model & Explainability
- Operational input features: strictly pre-event NWP and synoptic indicators (no target leakage).
- Benchmarking:
  - **Random Forest (Champion)**: ROC-AUC: `0.7005` | Precision: `0.2895` | Recall: `0.4904`
  - **XGBoost Classifier**: ROC-AUC: `0.6899` | F1: `0.3812`
  - **Ensemble (XGB+RF)**: ROC-AUC: `0.6978`
- Global Root-Cause Hierarchy: `wind_shear` ($0.0708$) > `pressure_drop` ($0.0631$) > `predicted_rainfall` ($0.0406$) > `predicted_temp` ($0.0193$).

### Phase 3: High-Performance FastAPI Microservice
- Typed Pydantic schema validation for spatial coordinates and synoptic variables.
- Structured response returning risk probability percentage ($0.0\%$ to $100.0\%$), high risk warning flag, and top-3 SHAP root causes with directional impact and meteorological descriptions.

### Phase 4: Operational Intelligence Hub
- Interactive Day 1 to Day 10 lead time slider.
- All-India geographical risk map with Folium:
  - 🔴 **Red Alert Nodes:** Bust Risk $> 70\%$
  - 🟠 **Elevated Risk:** $50\% - 70\%$
  - 🟢 **Reliable NWP:** $< 50\%$
- Interactive SHAP waterfall / horizontal bar charts quantifying local root cause drivers for any selected observation station.
