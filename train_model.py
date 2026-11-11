"""
train_model.py
==============
Phase 2: AI Model Training, Evaluation, and SHAP Explainability Engine.

This module:
1. Loads 'processed_weather_data.csv' and extracts non-leaking synoptic/NWP features.
2. Trains both an XGBoost Classifier and a Random Forest Classifier with imbalance handling.
3. Evaluates models and selects the top-performing architecture (or Voting Ensemble).
4. Integrates SHAP TreeExplainer for global feature importance ("Root Cause Drivers").
5. Serializes the trained model bundle to 'bust_model.pkl'.
6. Provides the `get_shap_explanation(input_data)` helper function for real-time inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

# Type hint alias
TupleData = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]

# Operational features available at forecast issuance time (prior to ground truth observation)
OPERATIONAL_FEATURES = [
    "predicted_temp",
    "predicted_rainfall",
    "wind_shear",
    "pressure_drop",
]
TARGET_COL = "is_bust"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed_weather_data.csv"
MODEL_PATH = BASE_DIR / "bust_model.pkl"
PIPELINE_VERSION = "v2-india-archive-2682"


def load_and_split_data(
    csv_path: Union[str, Path] = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> TupleData:
    """
    Load processed dataset and split into train and test partitions.
    Excludes ground-truth observations and error columns to avoid data leakage.
    """
    file_path = Path(csv_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input dataset not found at: {file_path.resolve()}")

    df = pd.read_csv(file_path)

    missing = [c for c in OPERATIONAL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    X = df[OPERATIONAL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train XGBoost, Random Forest, and a Soft-Voting Ensemble.
    Compares metrics (ROC-AUC, F1, Precision, Recall) and returns the best model.
    """
    # Calculate scale_pos_weight to compensate for ~15% bust prevalence
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    # 1. XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        base_score=0.5,
        eval_metric="logloss",
        random_state=random_state,
    )

    # 2. Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=6,
        class_weight="balanced",
        min_samples_split=6,
        random_state=random_state,
        n_jobs=-1
    )

    # 3. Soft Voting Ensemble
    ensemble_clf = VotingClassifier(
        estimators=[
            ("xgb", xgb_clf),
            ("rf", rf_clf)
        ],
        voting="soft"
    )

    candidates = {
        "XGBoost": xgb_clf,
        "RandomForest": rf_clf,
        "Ensemble (XGB+RF)": ensemble_clf
    }

    eval_results = {}
    best_name = None
    best_auc = -1.0
    best_model = None

    print("\n--- MODEL BENCHMARKING ON HOLD-OUT TEST SET ---")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        eval_results[name] = {
            "model": model,
            "roc_auc": round(float(auc), 4),
            "f1": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "accuracy": round(float(acc), 4),
        }

        print(f"[{name}] AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # Selection criteria: prioritizing ROC-AUC & F1 balance
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    print(f"\n[*] Selected Champion Architecture: {best_name} (ROC-AUC: {best_auc:.4f})")

    # If ensemble was chosen, we also keep the trained XGBoost for native TreeExplainer
    return {
        "champion_name": best_name,
        "champion_model": best_model,
        "xgb_model": xgb_clf,
        "eval_results": eval_results,
    }


def compute_shap_explainability(
    model: Any,
    X_test: pd.DataFrame
) -> Tuple[shap.TreeExplainer, np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values using TreeExplainer to diagnose global and local Root Cause Drivers.
    """
    print("\n--- SHAP EXPLAINABILITY ENGINE INITIALIZATION ---")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # In modern SHAP versions, binary classification can return an array or list of arrays
    if isinstance(shap_values, list):
        # Index 1 corresponds to positive class (is_bust == 1)
        values_arr = shap_values[1]
    elif len(shap_values.shape) == 3:
        values_arr = shap_values[:, :, 1]
    else:
        values_arr = shap_values

    # Calculate global Mean Absolute SHAP values across all test samples
    mean_abs_shap = np.abs(values_arr).mean(axis=0)
    driver_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_Abs_SHAP": np.round(mean_abs_shap, 4)
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print("Global Root Cause Drivers (Ranked by Mean |SHAP| impact on Bust Risk):")
    for idx, row in driver_df.iterrows():
        print(f"  {idx + 1}. {row['Feature']:<20}: {row['Mean_Abs_SHAP']:.4f}")

    return explainer, values_arr, driver_df


def save_model_artifact(
    model: Any,
    explainer: Any,
    feature_names: List[str],
    metadata: Dict[str, Any],
    destination: Path = MODEL_PATH
) -> None:
    """
    Serialize the complete model artifact bundle with metadata and explainer.
    """
    bundle = {
        "model": model,
        "explainer": explainer,
        "feature_names": feature_names,
        "metadata": metadata,
    }
    joblib.dump(bundle, destination)
    print(f"\n[+] Trained model bundle successfully saved to: {destination.resolve()}")


def get_shap_explanation(
    input_data: Union[Dict[str, float], pd.DataFrame, List[float]],
    model_bundle_path: Union[str, Path] = MODEL_PATH,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Real-time inference & Explainable AI helper function.
    
    Accepts:
        input_data: dictionary of feature names/values, a DataFrame row, or list in standard order.
    Returns:
        JSON-serializable dictionary with:
        - predicted_bust: 0 or 1
        - bust_probability: probability between 0.0 and 1.0
        - top_contributing_features: top K features driving the prediction, with direction of impact.
    """
    bundle_file = Path(model_bundle_path)
    if not bundle_file.is_file():
        raise FileNotFoundError(f"Model artifact not found at {bundle_file.resolve()}. Run train_model.py first.")

    bundle = joblib.load(bundle_file)
    model = bundle["model"]
    explainer = bundle["explainer"]
    feature_names = bundle["feature_names"]

    # Normalize input into a 1-row DataFrame
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_input = pd.DataFrame([input_data], columns=feature_names)
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data[feature_names].head(1).copy()
    else:
        raise TypeError("input_data must be a dict, pandas DataFrame, or list of feature values")

    # Ensure all required features are present in correct column order
    df_input = df_input[feature_names].astype(float)

    # Model inference
    pred_class = int(model.predict(df_input)[0])
    prob = float(model.predict_proba(df_input)[0, 1])

    # Compute instance SHAP values
    raw_shap = explainer.shap_values(df_input)
    if isinstance(raw_shap, list):
        shap_vec = raw_shap[1][0]
    elif len(raw_shap.shape) == 3:
        shap_vec = raw_shap[0, :, 1]
    else:
        shap_vec = raw_shap[0]

    # Rank features by absolute impact
    feature_impacts = []
    for feat_name, s_val in zip(feature_names, shap_vec):
        val = float(df_input[feat_name].iloc[0])
        impact_dir = "increases_bust_risk" if s_val > 0 else "decreases_bust_risk"
        feature_impacts.append({
            "feature": feat_name,
            "feature_value": round(val, 2),
            "shap_value": round(float(s_val), 4),
            "abs_impact": abs(float(s_val)),
            "impact": impact_dir
        })

    # Sort descending by absolute SHAP contribution
    feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
    top_drivers = [
        {k: v for k, v in item.items() if k != "abs_impact"}
        for item in feature_impacts[:top_k]
    ]

    return {
        "predicted_bust": pred_class,
        "bust_probability": round(prob, 4),
        "status": "BUST ALERT: High Probability of Forecast Failure" if pred_class == 1 else "NOMINAL: Forecast Reliable",
        "top_contributing_features": top_drivers
    }


def train_and_build_bundle(
    csv_path: Union[str, Path] = DATA_PATH,
    save_to_disk: bool = True,
    model_destination: Path = MODEL_PATH,
) -> Dict[str, Any]:
    """
    Train benchmark classifiers on the processed weather dataset, build the
    SHAP TreeExplainer bundle, and optionally persist to disk.
    """
    X_train, X_test, y_train, y_test = load_and_split_data(csv_path)
    sample_count = int(len(y_train) + len(y_test))
    bust_rate = round(float((y_train.sum() + y_test.sum()) / max(sample_count, 1)), 4)

    print(f"[+] Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")
    print(f"[+] Operational Features: {OPERATIONAL_FEATURES}")

    train_output = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    champion_name = train_output["champion_name"]
    champion_model = train_output["champion_model"]

    explainer_model = train_output["xgb_model"] if "Ensemble" in champion_name else champion_model
    try:
        explainer, _shap_vals, global_drivers = compute_shap_explainability(explainer_model, X_test)
    except Exception as exc:
        rf_fallback = next(
            (
                entry["model"]
                for name, entry in train_output["eval_results"].items()
                if "Random" in name and isinstance(entry, dict) and "model" in entry
            ),
            None,
        )
        if rf_fallback is None or rf_fallback is explainer_model:
            raise
        print(f"[!] SHAP TreeExplainer fallback to RandomForest ({exc})")
        explainer, _shap_vals, global_drivers = compute_shap_explainability(rf_fallback, X_test)

    raw_metrics = train_output["eval_results"][champion_name]
    clean_metrics = {k: v for k, v in raw_metrics.items() if k != "model"}

    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "training_source": "Open-Meteo NWP Archive (18 Indian Stations, Day+3 vs Day-0)",
        "sample_count": sample_count,
        "bust_rate": bust_rate,
        "champion_name": champion_name,
        "metrics": clean_metrics,
        "global_drivers": global_drivers.to_dict(orient="records"),
    }

    if save_to_disk:
        try:
            save_model_artifact(
                model=champion_model,
                explainer=explainer,
                feature_names=OPERATIONAL_FEATURES,
                metadata=metadata,
                destination=model_destination,
            )
        except OSError as exc:
            print(f"[!] Could not write model artifact to {model_destination}: {exc}")

    return {
        "model": champion_model,
        "explainer": explainer,
        "feature_names": OPERATIONAL_FEATURES,
        "metadata": metadata,
    }


def main():
    print("=" * 70)
    print("AI-BASED FORECAST BUST DETECTION: PHASE 2 TRAINING & SHAP ENGINE")
    print("=" * 70)

    train_and_build_bundle(
        csv_path=DATA_PATH,
        save_to_disk=True,
        model_destination=MODEL_PATH,
    )

    # Test Sample Explanation Helper
    sample_severe_case = {
        "predicted_temp": 32.5,
        "predicted_rainfall": 24.0,
        "wind_shear": 38.5,
        "pressure_drop": 18.2
    }
    print("\n--- TEST RUN OF get_shap_explanation() ON SEVERE TEST CASE ---")
    print(f"Sample Input: {sample_severe_case}")
    explanation = get_shap_explanation(sample_severe_case)
    print("\nPrediction & Root Cause Output (JSON formatted):")
    print(json.dumps(explanation, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
