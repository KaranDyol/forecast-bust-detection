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
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
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
    "lead_day",
    "relative_humidity",
    "cape",
    "model_temp_spread",
    "model_rain_spread",
]
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
TARGET_COL = "is_bust"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed_weather_data.csv"
MODEL_PATH = BASE_DIR / "bust_model.pkl"
PIPELINE_VERSION = "v3-multihorizon-spread-10728"


def align_feature_frame(
    df: pd.DataFrame,
    feature_names: List[str] = OPERATIONAL_FEATURES,
) -> pd.DataFrame:
    """Return a clean float DataFrame aligned to feature_names, backfilling defaults for legacy inputs."""
    aligned = df.copy()
    for col in feature_names:
        if col not in aligned.columns:
            aligned[col] = DEFAULT_FEATURE_VALUES.get(col, 0.0)
        else:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(DEFAULT_FEATURE_VALUES.get(col, 0.0))
    return aligned[feature_names].astype(float)


def _load_and_split_chronologically(
    csv_path: Union[str, Path] = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load processed dataset and apply a strict chronological train/test split when
    `valid_date` is present (out-of-time evaluation), else stratified random split.
    """
    file_path = Path(csv_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input dataset not found at: {file_path.resolve()}")

    df = pd.read_csv(file_path)
    core_required = ["predicted_temp", "predicted_rainfall", "wind_shear", "pressure_drop", TARGET_COL]
    missing = [c for c in core_required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if "valid_date" in df.columns and df["valid_date"].notna().sum() > 0:
        sort_cols = [c for c in ["valid_date", "station_name", "lead_day"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        unique_dates = sorted(df["valid_date"].astype(str).unique())
        if len(unique_dates) >= 5:
            split_idx = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * (1.0 - test_size))))
            train_dates = set(unique_dates[:split_idx])
            train_mask = df["valid_date"].astype(str).isin(train_dates)
            df_train = df.loc[train_mask].copy()
            df_test = df.loc[~train_mask].copy()
            split_meta = {
                "validation_scheme": "chronological_holdout",
                "train_range": f"{unique_dates[0]} .. {unique_dates[split_idx - 1]}",
                "test_range": f"{unique_dates[split_idx]} .. {unique_dates[-1]}",
            }
        else:
            df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COL])
            split_meta = {"validation_scheme": "stratified_random", "train_range": "n/a", "test_range": "n/a"}
    else:
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COL])
        split_meta = {"validation_scheme": "stratified_random", "train_range": "n/a", "test_range": "n/a"}

    X_train = align_feature_frame(df_train, OPERATIONAL_FEATURES)
    X_test = align_feature_frame(df_test, OPERATIONAL_FEATURES)
    y_train = df_train[TARGET_COL].astype(int).copy()
    y_test = df_test[TARGET_COL].astype(int).copy()
    return X_train, X_test, y_train, y_test, df_train, df_test, split_meta


def load_and_split_data(
    csv_path: Union[str, Path] = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> TupleData:
    """
    Load processed dataset and split into train and test partitions (chronologically when dates exist).
    Excludes ground-truth observations and error columns to avoid data leakage.
    """
    X_train, X_test, y_train, y_test, _df_tr, _df_te, _meta = _load_and_split_chronologically(
        csv_path=csv_path,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def _calibrate_classifier(
    base_estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Any, str, np.ndarray, np.ndarray]:
    """
    Apply isotonic probability calibration via internal out-of-fold CV on X_train only.
    Returns (model_for_inference, calibration_method, y_pred_test, y_prob_test).
    """
    raw_pred = base_estimator.predict(X_test)
    raw_prob = base_estimator.predict_proba(X_test)[:, 1]
    try:
        pos_count = int((y_train == 1).sum())
        cv_folds = 3 if pos_count >= 15 else 2
        if pos_count < 6:
            return base_estimator, "none", raw_pred, raw_prob

        calibrated = CalibratedClassifierCV(estimator=clone(base_estimator), method="isotonic", cv=cv_folds)
        calibrated.fit(X_train, y_train)
        cal_prob = calibrated.predict_proba(X_test)[:, 1]
        cal_pred = (cal_prob >= 0.5).astype(int)
        cal_auc = roc_auc_score(y_test, cal_prob)
        raw_auc = roc_auc_score(y_test, raw_prob)
        cal_brier = brier_score_loss(y_test, cal_prob)
        raw_brier = brier_score_loss(y_test, raw_prob)

        if cal_auc >= raw_auc - 0.015 and cal_brier <= raw_brier + 0.01:
            return calibrated, "isotonic", cal_pred, cal_prob
    except Exception as exc:
        print(f"[!] Calibration fallback to uncalibrated champion ({exc})")

    return base_estimator, "none", raw_pred, raw_prob


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train XGBoost, Random Forest, and a Soft-Voting Ensemble, evaluate chronological
    discrimination & Brier calibration, and calibrate the champion probabilities.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    xgb_clf = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        base_score=0.5,
        eval_metric="logloss",
        random_state=random_state,
    )

    rf_clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=7,
        class_weight="balanced",
        min_samples_split=6,
        random_state=random_state,
        n_jobs=-1
    )

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
    best_raw_model = None

    print("\n--- MODEL BENCHMARKING ON CHRONOLOGICAL HOLD-OUT TEST SET ---")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        eval_results[name] = {
            "model": model,
            "roc_auc": round(float(auc), 4),
            "brier_score": round(float(brier), 4),
            "f1": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "accuracy": round(float(acc), 4),
        }

        print(f"[{name}] AUC: {auc:.4f} | Brier: {brier:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_raw_model = model

    cal_model, calibration_method, cal_pred, cal_prob = _calibrate_classifier(
        best_raw_model, X_train, y_train, X_test, y_test
    )
    if calibration_method != "none":
        eval_results[best_name]["roc_auc"] = round(float(roc_auc_score(y_test, cal_prob)), 4)
        eval_results[best_name]["brier_score"] = round(float(brier_score_loss(y_test, cal_prob)), 4)
        eval_results[best_name]["f1"] = round(float(f1_score(y_test, cal_pred, zero_division=0)), 4)
        eval_results[best_name]["precision"] = round(float(precision_score(y_test, cal_pred, zero_division=0)), 4)
        eval_results[best_name]["recall"] = round(float(recall_score(y_test, cal_pred, zero_division=0)), 4)
        eval_results[best_name]["accuracy"] = round(float(accuracy_score(y_test, cal_pred)), 4)
        eval_results[best_name]["model"] = cal_model

    print(
        f"\n[*] Selected Champion Architecture: {best_name} "
        f"(Calibration: {calibration_method} | ROC-AUC: {eval_results[best_name]['roc_auc']:.4f} | "
        f"Brier: {eval_results[best_name]['brier_score']:.4f})"
    )

    return {
        "champion_name": best_name,
        "champion_model": cal_model,
        "raw_champion_model": best_raw_model,
        "xgb_model": xgb_clf,
        "raw_rf_model": rf_clf,
        "calibration_method": calibration_method,
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
    destination: Path = MODEL_PATH,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Serialize the complete model artifact bundle with metadata, explainer, and optional aux heads.
    """
    bundle = {
        "model": model,
        "explainer": explainer,
        "feature_names": feature_names,
        "metadata": metadata,
    }
    if extra_fields:
        bundle.update(extra_fields)
    joblib.dump(bundle, destination)
    print(f"\n[+] Trained model bundle successfully saved to: {destination.resolve()}")


def _fit_auxiliary_head(
    X_train: pd.DataFrame,
    y_train_sub: pd.Series,
    X_test: pd.DataFrame,
    y_test_sub: pd.Series,
    label: str,
    random_state: int = 42,
) -> Tuple[Optional[Any], Optional[float], Optional[float]]:
    """Train a calibrated XGBoost sub-head for temperature or rainfall bust decomposition."""
    pos_train = int((y_train_sub == 1).sum())
    pos_test = int((y_test_sub == 1).sum())
    if pos_train < 10 or pos_test < 3:
        return None, None, None

    neg_train = int((y_train_sub == 0).sum())
    spw = float(neg_train / max(pos_train, 1))
    clf = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        base_score=0.5,
        eval_metric="logloss",
        random_state=random_state,
    )
    clf.fit(X_train, y_train_sub)
    cal_head, _cal_method, _pred, cal_prob = _calibrate_classifier(
        clf, X_train, y_train_sub, X_test, y_test_sub
    )
    auc = round(float(roc_auc_score(y_test_sub, cal_prob)), 4)
    brier = round(float(brier_score_loss(y_test_sub, cal_prob)), 4)
    print(f"[+] Auxiliary {label} head trained: ROC-AUC={auc:.4f} | Brier={brier:.4f}")
    return cal_head, auc, brier


def get_shap_explanation(
    input_data: Union[Dict[str, float], pd.DataFrame, List[float]],
    model_bundle_path: Union[str, Path] = MODEL_PATH,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Real-time inference & Explainable AI helper function.
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
        df_raw = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        cols = feature_names[: len(input_data)]
        df_raw = pd.DataFrame([input_data], columns=cols)
    elif isinstance(input_data, pd.DataFrame):
        df_raw = input_data.head(1).copy()
    else:
        raise TypeError("input_data must be a dict, pandas DataFrame, or list of feature values")

    # Align and backfill optional multi-horizon defaults cleanly
    df_input = align_feature_frame(df_raw, feature_names)

    # Model inference
    pred_class = int(model.predict(df_input)[0])
    prob = float(model.predict_proba(df_input)[0, 1])

    temp_model = bundle.get("temp_model")
    rain_model = bundle.get("rain_model")
    temp_prob = round(float(temp_model.predict_proba(df_input)[0, 1]), 4) if temp_model is not None else None
    rain_prob = round(float(rain_model.predict_proba(df_input)[0, 1]), 4) if rain_model is not None else None

    # Compute instance SHAP values using native TreeExplainer
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

    result: Dict[str, Any] = {
        "predicted_bust": pred_class,
        "bust_probability": round(prob, 4),
        "status": "BUST ALERT: High Probability of Forecast Failure" if pred_class == 1 else "NOMINAL: Forecast Reliable",
        "top_contributing_features": top_drivers,
    }
    if temp_prob is not None:
        result["temp_bust_probability"] = temp_prob
    if rain_prob is not None:
        result["rain_bust_probability"] = rain_prob
    return result


def train_and_build_bundle(
    csv_path: Union[str, Path] = DATA_PATH,
    save_to_disk: bool = True,
    model_destination: Path = MODEL_PATH,
) -> Dict[str, Any]:
    """
    Train benchmark classifiers on the processed weather dataset using a strict
    chronological hold-out split, calibrate probabilities, fit auxiliary temp/rain
    sub-heads, build the SHAP TreeExplainer bundle, and optionally persist to disk.
    """
    X_train, X_test, y_train, y_test, df_train, df_test, split_meta = _load_and_split_chronologically(csv_path)
    sample_count = int(len(y_train) + len(y_test))
    bust_rate = round(float((y_train.sum() + y_test.sum()) / max(sample_count, 1)), 4)

    print(f"[+] Validation Scheme: {split_meta['validation_scheme']} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[+] Operational Features ({len(OPERATIONAL_FEATURES)}): {OPERATIONAL_FEATURES}")

    train_output = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    champion_name = train_output["champion_name"]
    champion_model = train_output["champion_model"]

    # TreeExplainer requires an uncalibrated native tree estimator (never CalibratedClassifierCV or VotingClassifier)
    explainer_model = (
        train_output["xgb_model"]
        if "Ensemble" in champion_name
        else train_output["raw_champion_model"]
    )
    try:
        explainer, _shap_vals, global_drivers = compute_shap_explainability(explainer_model, X_test)
    except Exception as exc:
        rf_fallback = train_output.get("raw_rf_model")
        if rf_fallback is None or rf_fallback is explainer_model:
            raise
        print(f"[!] SHAP TreeExplainer fallback to raw RandomForest ({exc})")
        explainer_model = rf_fallback
        explainer, _shap_vals, global_drivers = compute_shap_explainability(rf_fallback, X_test)

    temp_model: Optional[Any] = None
    rain_model: Optional[Any] = None
    temp_auc: Optional[float] = None
    rain_auc: Optional[float] = None
    if "is_temp_bust" in df_train.columns and "is_temp_bust" in df_test.columns:
        temp_model, temp_auc, _ = _fit_auxiliary_head(
            X_train, df_train["is_temp_bust"].astype(int), X_test, df_test["is_temp_bust"].astype(int), "Temperature Bust"
        )
    if "is_rain_bust" in df_train.columns and "is_rain_bust" in df_test.columns:
        rain_model, rain_auc, _ = _fit_auxiliary_head(
            X_train, df_train["is_rain_bust"].astype(int), X_test, df_test["is_rain_bust"].astype(int), "Rainfall Bust"
        )

    raw_metrics = train_output["eval_results"][champion_name]
    clean_metrics = {k: v for k, v in raw_metrics.items() if k != "model"}
    clean_metrics["calibration"] = train_output["calibration_method"]
    clean_metrics["validation_scheme"] = split_meta["validation_scheme"]
    if temp_auc is not None:
        clean_metrics["temp_head_auc"] = temp_auc
    if rain_auc is not None:
        clean_metrics["rain_head_auc"] = rain_auc

    global_drivers_list = global_drivers.to_dict(orient="records")
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "training_source": "Open-Meteo Multi-Horizon Archive (18 Indian Stations, Day+1/3/5/7 ECMWF vs GFS)",
        "validation_scheme": split_meta["validation_scheme"],
        "train_date_range": split_meta["train_range"],
        "test_date_range": split_meta["test_range"],
        "calibration": train_output["calibration_method"],
        "lead_days": [1, 3, 5, 7],
        "sample_count": sample_count,
        "bust_rate": bust_rate,
        "champion_name": champion_name,
        "metrics": clean_metrics,
        "global_drivers": global_drivers_list,
    }
    extra_fields = {
        "explainer_model": explainer_model,
        "temp_model": temp_model,
        "rain_model": rain_model,
    }

    if save_to_disk:
        try:
            save_model_artifact(
                model=champion_model,
                explainer=explainer,
                feature_names=OPERATIONAL_FEATURES,
                metadata=metadata,
                destination=model_destination,
                extra_fields=extra_fields,
            )
        except OSError as exc:
            print(f"[!] Could not write model artifact to {model_destination}: {exc}")

    try:
        from db import register_model_run

        register_model_run(
            model_version=PIPELINE_VERSION,
            champion_name=champion_name,
            training_source=metadata["training_source"],
            train_samples=sample_count,
            bust_rate=bust_rate,
            metrics=clean_metrics,
            global_drivers=global_drivers_list,
            artifact_uri=model_destination.name,
            set_active=True,
        )
    except Exception as exc:
        print(f"[!] Model registry DB update skipped: {exc}")

    return {
        "model": champion_model,
        "explainer": explainer,
        "explainer_model": explainer_model,
        "temp_model": temp_model,
        "rain_model": rain_model,
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
        "pressure_drop": 18.2,
        "lead_day": 5,
        "relative_humidity": 88.0,
        "cape": 1450.0,
        "model_temp_spread": 3.4,
        "model_rain_spread": 18.5,
    }
    print("\n--- TEST RUN OF get_shap_explanation() ON SEVERE TEST CASE ---")
    print(f"Sample Input: {sample_severe_case}")
    explanation = get_shap_explanation(sample_severe_case)
    print("\nPrediction & Root Cause Output (JSON formatted):")
    print(json.dumps(explanation, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
