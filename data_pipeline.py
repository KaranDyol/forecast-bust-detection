"""
data_pipeline.py
================
Foundational Data Pipeline & Feature Engineering for AI-Based Forecast Bust Detection.

This module is responsible for:
1. Generating a synthetic, meteorologically realistic dataset simulating Numerical
   Weather Prediction (NWP) model outputs vs. ground-truth observation/analysis,
   OR loading the Kaggle Weather-Dataset-Rattle (weatherAUS) CSV.
2. Computing temperature and precipitation forecast errors.
3. Labeling severe forecast failures ("Forecast Busts") based on meteorological thresholds:
   - Temperature Error > 3.0 deg C
   - OR Rainfall Error > 15.0 mm
4. Exporting a clean dataset ('processed_weather_data.csv') ready for ML training & SHAP explainability.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def generate_synthetic_weather_data(
    n_samples: int = 5000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a meteorologically grounded synthetic dataset simulating NWP forecast errors.

    Physical realism baked into the generation:
    - Baroclinic instability: Rapid pressure drops (falling barometer) correlate with
      approaching frontal boundaries and deep convective storm cells.
    - Vertical wind shear: High wind shear destabilizes atmospheric columns and
      organizes convection, which medium-range NWP models notoriously mispredict in terms
      of precipitation displacement and localized temperature drops.
    - NWP biases: Under high baroclinicity (high shear + steep pressure drops), NWP error
      variance increases non-linearly, leading to realistic forecast busts.
    """
    rng = np.random.default_rng(random_state)

    # 1. Atmospheric Drivers
    # Pressure drop over 24h (hPa): typically 0 to 25 hPa (mean ~6, skewed)
    pressure_drop = rng.gamma(shape=2.5, scale=2.8, size=n_samples)
    pressure_drop = np.clip(pressure_drop, 0.0, 32.0)

    # Deep layer vertical wind shear (0-6 km bulk shear in m/s): typical range 5 to 40 m/s
    wind_shear = rng.normal(loc=18.0, scale=7.0, size=n_samples)
    wind_shear = np.clip(wind_shear, 2.0, 48.0)

    # 2. Temperature Dynamics (deg C)
    # Underlying actual temperature (climatological distribution: ~5 to 38 deg C)
    actual_temp = rng.normal(loc=22.0, scale=6.5, size=n_samples)
    actual_temp = np.clip(actual_temp, 2.0, 42.0)

    # NWP Temperature Forecast error dynamics:
    # Standard baseline error is small (~0.8 deg C standard deviation).
    # High pressure drop or high wind shear introduces frontal timing error (up to 4-7 deg C error).
    instability_index = (pressure_drop / 15.0) ** 1.5 + (wind_shear / 25.0) ** 1.3
    temp_noise_scale = 0.8 + 0.9 * instability_index
    temp_error_component = rng.normal(loc=0.0, scale=temp_noise_scale, size=n_samples)

    # Add occasional abrupt frontal bust shocks (5-7% of forecasts suffer frontal displacement)
    bust_shock_mask = rng.random(n_samples) < 0.06
    frontal_shock = rng.choice([-1, 1], size=n_samples) * rng.uniform(3.5, 6.5, size=n_samples)
    temp_error_component[bust_shock_mask] += frontal_shock[bust_shock_mask]

    predicted_temp = actual_temp + temp_error_component
    predicted_temp = np.clip(predicted_temp, 0.0, 45.0)

    # 3. Rainfall Dynamics (mm)
    # Probability of precipitation (PoP) increases with pressure drop and wind shear
    rain_probability = 1.0 / (1.0 + np.exp(-(0.18 * pressure_drop + 0.05 * wind_shear - 2.8)))
    is_rain_day = rng.random(n_samples) < rain_probability

    # Actual rainfall amount (zero-inflated exponential/gamma distribution)
    actual_rainfall = np.zeros(n_samples)
    actual_rainfall[is_rain_day] = rng.exponential(scale=14.0, size=int(np.sum(is_rain_day)))

    # NWP Rainfall Forecast:
    # Models often smooth convective precipitation (under-predict localized deluge, over-predict drizzle)
    predicted_rainfall = np.zeros(n_samples)
    # Baseline model prediction with some displacement & intensity error
    predicted_rainfall[is_rain_day] = actual_rainfall[is_rain_day] * rng.uniform(0.5, 1.4, size=int(np.sum(is_rain_day))) + rng.normal(0, 2.0, size=int(np.sum(is_rain_day)))

    # False alarms (model predicts rain when none occurs)
    false_alarm_mask = (~is_rain_day) & (rng.random(n_samples) < 0.12)
    predicted_rainfall[false_alarm_mask] = rng.exponential(scale=4.5, size=int(np.sum(false_alarm_mask)))

    # Convective bust shocks (NWP completely misses convective initiation or produces extreme deluge bust)
    convective_bust_mask = (pressure_drop > 14.0) & (rng.random(n_samples) < 0.25)
    # Severe actual rain missed by model, or false severe storm predicted
    severe_error_type = rng.choice(["missed_deluge", "false_deluge"], size=n_samples)
    for i in np.where(convective_bust_mask)[0]:
        if severe_error_type[i] == "missed_deluge":
            actual_rainfall[i] += rng.uniform(18.0, 45.0)
        else:
            predicted_rainfall[i] += rng.uniform(18.0, 40.0)

    actual_rainfall = np.clip(np.round(actual_rainfall, 1), 0.0, 150.0)
    predicted_rainfall = np.clip(np.round(predicted_rainfall, 1), 0.0, 150.0)

    # Assemble raw DataFrame
    df = pd.DataFrame({
        "predicted_temp": np.round(predicted_temp, 2),
        "actual_temp": np.round(actual_temp, 2),
        "predicted_rainfall": predicted_rainfall,
        "actual_rainfall": actual_rainfall,
        "wind_shear": np.round(wind_shear, 2),
        "pressure_drop": np.round(pressure_drop, 2),
    })

    return df


def load_rattle_weather_dataset(file_path: str) -> pd.DataFrame:
    """
    Load and adapt the Kaggle Weather-Dataset-Rattle (weatherAUS.csv) into the required format.

    Maps:
    - MinTemp/MaxTemp -> actual_temp (MaxTemp proxy)
    - Simulated NWP predictions based on lagged observations and synoptic pressure drop
    - Rainfall -> actual_rainfall
    - WindGustSpeed -> wind_shear proxy
    - Pressure9am - Pressure3pm -> pressure_drop
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Weather dataset file not found at: {path.resolve()}")

    raw = pd.read_csv(path)

    # Select relevant columns and drop rows with critical missing values
    needed_cols = ["MaxTemp", "Rainfall", "WindGustSpeed", "Pressure9am", "Pressure3pm"]
    df = raw[needed_cols].dropna().copy()

    # Derived synoptic features
    df["actual_temp"] = df["MaxTemp"].astype(float)
    df["actual_rainfall"] = df["Rainfall"].astype(float)
    df["wind_shear"] = df["WindGustSpeed"].astype(float) * 0.514444  # knots to m/s
    df["pressure_drop"] = (df["Pressure9am"] - df["Pressure3pm"]).clip(lower=0.0)

    # Simulate realistic NWP medium-range forecast baseline with realistic forecast variance
    rng = np.random.default_rng(42)
    n = len(df)
    nwp_temp_noise = rng.normal(0, 1.2, size=n) + (df["pressure_drop"].values / 10.0) * rng.normal(0, 1.0, size=n)
    df["predicted_temp"] = np.round(df["actual_temp"] + nwp_temp_noise, 2)

    rain_noise = rng.normal(0, 2.5, size=n)
    df["predicted_rainfall"] = np.clip(np.round(df["actual_rainfall"] * rng.uniform(0.7, 1.3, size=n) + rain_noise, 1), 0.0, None)

    # Retain strictly required foundational columns
    cols = ["predicted_temp", "actual_temp", "predicted_rainfall", "actual_rainfall", "wind_shear", "pressure_drop"]
    return df[cols].reset_index(drop=True)


def compute_forecast_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate absolute forecast errors for temperature and precipitation.
    - temp_error = |predicted_temp - actual_temp|
    - rainfall_error = |predicted_rainfall - actual_rainfall|
    """
    df = df.copy()
    df["temp_error"] = (df["predicted_temp"] - df["actual_temp"]).abs().round(2)
    df["rainfall_error"] = (df["predicted_rainfall"] - df["actual_rainfall"]).abs().round(2)
    return df


def label_forecast_busts(
    df: pd.DataFrame,
    temp_threshold: float = 3.0,
    rainfall_threshold: float = 15.0
) -> pd.DataFrame:
    """
    Define and label a Forecast Bust:
    is_bust = 1 if (temp_error > temp_threshold OR rainfall_error > rainfall_threshold) else 0.
    """
    df = df.copy()
    is_bust_condition = (df["temp_error"] > temp_threshold) | (df["rainfall_error"] > rainfall_threshold)
    df["is_bust"] = is_bust_condition.astype(int)
    return df


def run_pipeline(
    dataset_path: Optional[str] = None,
    output_path: str = "processed_weather_data.csv",
    n_synthetic_samples: int = 5000,
    random_state: int = 42,
    temp_threshold: float = 3.0,
    rainfall_threshold: float = 15.0
) -> pd.DataFrame:
    """
    End-to-end execution of Phase 1 Data Pipeline:
    1. Loads Rattle dataset if supplied, otherwise generates synthetic meteorological dataset.
    2. Validates schema and cleans records (removes nulls/infinities).
    3. Calculates forecast error metrics.
    4. Labels forecast busts.
    5. Saves output to CSV and prints summary metrics.
    """
    print("=" * 70)
    print("AI-BASED FORECAST BUST DETECTION: PHASE 1 DATA PIPELINE")
    print("=" * 70)

    # 1. Ingestion / Generation
    if dataset_path and Path(dataset_path).exists():
        print(f"[+] Loading Kaggle Rattle dataset from: {dataset_path}")
        df_raw = load_rattle_weather_dataset(dataset_path)
    else:
        if dataset_path:
            print(f"[!] Warning: Specified path '{dataset_path}' not found. Falling back to synthetic generation.")
        print(f"[+] Generating {n_synthetic_samples} synthetic meteorological records...")
        df_raw = generate_synthetic_weather_data(n_samples=n_synthetic_samples, random_state=random_state)

    # 2. Schema check & Cleaning
    expected_cols = ["predicted_temp", "actual_temp", "predicted_rainfall", "actual_rainfall", "wind_shear", "pressure_drop"]
    assert all(col in df_raw.columns for col in expected_cols), "Missing mandatory meteorological columns!"

    df_clean = df_raw.dropna().copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

    # 3. Compute Errors
    df_with_errors = compute_forecast_errors(df_clean)

    # 4. Label Busts
    df_final = label_forecast_busts(
        df_with_errors,
        temp_threshold=temp_threshold,
        rainfall_threshold=rainfall_threshold
    )

    # 5. Persist Output
    out_file = Path(output_path)
    df_final.to_csv(out_file, index=False)
    print(f"[+] Processed data successfully written to: {out_file.resolve()}")

    # 6. Pipeline Summary
    total_records = len(df_final)
    bust_records = int(df_final["is_bust"].sum())
    bust_rate = (bust_records / total_records) * 100.0

    print("\n--- DATA PIPELINE SUMMARY ---")
    print(f"Total Records Processed : {total_records}")
    print(f"Bust Occurrences (Class 1): {bust_records} ({bust_rate:.2f}%)")
    print(f"Normal Forecasts (Class 0): {total_records - bust_records} ({100.0 - bust_rate:.2f}%)")
    print(f"Mean Temperature Error  : {df_final['temp_error'].mean():.2f} °C (Max: {df_final['temp_error'].max():.2f} °C)")
    print(f"Mean Rainfall Error     : {df_final['rainfall_error'].mean():.2f} mm (Max: {df_final['rainfall_error'].max():.2f} mm)")
    print("\nFirst 5 Records:")
    print(df_final.head())
    print("=" * 70)

    return df_final


if __name__ == "__main__":
    # Execute pipeline and write processed_weather_data.csv to working directory
    run_pipeline()
