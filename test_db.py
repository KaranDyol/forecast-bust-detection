"""
test_db.py
==========
Unit test suite verifying SQLAlchemy schema creation, idempotent upserts,
cross-dialect paired verification view, prediction logging, and model registry.
"""

from datetime import date
from pathlib import Path
import tempfile

import db


def test_database_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "unit_test.db"
        engine = db.get_engine(f"sqlite:///{db_file.as_posix()}")

        summary = db.bootstrap_database(engine=engine)
        assert summary["backend"] == "SQLite"
        assert summary["stations"] == 18, f"Expected 18 stations, got {summary['stations']}"

        inserted_fc = db.upsert_forecast_snapshots(
            [
                {
                    "station_name": "Mumbai (Santacruz)",
                    "issued_date": "2026-11-08",
                    "valid_date": "2026-11-11",
                    "lead_day": 3,
                    "predicted_temp": 33.5,
                    "predicted_rainfall": 45.0,
                    "wind_shear": 24.5,
                    "pressure_drop": 9.2,
                    "risk_percentage": 81.2,
                },
                {
                    "station_name": "New Delhi (Safdarjung)",
                    "issued_date": "2026-11-08",
                    "valid_date": "2026-11-11",
                    "lead_day": 3,
                    "predicted_temp": 29.0,
                    "predicted_rainfall": 0.0,
                    "wind_shear": 8.0,
                    "pressure_drop": 1.4,
                    "risk_percentage": 12.0,
                },
            ],
            nwp_model="ecmwf_ifs025",
            model_version="v2-india-archive-2682",
            engine=engine,
        )
        assert inserted_fc == 2

        # Re-upserting identical key must remain idempotent
        db.upsert_forecast_snapshots(
            [
                {
                    "station_name": "Mumbai (Santacruz)",
                    "issued_date": date(2026, 11, 8),
                    "valid_date": date(2026, 11, 11),
                    "lead_day": 3,
                    "predicted_temp": 33.5,
                    "predicted_rainfall": 45.0,
                    "wind_shear": 24.5,
                    "pressure_drop": 9.2,
                    "risk_percentage": 81.2,
                }
            ],
            nwp_model="ecmwf_ifs025",
            engine=engine,
        )

        db.upsert_verification_outcomes(
            [
                {
                    "station_name": "Mumbai (Santacruz)",
                    "obs_date": "2026-11-11",
                    "actual_temp": 30.0,
                    "actual_rainfall": 12.0,
                },
                {
                    "station_name": "New Delhi (Safdarjung)",
                    "obs_date": "2026-11-11",
                    "actual_temp": 28.5,
                    "actual_rainfall": 0.0,
                },
            ],
            engine=engine,
        )

        pairs = db.fetch_verified_training_records(engine=engine)
        assert len(pairs) == 2
        by_station = {row["station_name"]: row for row in pairs}
        assert by_station["Mumbai (Santacruz)"]["is_bust"] == 1
        assert by_station["New Delhi (Safdarjung)"]["is_bust"] == 0

        db.log_prediction(
            input_features={"predicted_temp": 33.5, "predicted_rainfall": 45.0, "wind_shear": 24.5, "pressure_drop": 9.2},
            risk_prob_pct=81.2,
            risk_level="CRITICAL",
            shap_factors=[{"feature": "predicted_rainfall", "shap_value": 0.21}],
            model_version="v2-india-archive-2682",
            latitude=19.076,
            longitude=72.8777,
            station_name="Mumbai (Santacruz)",
            engine=engine,
        )
        db.register_model_run(
            model_version="v2-india-archive-2682",
            champion_name="RandomForest",
            training_source="Open-Meteo NWP Archive",
            train_samples=2682,
            bust_rate=0.1119,
            metrics={"roc_auc": 0.824, "f1": 0.444, "precision": 0.351, "recall": 0.600},
            global_drivers=[{"Feature": "predicted_rainfall", "Mean_Abs_SHAP": 0.1904}],
            engine=engine,
        )

        final_summary = db.get_database_summary(engine=engine)
        assert final_summary["forecast_runs"] == 2
        assert final_summary["verification_outcomes"] == 2
        assert final_summary["verified_pairs"] == 2
        assert final_summary["prediction_logs"] == 1
        assert final_summary["model_runs"] == 1

        engine.dispose()

    print("[SUCCESS] SQLAlchemy DB roundtrip passed all assertions cleanly.")


if __name__ == "__main__":
    test_database_roundtrip()
