"""
run_project.py
==============
Master orchestrator script for AI-Based Forecast Bust Detection.

Usage:
  python run_project.py --pipeline     # Run Phase 1 data pipeline & generate data
  python run_project.py --train        # Run Phase 2 model training & SHAP explainability
  python run_project.py --api          # Start Phase 3 FastAPI microservice
  python run_project.py --dashboard    # Start Phase 4 Streamlit dashboard
  python run_project.py --all          # Launch both FastAPI backend and Streamlit frontend
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


def run_pipeline():
    print("\n[>>>] Running Phase 1 Data Pipeline...")
    subprocess.run([PYTHON_EXE, str(BASE_DIR / "data_pipeline.py")], check=True)


def run_training():
    print("\n[>>>] Running Phase 2 Model Training & SHAP Engine...")
    subprocess.run([PYTHON_EXE, str(BASE_DIR / "train_model.py")], check=True)


def run_api():
    print("\n[>>>] Starting Phase 3 FastAPI Microservice on http://127.0.0.1:8000...")
    subprocess.run([PYTHON_EXE, "-m", "uvicorn", "main_api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"], cwd=BASE_DIR)


def run_dashboard():
    print("\n[>>>] Starting Phase 4 Streamlit Operational Hub on http://localhost:8501...")
    subprocess.run(
        [PYTHON_EXE, "-m", "streamlit", "run", str(BASE_DIR / "dashboard.py"), "--server.headless", "true", "--server.port", "8501"],
        cwd=BASE_DIR
    )


def run_all():
    print("\n" + "=" * 70)
    print("AI-BASED FORECAST BUST DETECTION: FULL-STACK LAUNCH")
    print("=" * 70)

    # 1. Check prerequisites
    if not (BASE_DIR / "processed_weather_data.csv").is_file():
        run_pipeline()
    if not (BASE_DIR / "bust_model.pkl").is_file():
        run_training()

    print("\n[1/2] Spawning FastAPI microservice background daemon...")
    api_proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "uvicorn", "main_api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=BASE_DIR
    )

    print("[2/2] Spawning Streamlit operational console...")
    try:
        subprocess.run(
            [PYTHON_EXE, "-m", "streamlit", "run", str(BASE_DIR / "dashboard.py"), "--server.headless", "true", "--server.port", "8501"],
            cwd=BASE_DIR
        )
    finally:
        print("\n[*] Shutting down FastAPI microservice...")
        api_proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast Bust Detection Runner")
    parser.add_argument("--pipeline", action="store_true", help="Run Phase 1 Data Pipeline")
    parser.add_argument("--train", action="store_true", help="Run Phase 2 Model Training")
    parser.add_argument("--api", action="store_true", help="Launch Phase 3 FastAPI Service")
    parser.add_argument("--dashboard", action="store_true", help="Launch Phase 4 Streamlit Dashboard")
    parser.add_argument("--all", action="store_true", help="Launch both Backend and Frontend")

    args = parser.parse_args()

    if args.pipeline:
        run_pipeline()
    elif args.train:
        run_training()
    elif args.api:
        run_api()
    elif args.dashboard:
        run_dashboard()
    else:
        run_all()
