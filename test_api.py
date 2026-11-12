"""
test_api.py
===========
Automated test suite verifying the FastAPI microservice end-to-end.
"""

import json
from fastapi.testclient import TestClient
from main_api import app

def test_api():
    print("=" * 70)
    print("TESTING FASTAPI MICROSERVICE ENDPOINTS")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Health check & Root
        res_root = client.get("/")
        assert res_root.status_code == 200, f"Root failed: {res_root.text}"
        print(f"[+] GET / -> {res_root.json()['service']} ({res_root.json()['status']})")

        res_health = client.get("/health")
        assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
        print(f"[+] GET /health -> Status: {res_health.json()['status']}")

        res_db = client.get("/db/health")
        assert res_db.status_code == 200, f"DB health check failed: {res_db.text}"
        print(f"[+] GET /db/health -> {res_db.json()['backend']} ({res_db.json()['stations']} stations)")

        # 2. Test Severe Weather Case (High Bust Probability expected)
        severe_payload = {
            "latitude": -33.8688,
            "longitude": 151.2093,
            "predicted_temp": 34.5,
            "predicted_rainfall": 28.0,
            "wind_shear": 42.0,
            "pressure_drop": 21.5,
            "relative_humidity": 85.0,
            "lead_time_hours": 72
        }
        print(f"\n[+] Testing POST /predict_bust_risk (Severe Storm Payload):")
        print(json.dumps(severe_payload, indent=2))

        res_severe = client.post("/predict_bust_risk", json=severe_payload)
        assert res_severe.status_code == 200, f"Prediction failed: {res_severe.text}"
        data_severe = res_severe.json()
        print("\n--- RESPONSE: SEVERE CASE ---")
        print(json.dumps(data_severe, indent=2))

        assert "coordinates" in data_severe
        assert data_severe["coordinates"]["latitude"] == -33.8688
        assert "risk_probability_percentage" in data_severe
        assert "is_high_risk_warning" in data_severe
        assert "explainability_factors" in data_severe
        assert len(data_severe["explainability_factors"]) == 3
        print(f"[+] High risk warning triggered: {data_severe['is_high_risk_warning']}")
        print(f"[+] Risk probability: {data_severe['risk_probability_percentage']}%")

        # 3. Test Mild / Calm Weather Case (Low Bust Probability expected)
        calm_payload = {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "predicted_temp": 21.0,
            "predicted_rainfall": 0.0,
            "wind_shear": 8.0,
            "pressure_drop": 2.1,
            "relative_humidity": 45.0,
            "lead_time_hours": 48
        }
        res_calm = client.post("/predict_bust_risk", json=calm_payload)
        assert res_calm.status_code == 200, f"Prediction failed: {res_calm.text}"
        data_calm = res_calm.json()
        print("\n--- RESPONSE: CALM CASE ---")
        print(json.dumps(data_calm, indent=2))
        print(f"[+] Calm Case Risk: {data_calm['risk_probability_percentage']}% (Warning: {data_calm['is_high_risk_warning']})")

        # 4. Input Validation Error Test (Latitude > 90)
        bad_payload = severe_payload.copy()
        bad_payload["latitude"] = 999.0
        res_bad = client.post("/predict_bust_risk", json=bad_payload)
        assert res_bad.status_code == 422, "Pydantic validation failed to catch bad coordinates"
        print("\n[+] Pydantic coordinate validation correctly returned 422 Unprocessable Entity for invalid latitude.")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL FASTAPI MICROSERVICE TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_api()
