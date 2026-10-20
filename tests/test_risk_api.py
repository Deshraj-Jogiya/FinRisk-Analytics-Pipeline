"""Scoring endpoint checks. Needs the trained model: run etl/pipeline.py and models/train.py first (CI does)."""
import os
import statistics
import sys
import time

import pytest
from fastapi.testclient import TestClient

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
MODEL = os.path.join(BASE, "models", "risk_model.pkl")
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="train the model first")

from api.app import app  # noqa: E402

SAFE = {"credit_score": 790, "annual_income": 120000, "employment_years": 10, "debt_amount": 5000, "loan_amount": 15000,
        "term_months": 36, "loan_purpose": "Home Improvement", "interest_rate": 6.5, "prior_delinquencies": 0,
        "credit_utilization": 0.1}
RISKY = {**SAFE, "credit_score": 480, "annual_income": 28000, "employment_years": 0, "debt_amount": 9000,
         "interest_rate": 18.0, "prior_delinquencies": 6, "credit_utilization": 0.95}


def test_risky_applicant_scores_higher_and_is_declined():
    client = TestClient(app)
    safe = client.post("/api/predict", json=SAFE).json()
    risky = client.post("/api/predict", json=RISKY).json()
    assert safe["default_risk_probability"] < 0.2 < 0.8 <= risky["default_risk_probability"]
    assert risky["decision"] == "REJECTED" and safe["decision"] == "APPROVED"


def test_median_scoring_latency_is_under_100_ms():
    client = TestClient(app)
    client.post("/api/predict", json=SAFE)  # first call loads the model
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        assert client.post("/api/predict", json=SAFE).status_code == 200
        timings.append((time.perf_counter() - start) * 1000)
    assert statistics.median(timings) < 100, f"median {statistics.median(timings):.1f} ms"
