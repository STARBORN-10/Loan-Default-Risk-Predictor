import pytest
from starlette.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
    assert data["predict_endpoint"] == "/predict"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["features_count"] == 173
    assert data["threshold"] == 0.53
    assert data["model_type"] == "LogisticRegression"

def test_predict_endpoint_low_risk():
    payload = {
        "loan_amnt": 8000,
        "term": "36 months",
        "int_rate": 7.2,
        "grade": "A",
        "sub_grade": "A2",
        "annual_inc": 120000,
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "purpose": "credit_card",
        "addr_state": "CA",
        "dti": 10.0,
        "delinq_2yrs": 0,
        "inq_last_6mths": 0,
        "open_acc": 14,
        "pub_rec": 0,
        "revol_bal": 4000,
        "revol_util": 18.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert data["prediction"] == 0
    assert data["risk_level"] in ["Low", "Medium"]
    assert len(data["top_risk_decreasing_factors"]) > 0

def test_predict_endpoint_high_risk():
    payload = {
        "loan_amnt": 35000,
        "term": "60 months",
        "int_rate": 26.5,
        "grade": "F",
        "sub_grade": "F4",
        "annual_inc": 30000,
        "emp_length": "< 1 year",
        "home_ownership": "RENT",
        "verification_status": "Not Verified",
        "purpose": "debt_consolidation",
        "addr_state": "NY",
        "dti": 38.0,
        "delinq_2yrs": 3,
        "inq_last_6mths": 4,
        "open_acc": 5,
        "pub_rec": 2,
        "revol_bal": 20000,
        "revol_util": 90.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["risk_level"] == "High"
    assert len(data["top_risk_increasing_factors"]) > 0

def test_explain_endpoint():
    payload = {
        "loan_amnt": 15000,
        "int_rate": 15.0,
        "grade": "C",
        "annual_inc": 55000
    }
    response = client.post("/explain?top_k=5", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_risk_increasing_factors" in data
    assert "top_risk_decreasing_factors" in data
