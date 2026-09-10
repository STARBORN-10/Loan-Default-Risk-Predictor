import pytest
import numpy as np
import pandas as pd
from app.inference import get_predictor, LoanRiskPredictor

@pytest.fixture(scope="module")
def predictor():
    return get_predictor()

def test_predictor_initialization(predictor):
    """Verifies that all 4 artifacts + training medians are loaded properly."""
    assert predictor.model is not None
    assert predictor.scaler is not None
    assert predictor.feature_columns is not None
    assert len(predictor.feature_columns) == 173
    assert predictor.threshold == 0.53
    assert hasattr(predictor.scaler, 'mean_')
    assert predictor.scaler.mean_.shape[0] == 173
    assert predictor.model.coef_.shape == (1, 173)

def test_low_risk_applicant(predictor):
    """Tests a high-creditworthiness prime borrower."""
    low_risk_input = {
        "loan_amnt": 10000,
        "term": "36 months",
        "int_rate": 6.5,
        "installment": 306.49,
        "grade": "A",
        "sub_grade": "A1",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 150000.0,
        "verification_status": "Verified",
        "purpose": "credit_card",
        "addr_state": "CA",
        "dti": 8.5,
        "delinq_2yrs": 0,
        "inq_last_6mths": 0,
        "open_acc": 12,
        "pub_rec": 0,
        "revol_bal": 5000,
        "revol_util": 15.0,
        "total_acc": 25,
        "earliest_cr_line": "Jan-2005",
        "issue_d": "Dec-2018"
    }

    result = predictor.predict(low_risk_input)

    assert "default_probability" in result
    assert "prediction" in result
    assert "risk_level" in result
    assert result["default_probability"] < 0.53
    assert result["prediction"] == 0
    assert result["risk_level"] in ["Low", "Medium"]
    assert len(result["top_risk_decreasing_factors"]) > 0

def test_high_risk_applicant(predictor):
    """Tests a distressed subprime borrower."""
    high_risk_input = {
        "loan_amnt": 35000,
        "term": "60 months",
        "int_rate": 28.5,
        "installment": 1100.0,
        "grade": "G",
        "sub_grade": "G5",
        "emp_length": "< 1 year",
        "home_ownership": "RENT",
        "annual_inc": 28000.0,
        "verification_status": "Not Verified",
        "purpose": "small_business",
        "addr_state": "FL",
        "dti": 45.0,
        "delinq_2yrs": 4,
        "inq_last_6mths": 5,
        "open_acc": 4,
        "pub_rec": 3,
        "revol_bal": 25000,
        "revol_util": 95.0,
        "total_acc": 8,
        "earliest_cr_line": "Jan-2016",
        "issue_d": "Dec-2018"
    }

    result = predictor.predict(high_risk_input)

    assert result["default_probability"] > 0.53
    assert result["prediction"] == 1
    assert result["risk_level"] == "High"
    assert len(result["top_risk_increasing_factors"]) > 0

def test_missing_and_partial_input(predictor):
    """Tests that missing fields are gracefully imputed without raising errors."""
    partial_input = {
        "loan_amnt": 12000,
        "annual_inc": 65000
    }

    result = predictor.predict(partial_input)

    assert isinstance(result["default_probability"], float)
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["prediction"] in [0, 1]
    assert result["features_used_count"] == 173

def test_feature_alignment_and_order(predictor):
    """Verifies that the preprocessed dataframe has exactly 173 features in the exact expected order."""
    raw_sample = {
        "loan_amnt": 15000,
        "int_rate": 12.0,
        "grade": "B",
        "sub_grade": "B3",
        "term": "36 months",
        "home_ownership": "RENT"
    }

    df_imputed = predictor.preprocess_input(raw_sample)

    assert list(df_imputed.columns) == predictor.feature_columns
    assert df_imputed.shape == (1, 173)
    assert not df_imputed.isnull().any().any()

def test_explain_prediction_structure(predictor):
    """Verifies SHAP explanation structure."""
    sample = {
        "loan_amnt": 20000,
        "int_rate": 18.0,
        "grade": "D",
        "annual_inc": 50000,
        "dti": 25.0
    }

    df_imputed = predictor.preprocess_input(sample)
    inc, dec = predictor.explain_prediction(df_imputed, top_k=3)

    assert isinstance(inc, list)
    assert isinstance(dec, list)
    for factor in inc:
        assert "feature" in factor
        assert "impact" in factor
        assert factor["impact"] > 0
    for factor in dec:
        assert "feature" in factor
        assert "impact" in factor
        assert factor["impact"] < 0
