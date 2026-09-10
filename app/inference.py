import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class LoanRiskPredictor:
    """
    Production inference engine for Loan Default Risk Prediction.
    Strictly uses legitimate pre-loan/origination features without proprietary Sub-grades or post-loan leakage.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            possible_dirs = [
                os.path.join(base_dir, "models"),
                os.path.join(base_dir, "Models"),
                "models"
            ]
            for p in possible_dirs:
                if os.path.exists(os.path.join(p, "loan_default_model.pkl")):
                    models_dir = p
                    break
            if models_dir is None:
                raise FileNotFoundError("Could not locate models directory with loan_default_model.pkl.")

        self.models_dir = models_dir
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads model, scaler, feature columns, threshold, and training medians."""
        model_path = os.path.join(self.models_dir, "loan_default_model.pkl")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        features_path = os.path.join(self.models_dir, "feature_columns.pkl")
        threshold_path = os.path.join(self.models_dir, "threshold.pkl")
        medians_path = os.path.join(self.models_dir, "training_medians.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature columns file not found: {features_path}")
        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(features_path)
        self.threshold = float(joblib.load(threshold_path))

        if os.path.exists(medians_path):
            self.training_medians = joblib.load(medians_path)
        else:
            self.training_medians = pd.Series(0, index=self.feature_columns)

        self.expected_features_count = len(self.feature_columns)

    def preprocess_input(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms user applicant input into the exact 155 clean feature space.
        """
        data = raw_data.copy()

        # Date calculations (credit_history_years, issue_year, issue_month)
        if 'credit_history_years' not in data or pd.isna(data.get('credit_history_years')):
            if 'earliest_cr_line' in data and data['earliest_cr_line']:
                try:
                    earliest = pd.to_datetime(data['earliest_cr_line'])
                    issue = pd.to_datetime(data.get('issue_d', pd.Timestamp.now()))
                    data['credit_history_years'] = max(0.0, (issue - earliest).days / 365.25)
                except Exception:
                    data['credit_history_years'] = 10.0
            else:
                data['credit_history_years'] = 10.0

        if 'issue_year' not in data or pd.isna(data.get('issue_year')):
            if 'issue_d' in data and data['issue_d']:
                try:
                    data['issue_year'] = pd.to_datetime(data['issue_d']).year
                except Exception:
                    data['issue_year'] = 2018
            else:
                data['issue_year'] = 2018

        if 'issue_month' not in data or pd.isna(data.get('issue_month')):
            if 'issue_d' in data and data['issue_d']:
                try:
                    data['issue_month'] = pd.to_datetime(data['issue_d']).month
                except Exception:
                    data['issue_month'] = 12
            else:
                data['issue_month'] = 12

        # Auto-compute installment if not provided
        if 'installment' not in data or pd.isna(data.get('installment')) or data['installment'] == 0:
            loan_amnt = float(data.get('loan_amnt', 10000.0))
            int_rate = float(data.get('int_rate', 12.0))
            term_str = str(data.get('term', '36 months'))
            months = 36 if '36' in term_str else 60
            r = (int_rate / 100.0) / 12.0
            if r > 0:
                data['installment'] = (loan_amnt * r) / (1.0 - (1.0 + r)**(-months))
            else:
                data['installment'] = loan_amnt / months

        # Create DataFrame
        df_input = pd.DataFrame([data])

        # Standardize term string format
        if 'term' in df_input.columns and df_input['term'].notna().all():
            val = str(df_input['term'].iloc[0]).strip()
            df_input['term'] = f" {val}"

        # Categorical one-hot encoding
        one_hot_cols = [
            'term', 'emp_length', 'home_ownership', 'verification_status',
            'purpose', 'addr_state', 'initial_list_status', 'application_type',
            'disbursement_method'
        ]
        present_oh_cols = [c for c in one_hot_cols if c in df_input.columns]
        if present_oh_cols:
            df_encoded = pd.get_dummies(df_input, columns=present_oh_cols, drop_first=False)
        else:
            df_encoded = df_input.copy()

        # Reindex with exact feature columns
        df_aligned = df_encoded.reindex(columns=self.feature_columns, fill_value=0).copy()

        # Convert to numeric
        df_numeric = df_aligned.apply(pd.to_numeric, errors='coerce')

        # Impute missing values with training medians
        df_imputed = df_numeric.fillna(self.training_medians).fillna(0)

        return df_imputed

    def explain_prediction(self, df_imputed: pd.DataFrame, top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Calculates local SHAP feature contributions for the applicant.
        """
        X_scaled = self.scaler.transform(df_imputed)[0]
        coefs = self.model.coef_[0]
        contributions = X_scaled * coefs

        feature_map_readable = {
            "int_rate": "Interest Rate (APR)",
            "dti": "Debt-to-Income (DTI)",
            "annual_inc": "Annual Income",
            "loan_amnt": "Loan Amount",
            "installment": "Monthly Installment",
            "revol_util": "Revolving Credit Utilization",
            "inq_last_6mths": "Recent Credit Inquiries (6m)",
            "delinq_2yrs": "Past Delinquencies (2yr)",
            "open_acc": "Open Credit Accounts",
            "total_acc": "Total Credit History Lines",
            "credit_history_years": "Years of Credit History",
            "revol_bal": "Revolving Credit Balance",
            "term_ 60 months": "60-Month Loan Term",
            "home_ownership_RENT": "Housing Status: Rent",
            "home_ownership_MORTGAGE": "Housing Status: Mortgage",
            "home_ownership_OWN": "Housing Status: Own",
            "purpose_small_business": "Purpose: Small Business",
            "purpose_debt_consolidation": "Purpose: Debt Consolidation",
            "purpose_credit_card": "Purpose: Credit Card Refinance"
        }

        feature_impacts = []
        for feat, impact, val in zip(self.feature_columns, contributions, df_imputed.iloc[0]):
            if abs(impact) > 1e-4:
                readable_name = feature_map_readable.get(feat, feat.replace("_", " ").title())
                feature_impacts.append({
                    "feature": feat,
                    "readable_feature": readable_name,
                    "impact": float(np.round(impact, 4)),
                    "value": float(np.round(val, 2)) if isinstance(val, (int, float, np.number)) else str(val)
                })

        increasing = sorted([f for f in feature_impacts if f["impact"] > 0], key=lambda x: x["impact"], reverse=True)[:top_k]
        decreasing = sorted([f for f in feature_impacts if f["impact"] < 0], key=lambda x: x["impact"])[:top_k]

        return increasing, decreasing

    def predict(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the complete end-to-end prediction pipeline on a single loan applicant.
        """
        df_imputed = self.preprocess_input(raw_data)

        # Scale features
        X_scaled = self.scaler.transform(df_imputed)

        # Predict probability of default (Class 1)
        prob = float(self.model.predict_proba(X_scaled)[0, 1])

        # Apply classification threshold (0.52)
        pred = int(prob >= self.threshold)

        # Objective risk categorization based on probability distribution
        if prob < 0.30:
            risk_level = "LOW RISK"
            prediction_label = "LIKELY TO REPAY (APPROVED)"
            risk_desc = "Strong creditworthiness. Probability of default is significantly below the decision threshold."
        elif prob < 0.55:
            risk_level = "MEDIUM RISK"
            prediction_label = "CONDITIONAL REVIEW"
            risk_desc = "Moderate default probability. Standard monitoring or income verification recommended."
        else:
            risk_level = "HIGH RISK"
            prediction_label = "HIGH DEFAULT RISK (DECLINE)"
            risk_desc = "High default probability exceeds the risk tolerance threshold."

        # SHAP attribution
        top_inc, top_dec = self.explain_prediction(df_imputed, top_k=5)

        return {
            "default_probability": round(prob, 4),
            "default_percentage": round(prob * 100, 1),
            "prediction": pred,
            "prediction_label": prediction_label,
            "risk_level": risk_level,
            "risk_description": risk_desc,
            "threshold_used": self.threshold,
            "features_used_count": self.expected_features_count,
            "top_risk_increasing_factors": top_inc,
            "top_risk_decreasing_factors": top_dec
        }

# Global singleton
_predictor_instance = None

def get_predictor() -> LoanRiskPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = LoanRiskPredictor()
    return _predictor_instance
