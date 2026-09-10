from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn

from app.inference import get_predictor

app = FastAPI(
    title="Loan Default Risk Predictor API",
    description="Production REST API for predicting LendingClub loan default risk with SHAP explainability.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoanApplicantInput(BaseModel):
    # Core Loan Details
    loan_amnt: float = Field(10000.0, ge=500.0, le=50000.0, description="Requested loan amount in USD")
    term: str = Field("36 months", description="Loan term: '36 months' or '60 months'")
    int_rate: float = Field(12.5, ge=1.0, le=40.0, description="Interest rate percentage")
    installment: Optional[float] = Field(None, ge=0.0, description="Monthly installment in USD (auto-computed if None)")
    grade: str = Field("B", description="Loan grade: 'A', 'B', 'C', 'D', 'E', 'F', 'G'")
    sub_grade: Optional[str] = Field("B2", description="Loan sub-grade: 'A1' through 'G5'")
    
    # Financial & Employment Background
    annual_inc: float = Field(65000.0, ge=0.0, description="Annual self-reported income in USD")
    emp_length: str = Field("5 years", description="Employment length: '< 1 year', '1 year', ..., '10+ years', 'Unknown'")
    home_ownership: str = Field("RENT", description="Home ownership: 'RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE', 'ANY'")
    verification_status: str = Field("Source Verified", description="Income verification: 'Not Verified', 'Source Verified', 'Verified'")
    purpose: str = Field("debt_consolidation", description="Loan purpose: 'debt_consolidation', 'credit_card', 'home_improvement', etc.")
    addr_state: str = Field("CA", description="US State code (2-letter)")
    
    # Credit Profile & Risk Metrics
    dti: float = Field(16.5, ge=0.0, le=100.0, description="Debt-to-Income ratio (%)")
    delinq_2yrs: Optional[float] = Field(0.0, ge=0.0, description="30+ days past-due delinquencies in past 2 years")
    inq_last_6mths: Optional[float] = Field(0.0, ge=0.0, description="Credit inquiries in past 6 months")
    open_acc: Optional[float] = Field(10.0, ge=0.0, description="Number of open credit lines")
    pub_rec: Optional[float] = Field(0.0, ge=0.0, description="Number of derogatory public records")
    revol_bal: Optional[float] = Field(8500.0, ge=0.0, description="Total credit revolving balance in USD")
    revol_util: Optional[float] = Field(45.0, ge=0.0, le=150.0, description="Revolving line utilization rate (%)")
    total_acc: Optional[float] = Field(20.0, ge=0.0, description="Total credit lines in history")
    
    # Dates
    earliest_cr_line: Optional[str] = Field("Jan-2010", description="Month/Year of earliest credit line (e.g. 'Jan-2010')")
    issue_d: Optional[str] = Field("Dec-2018", description="Loan issue date (e.g. 'Dec-2018')")
    credit_history_years: Optional[float] = Field(None, ge=0.0, description="Years of credit history")

    # Optional detailed fields
    initial_list_status: Optional[str] = Field("w", description="Initial listing status: 'w' or 'f'")
    application_type: Optional[str] = Field("Individual", description="Application type: 'Individual' or 'Joint App'")
    disbursement_method: Optional[str] = Field("Cash", description="Disbursement method: 'Cash' or 'DirectPay'")
    collections_12_mths_ex_med: Optional[float] = Field(0.0, ge=0.0)
    acc_now_delinq: Optional[float] = Field(0.0, ge=0.0)
    tot_coll_amt: Optional[float] = Field(0.0, ge=0.0)
    tot_cur_bal: Optional[float] = Field(0.0, ge=0.0)
    total_rev_hi_lim: Optional[float] = Field(20000.0, ge=0.0)
    acc_open_past_24mths: Optional[float] = Field(2.0, ge=0.0)
    mort_acc: Optional[float] = Field(0.0, ge=0.0)
    pub_rec_bankruptcies: Optional[float] = Field(0.0, ge=0.0)
    tax_liens: Optional[float] = Field(0.0, ge=0.0)

class FactorImpact(BaseModel):
    feature: str
    readable_feature: str
    impact: float
    value: Any

class PredictionResponse(BaseModel):
    default_probability: float
    default_percentage: float
    prediction: int
    prediction_label: str
    risk_level: str
    risk_description: str
    threshold_used: float
    features_used_count: int
    top_risk_increasing_factors: List[FactorImpact]
    top_risk_decreasing_factors: List[FactorImpact]

class HealthResponse(BaseModel):
    status: str
    model_type: str
    features_count: int
    threshold: float
    classes: List[float]

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the Loan Default Risk Predictor API",
        "documentation": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    try:
        predictor = get_predictor()
        return {
            "status": "healthy",
            "model_type": type(predictor.model).__name__,
            "features_count": predictor.expected_features_count,
            "threshold": predictor.threshold,
            "classes": predictor.model.classes_.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(applicant: LoanApplicantInput):
    try:
        predictor = get_predictor()
        input_dict = applicant.model_dump()
        result = predictor.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/explain", tags=["Explainability"])
def explain(applicant: LoanApplicantInput, top_k: int = 10):
    try:
        predictor = get_predictor()
        input_dict = applicant.model_dump()
        df_imputed = predictor.preprocess_input(input_dict)
        inc, dec = predictor.explain_prediction(df_imputed, top_k=top_k)
        return {
            "top_risk_increasing_factors": inc,
            "top_risk_decreasing_factors": dec
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explainability error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
