import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.inference import get_predictor

# Page Configuration
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Preserving Material Icons for Streamlit Sidebar)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* Scoped font-family without overriding Streamlit Material Icons */
    html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Preserve Material Symbols for Streamlit sidebar toggle and controls */
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] i,
    button[kind="header"] span,
    i.material-icons,
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    /* Main App Dark Canvas */
    .stApp {
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
    }

    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        border-left: 5px solid #6366F1;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .header-title-text {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.025em;
        margin: 0;
    }

    .header-desc-text {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* Section Cards */
    .dark-card {
        background-color: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }

    .card-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.01em;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* KPI Summary Tiles */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 22px;
    }

    .kpi-card-dark {
        background-color: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .kpi-card-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-card-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em;
    }

    /* Result Banners */
    .result-low {
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.25) 0%, rgba(16, 185, 129, 0.12) 100%);
        border: 1.5px solid rgba(16, 185, 129, 0.4);
        border-left: 6px solid #10B981;
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 22px;
    }

    .result-medium {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.25) 0%, rgba(245, 158, 11, 0.12) 100%);
        border: 1.5px solid rgba(245, 158, 11, 0.4);
        border-left: 6px solid #F59E0B;
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 22px;
    }

    .result-high {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.25) 0%, rgba(239, 68, 68, 0.12) 100%);
        border: 1.5px solid rgba(239, 68, 68, 0.4);
        border-left: 6px solid #EF4444;
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 22px;
    }

    /* Inputs Dark Styling */
    .stWidgetLabel p, label p {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Submit Button */
    div.stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 14px 28px !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }

    /* Sidebar Dark Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #0F172A !important;
        padding: 4px !important;
        border-radius: 8px !important;
        gap: 6px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 6px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Top Header Card
st.markdown("""
<div class="header-card">
    <div class="header-title-text">Loan Default Risk Predictor</div>
    <div class="header-desc-text">
        Production Machine Learning inference system trained on LendingClub historical loan records.
        Estimates borrower default risk probability and computes local SHAP feature contributions.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Preset Profiles
st.sidebar.markdown("### Benchmark Applicant Profiles")
st.sidebar.markdown("Select an applicant profile to auto-populate the form:")

preset_selection = st.sidebar.radio(
    "Applicant Profile:",
    [
        "Prime Applicant (Low Risk)",
        "Near-Prime Applicant (Medium Risk)",
        "Subprime Applicant (High Risk)",
        "Custom Parameters"
    ],
    index=0
)

# Preset Values
if "Prime" in preset_selection:
    p = {
        "loan_amnt": 12000.0, "term": "36 months", "int_rate": 6.8, "grade": "A", "sub_grade": "A2",
        "annual_inc": 140000.0, "emp_length": "10+ years", "home_ownership": "MORTGAGE",
        "verification_status": "Verified", "purpose": "credit_card", "addr_state": "CA",
        "dti": 8.5, "delinq_2yrs": 0, "inq_last_6mths": 0, "open_acc": 14, "pub_rec": 0,
        "revol_bal": 4200.0, "revol_util": 14.0, "total_acc": 28, "earliest_cr_line": "2006-03-01"
    }
elif "Near-Prime" in preset_selection:
    p = {
        "loan_amnt": 18000.0, "term": "36 months", "int_rate": 13.5, "grade": "C", "sub_grade": "C2",
        "annual_inc": 65000.0, "emp_length": "4 years", "home_ownership": "RENT",
        "verification_status": "Source Verified", "purpose": "debt_consolidation", "addr_state": "TX",
        "dti": 19.5, "delinq_2yrs": 0, "inq_last_6mths": 1, "open_acc": 9, "pub_rec": 0,
        "revol_bal": 11500.0, "revol_util": 54.0, "total_acc": 18, "earliest_cr_line": "2012-07-01"
    }
elif "Subprime" in preset_selection:
    p = {
        "loan_amnt": 35000.0, "term": "60 months", "int_rate": 27.8, "grade": "F", "sub_grade": "F4",
        "annual_inc": 32000.0, "emp_length": "< 1 year", "home_ownership": "RENT",
        "verification_status": "Not Verified", "purpose": "debt_consolidation", "addr_state": "FL",
        "dti": 38.5, "delinq_2yrs": 3, "inq_last_6mths": 4, "open_acc": 4, "pub_rec": 2,
        "revol_bal": 24000.0, "revol_util": 94.0, "total_acc": 7, "earliest_cr_line": "2017-02-01"
    }
else:
    p = {
        "loan_amnt": 15000.0, "term": "36 months", "int_rate": 12.0, "grade": "B", "sub_grade": "B3",
        "annual_inc": 75000.0, "emp_length": "5 years", "home_ownership": "MORTGAGE",
        "verification_status": "Source Verified", "purpose": "debt_consolidation", "addr_state": "NY",
        "dti": 15.0, "delinq_2yrs": 0, "inq_last_6mths": 0, "open_acc": 11, "pub_rec": 0,
        "revol_bal": 8500.0, "revol_util": 38.0, "total_acc": 20, "earliest_cr_line": "2010-05-01"
    }

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Specifications")
st.sidebar.markdown("""
<div style="background-color: #1E293B; border: 1px solid rgba(255,255,255,0.08); padding: 14px; border-radius: 8px; font-size: 0.82rem; line-height: 1.6; color: #94A3B8;">
    <b>Model Type:</b> Logistic Regression<br>
    <b>Decision Threshold:</b> <code>0.530</code><br>
    <b>Test ROC-AUC:</b> <span style="color: #34D399; font-weight: 700;">0.7202</span><br>
    <b>Feature Space:</b> 173 Model Features<br>
    <b>Test Sample:</b> 260,728 records
</div>
""", unsafe_allow_html=True)

# Main Form
with st.form("loan_application_form"):
    st.markdown('<div class="card-heading">Applicant & Loan Information</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Loan Parameters", "Borrower Background", "Credit History"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=1000.0, max_value=40000.0, value=float(p["loan_amnt"]), step=1000.0)
            term = st.selectbox("Term Duration", ["36 months", "60 months"], index=0 if p["term"] == "36 months" else 1)
        with col2:
            int_rate = st.number_input("Interest Rate APR (%)", min_value=5.0, max_value=35.0, value=float(p["int_rate"]), step=0.25)
            grade = st.selectbox("Assigned Grade", ["A", "B", "C", "D", "E", "F", "G"], index=["A", "B", "C", "D", "E", "F", "G"].index(p["grade"]))
        with col3:
            subgrades = [f"{g}{i}" for g in ["A","B","C","D","E","F","G"] for i in range(1, 6)]
            sub_grade = st.selectbox("Sub-Grade", subgrades, index=subgrades.index(p["sub_grade"]) if p["sub_grade"] in subgrades else 6)
            purpose_options = ["debt_consolidation", "credit_card", "home_improvement", "small_business", "major_purchase", "medical", "car", "house", "vacation", "moving", "other"]
            purpose = st.selectbox("Declared Purpose", purpose_options, index=purpose_options.index(p["purpose"]) if p["purpose"] in purpose_options else 0)

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            annual_inc = st.number_input("Annual Gross Income ($)", min_value=0.0, max_value=2000000.0, value=float(p["annual_inc"]), step=5000.0)
            emp_length_options = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years", "Unknown"]
            emp_length = st.selectbox("Employment Length", emp_length_options, index=emp_length_options.index(p["emp_length"]))
        with col2:
            home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"], index=["MORTGAGE", "RENT", "OWN", "OTHER"].index(p["home_ownership"]) if p["home_ownership"] in ["MORTGAGE", "RENT", "OWN", "OTHER"] else 0)
            verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"], index=["Verified", "Source Verified", "Not Verified"].index(p["verification_status"]))
        with col3:
            states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA", "CO", "TN", "MD", "MO", "IN", "WI", "MN", "OR", "SC", "AL", "KY", "LA", "OK", "CT", "UT", "NV", "AR", "MS", "KS", "NE", "NM", "WV", "HI", "NH", "RI", "ID", "ME", "MT", "DE", "ND", "SD", "AK", "VT", "WY", "DC"]
            addr_state = st.selectbox("Borrower State", states, index=states.index(p["addr_state"]) if p["addr_state"] in states else 0)

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            dti = st.number_input("Debt-to-Income (DTI %)", min_value=0.0, max_value=100.0, value=float(p["dti"]), step=0.5)
            inq_last_6mths = st.number_input("Inquiries in Past 6 Months", min_value=0, max_value=25, value=int(p["inq_last_6mths"]))
            delinq_2yrs = st.number_input("Delinquencies in Past 2 Years", min_value=0, max_value=25, value=int(p["delinq_2yrs"]))
        with col2:
            revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=float(p["revol_util"]), step=1.0)
            revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, max_value=1000000.0, value=float(p["revol_bal"]), step=1000.0)
            open_acc = st.number_input("Open Credit Accounts", min_value=1, max_value=80, value=int(p["open_acc"]))
        with col3:
            total_acc = st.number_input("Total Historical Accounts", min_value=1, max_value=120, value=int(p["total_acc"]))
            pub_rec = st.number_input("Public Derogatory Records", min_value=0, max_value=20, value=int(p["pub_rec"]))
            earliest_cr_line = st.text_input("Earliest Credit Line (YYYY-MM)", value=p["earliest_cr_line"])

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("Predict Loan Default Risk")

# Prediction Execution
if submit_btn:
    try:
        predictor = get_predictor()
        input_data = {
            "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "grade": grade,
            "sub_grade": sub_grade, "purpose": purpose, "annual_inc": annual_inc,
            "emp_length": emp_length, "home_ownership": home_ownership,
            "verification_status": verification_status, "addr_state": addr_state,
            "dti": dti, "inq_last_6mths": inq_last_6mths, "delinq_2yrs": delinq_2yrs,
            "revol_util": revol_util, "revol_bal": revol_bal, "open_acc": open_acc,
            "total_acc": total_acc, "pub_rec": pub_rec, "earliest_cr_line": earliest_cr_line
        }

        with st.spinner("Executing model inference and computing SHAP attribution..."):
            result = predictor.predict(input_data)

        prob = result["default_probability"]
        prob_pct = result["default_percentage"]
        risk_level = result["risk_level"]
        prediction = result["prediction"]
        threshold = result["threshold_used"]

        # Financial Calculations
        term_m = 36 if "36" in term else 60
        r_monthly = (int_rate / 100) / 12
        monthly_pmt = (loan_amnt * r_monthly) / (1 - (1 + r_monthly)**(-term_m))
        total_interest = (monthly_pmt * term_m) - loan_amnt

        st.markdown("<br>", unsafe_allow_html=True)

        # Status Outcome Banner
        if risk_level == "Low":
            st.markdown(f"""
            <div class="result-low">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #34D399;">LOW RISK (APPROVED)</div>
                        <div style="color: #CBD5E1; font-size: 0.95rem; margin-top: 4px;">Default probability of <b>{prob_pct}%</b> is comfortably below the 0.53 decision threshold.</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #34D399;">{prob_pct}%</div>
                        <div style="font-size: 0.75rem; font-weight: 700; color: #A7F3D0; text-transform: uppercase;">Default Probability</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif risk_level == "Medium":
            st.markdown(f"""
            <div class="result-medium">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #FBBF24;">MEDIUM RISK (CONDITIONAL REVIEW)</div>
                        <div style="color: #CBD5E1; font-size: 0.95rem; margin-top: 4px;">Moderate default indicators present (<b>{prob_pct}%</b>). Manual underwriting review recommended.</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #FBBF24;">{prob_pct}%</div>
                        <div style="font-size: 0.75rem; font-weight: 700; color: #FDE68A; text-transform: uppercase;">Default Probability</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-high">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #F87171;">HIGH RISK (DEFAULT HAZARD)</div>
                        <div style="color: #CBD5E1; font-size: 0.95rem; margin-top: 4px;">High default probability of <b>{prob_pct}%</b> exceeds the 0.53 decision threshold.</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #F87171;">{prob_pct}%</div>
                        <div style="font-size: 0.75rem; font-weight: 700; color: #FECACA; text-transform: uppercase;">Default Probability</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 4 Summary KPI Tiles
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card-dark">
                <div class="kpi-card-label">Default Probability</div>
                <div class="kpi-card-value" style="color: {'#34D399' if risk_level=='Low' else '#FBBF24' if risk_level=='Medium' else '#F87171'}">{prob_pct}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card-dark">
                <div class="kpi-card-label">Decision Threshold</div>
                <div class="kpi-card-value">{threshold * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card-dark">
                <div class="kpi-card-label">Monthly Installment</div>
                <div class="kpi-card-value">${monthly_pmt:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card-dark">
                <div class="kpi-card-label">Total Finance Charge</div>
                <div class="kpi-card-value">${total_interest:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row
        g_col, s_col = st.columns([1.2, 1])

        with g_col:
            st.markdown('<div class="card-heading">Risk Probability Gauge</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={'suffix': "%", 'font': {'size': 32, 'color': '#FFFFFF', 'family': 'JetBrains Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#64748B", 'ticksuffix': "%"},
                    'bar': {'color': "#6366F1", 'thickness': 0.28},
                    'bgcolor': "#0F172A",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.25)'},
                        {'range': [30, 53], 'color': 'rgba(245, 158, 11, 0.25)'},
                        {'range': [53, 75], 'color': 'rgba(239, 68, 68, 0.25)'},
                        {'range': [75, 100], 'color': 'rgba(220, 38, 38, 0.5)'}
                    ],
                    'threshold': {
                        'line': {'color': "#EF4444", 'width': 4},
                        'thickness': 0.8,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with s_col:
            st.markdown('<div class="card-heading">Underwriting Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dark-card" style="margin-bottom: 0;">
                <div style="font-size: 0.93rem; line-height: 1.7; color: #CBD5E1;">
                    <b>Loan Grade:</b> {grade} (Sub-grade {sub_grade})<br>
                    <b>Loan Structure:</b> ${loan_amnt:,.0f} over {term} @ {int_rate}% APR<br>
                    <b>Annual Income:</b> ${annual_inc:,.0f} ({verification_status})<br>
                    <b>Debt-to-Income (DTI):</b> {dti:.1f}%<br>
                    <b>Revolving Utilization:</b> {revol_util:.1f}% (${revol_bal:,.0f})<br>
                    <hr style="margin: 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);">
                    <b>Outcome Recommendation:</b><br>
                    <span style="font-weight: 700; color: {'#34D399' if risk_level=='Low' else '#FBBF24' if risk_level=='Medium' else '#F87171'};">
                        {'Approve standard loan application' if risk_level=='Low' else 'Conduct manual underwriting review' if risk_level=='Medium' else 'Decline loan application'}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # SHAP Feature Attribution
        st.markdown("---")
        st.markdown('<div class="card-heading">Model Explainability (SHAP Feature Contributions)</div>', unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Top positive and negative feature contributions driving this specific applicant's prediction:</p>", unsafe_allow_html=True)

        exp1, exp2 = st.columns(2)
        inc = result["top_risk_increasing_factors"]
        dec = result["top_risk_decreasing_factors"]

        with exp1:
            st.markdown("##### Factors Increasing Risk")
            if inc:
                df_inc = pd.DataFrame(inc)
                fig_inc = px.bar(
                    df_inc,
                    x="impact",
                    y="readable_feature",
                    orientation="h",
                    color_discrete_sequence=["#EF4444"],
                    labels={"impact": "Risk Contribution (+)", "readable_feature": "Feature"}
                )
                fig_inc.update_layout(
                    height=260,
                    yaxis={'categoryorder': 'total ascending', 'tickfont': {'color': '#E2E8F0'}},
                    xaxis={'tickfont': {'color': '#94A3B8'}, 'gridcolor': 'rgba(255,255,255,0.06)'},
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_inc, use_container_width=True)
            else:
                st.info("No significant risk-increasing factors.")

        with exp2:
            st.markdown("##### Factors Decreasing Risk (Protective)")
            if dec:
                df_dec = pd.DataFrame(dec)
                fig_dec = px.bar(
                    df_dec,
                    x="impact",
                    y="readable_feature",
                    orientation="h",
                    color_discrete_sequence=["#10B981"],
                    labels={"impact": "Risk Reduction (-)", "readable_feature": "Feature"}
                )
                fig_dec.update_layout(
                    height=260,
                    yaxis={'categoryorder': 'total descending', 'tickfont': {'color': '#E2E8F0'}},
                    xaxis={'tickfont': {'color': '#94A3B8'}, 'gridcolor': 'rgba(255,255,255,0.06)'},
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_dec, use_container_width=True)
            else:
                st.info("No significant protective factors.")

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
