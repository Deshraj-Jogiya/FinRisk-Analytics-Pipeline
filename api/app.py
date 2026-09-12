# FastAPI Serving & Risk Dashboard Application
import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# App directories
API_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(API_DIR)
DB_PATH = os.path.join(BASE_DIR, "app.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_model.pkl")

app = FastAPI(
    title="FinTech Credit Risk & Fraud Detection API",
    description="End-to-end real-time credit risk forecasting and transaction fraud detection pipeline.",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory=os.path.join(API_DIR, "templates"))
# Ensure static directory exists
os.makedirs(os.path.join(API_DIR, "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(API_DIR, "static")), name="static")

# Pydantic Model for Prediction Request
class LoanApplicationRequest(BaseModel):
    credit_score: int = Field(..., ge=300, le=850, description="Credit Score (300 - 850)")
    annual_income: float = Field(..., ge=0, description="Annual Income in USD")
    employment_years: int = Field(..., ge=0, description="Years in current employment")
    debt_amount: float = Field(..., ge=0, description="Current total outstanding debt in USD")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount in USD")
    term_months: int = Field(..., description="Loan term (12, 24, 36, 48, 60)")
    loan_purpose: str = Field(..., description="Purpose (e.g. Debt Consolidation, Home Improvement, etc.)")
    interest_rate: float = Field(..., ge=0, description="Interest rate (%)")
    dti_ratio: float = Field(None, description="Optional DTI ratio. If not provided, will be calculated.")

# Load model pipeline
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run training script first.")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    if not os.path.exists(DB_PATH):
        return templates.TemplateResponse(request, "error.html", {"message": "Database not initialized. Please run ETL pipeline first."})
        
    conn = get_db_connection()
    
    # 1. KPI Summaries
    kpi_query = """
        SELECT 
            COUNT(*) as total_apps,
            SUM(CASE WHEN decision_status = 'APPROVED' THEN 1 ELSE 0 END) as approved_apps,
            SUM(CASE WHEN decision_status = 'REJECTED' THEN 1 ELSE 0 END) as rejected_apps,
            SUM(CASE WHEN decision_status = 'PENDING_REVIEW' THEN 1 ELSE 0 END) as pending_apps,
            SUM(CASE WHEN decision_status = 'APPROVED' THEN loan_amount ELSE 0 END) as total_approved_val,
            AVG(risk_score) as avg_risk
        FROM loan_applications
    """
    kpis = conn.execute(kpi_query).fetchone()
    
    # Anomaly rate
    anomaly_query = """
        SELECT 
            COUNT(*) as total_tx,
            SUM(is_fraud_flag) as fraud_tx,
            SUM(CASE WHEN anomaly_score > 0.7 THEN 1 ELSE 0 END) as anomaly_tx
        FROM transactions
    """
    tx_kpis = conn.execute(anomaly_query).fetchone()
    
    # 2. Recent Applications List
    recent_apps_query = """
        SELECT 
            l.application_id,
            c.first_name || ' ' || c.last_name as name,
            c.credit_score,
            l.loan_amount,
            l.loan_purpose,
            l.risk_score,
            l.decision_status,
            l.applied_at
        FROM loan_applications l
        JOIN customers c ON l.customer_id = c.customer_id
        ORDER BY l.applied_at DESC
        LIMIT 10
    """
    recent_apps = conn.execute(recent_apps_query).fetchall()
    
    # 3. Recent Anomalous Transactions
    recent_anomalies_query = """
        SELECT 
            t.transaction_id,
            c.first_name || ' ' || c.last_name as name,
            t.amount,
            t.merchant_category,
            t.location,
            t.transaction_time,
            t.is_fraud_flag,
            t.anomaly_score
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        WHERE t.anomaly_score > 0.65 OR t.is_fraud_flag = 1
        ORDER BY t.transaction_time DESC
        LIMIT 10
    """
    recent_anomalies = conn.execute(recent_anomalies_query).fetchall()
    
    # 4. Data for Chart.js
    # Credit brackets chart data
    credit_chart_query = """
        SELECT 
            CASE 
                WHEN c.credit_score >= 800 THEN 'Exceptional (800+)'
                WHEN c.credit_score >= 740 THEN 'Very Good (740-799)'
                WHEN c.credit_score >= 670 THEN 'Good (670-739)'
                WHEN c.credit_score >= 580 THEN 'Fair (580-669)'
                ELSE 'Poor (<580)'
            END as credit_bracket,
            COUNT(*) as total,
            SUM(CASE WHEN l.decision_status = 'APPROVED' THEN 1 ELSE 0 END) as approved
        FROM loan_applications l
        JOIN customers c ON l.customer_id = c.customer_id
        GROUP BY credit_bracket
        ORDER BY c.credit_score DESC
    """
    credit_brackets = conn.execute(credit_chart_query).fetchall()
    
    # Merchant fraud chart data
    merchant_chart_query = """
        SELECT 
            merchant_category,
            COUNT(*) as total,
            SUM(is_fraud_flag) as fraud_count
        FROM transactions
        GROUP BY merchant_category
        ORDER BY fraud_count DESC
    """
    merchant_data = conn.execute(merchant_chart_query).fetchall()
    
    conn.close()
    
    # Format credit bracket data for charts
    credit_labels = [row['credit_bracket'] for row in credit_brackets]
    credit_totals = [row['total'] for row in credit_brackets]
    credit_approved = [row['approved'] for row in credit_brackets]
    
    # Format merchant data
    merchant_labels = [row['merchant_category'] for row in merchant_data]
    merchant_totals = [row['total'] for row in merchant_data]
    merchant_frauds = [row['fraud_count'] for row in merchant_data]
    
    return templates.TemplateResponse(request, "index.html", {
        "kpis": kpis,
        "tx_kpis": tx_kpis,
        "recent_apps": recent_apps,
        "recent_anomalies": recent_anomalies,
        "credit_labels": credit_labels,
        "credit_totals": credit_totals,
        "credit_approved": credit_approved,
        "merchant_labels": merchant_labels,
        "merchant_totals": merchant_totals,
        "merchant_frauds": merchant_frauds
    })

@app.post("/api/predict", response_class=JSONResponse)
async def predict_risk(request: LoanApplicationRequest):
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Calculate DTI if not provided
    dti = request.dti_ratio
    if dti is None:
        monthly_income = request.annual_income / 12
        # Calculate loan installment: standard amortization formula
        r = (request.interest_rate / 100) / 12
        if r > 0:
            installment = request.loan_amount * (r * (1 + r)**request.term_months) / ((1 + r)**request.term_months - 1)
        else:
            installment = request.loan_amount / request.term_months
            
        existing_monthly_debt = request.debt_amount * 0.03 # assume 3% monthly servicing
        dti = (existing_monthly_debt + installment) / monthly_income
        dti = round(dti, 4)
        
    # Create DataFrame for inference
    input_data = pd.DataFrame([{
        'credit_score': request.credit_score,
        'annual_income': request.annual_income,
        'employment_years': request.employment_years,
        'debt_amount': request.debt_amount,
        'loan_amount': request.loan_amount,
        'term_months': request.term_months,
        'loan_purpose': request.loan_purpose,
        'interest_rate': request.interest_rate,
        'dti_ratio': dti
    }])
    
    # Run prediction pipeline
    try:
        prob = model.predict_proba(input_data)[0][1] # Probability of Approval (Target=1 is Approved)
        risk_prob = 1.0 - prob # Risk of Default = 1 - Approval Probability
        
        # Determine status classification
        if risk_prob < 0.28 and dti < 0.45 and request.credit_score >= 580:
            decision = 'APPROVED'
        elif risk_prob > 0.60 or dti > 0.55 or request.credit_score < 520:
            decision = 'REJECTED'
        else:
            decision = 'PENDING_REVIEW'
            
        return {
            "success": True,
            "decision": decision,
            "approval_probability": round(float(prob), 4),
            "default_risk_probability": round(float(risk_prob), 4),
            "calculated_dti": dti
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/api/kpis", response_class=JSONResponse)
async def get_kpis():
    if not os.path.exists(DB_PATH):
        return JSONResponse(status_code=400, content={"error": "Database not initialized."})
    conn = get_db_connection()
    kpis = conn.execute("SELECT COUNT(*) as count, AVG(risk_score) as avg_risk FROM loan_applications").fetchone()
    conn.close()
    return dict(kpis)
