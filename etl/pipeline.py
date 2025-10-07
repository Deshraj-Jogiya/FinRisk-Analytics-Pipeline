# End-to-End ETL Pipeline for Fintech Risk Analytics
import os
import sqlite3
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.db")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "schema.sql")

def initialize_database():
    print(f"Initializing database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
        
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()
    print("Database schema successfully applied.")

def generate_mock_data():
    print("Generating synthetic customers, loan applications, and transactions...")
    np.random.seed(42)
    random.seed(42)
    
    # Names for synthetic customers
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth",
                   "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson",
                  "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White"]
    
    categories = ["Retail", "Travel", "Groceries", "Entertainment", "Utilities", "Gadgets", "Cash_Withdrawal"]
    locations = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ", "Philadelphia, PA"]
    purposes = ["Debt Consolidation", "Home Improvement", "Major Purchase", "Medical Expenses", "Business Investment", "Education"]
    
    num_customers = 200
    customers = []
    
    # 1. Generate Customers
    for i in range(num_customers):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        email = f"{fn.lower()}.{ln.lower()}{random.randint(10, 999)}@fintechmail.local"
        
        # Credit score: center around 680 with some poor and some exceptional
        credit_score = int(np.clip(np.random.normal(675, 80), 300, 850))
        
        # Income: center around $70k with lognormal distribution (skewed right)
        annual_income = round(float(np.random.lognormal(11.1, 0.45)), 2)
        annual_income = max(15000.0, min(annual_income, 350000.0))
        
        # Employment years
        employment_years = int(np.clip(np.random.exponential(5), 0, 40))
        
        # Debt amount
        debt_amount = round(float(np.random.uniform(0.05, 0.4) * annual_income * (random.random() > 0.1)), 2)
        
        customers.append((fn, ln, email, credit_score, annual_income, employment_years, debt_amount))
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.executemany("""
        INSERT INTO customers (first_name, last_name, email, credit_score, annual_income, employment_years, debt_amount)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, customers)
    conn.commit()
    print(f"Inserted {num_customers} customer records.")
    
    # Retrieve customer details to generate loan apps and transactions
    cursor.execute("SELECT customer_id, credit_score, annual_income, debt_amount FROM customers")
    db_customers = cursor.fetchall()
    
    # 2. Generate Loan Applications
    loan_apps = []
    purposes_weights = [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
    
    for cid, cs, income, debt in db_customers:
        # 85% of customers apply for a loan
        if random.random() > 0.85:
            continue
            
        # Loan amount proportional to income
        max_loan = income * 0.4
        loan_amount = round(float(random.uniform(5000, max_loan)), 2)
        loan_amount = round(loan_amount / 100) * 100 # Round to nearest 100
        
        term = random.choice([12, 24, 36, 48, 60])
        purpose = random.choices(purposes, weights=purposes_weights)[0]
        
        # Interest rate based on credit score
        # Base rate 6%, increases for lower credit scores
        base_rate = 6.0
        if cs >= 800:
            rate = base_rate
        elif cs >= 740:
            rate = base_rate + 1.5
        elif cs >= 670:
            rate = base_rate + 3.0
        elif cs >= 580:
            rate = base_rate + 6.5
        else:
            rate = base_rate + 12.0
            
        rate += random.uniform(-0.5, 0.5)
        rate = round(max(4.5, rate), 2)
        
        # Calculate monthly payments
        r = (rate / 100) / 12
        monthly_payment = loan_amount * (r * (1 + r)**term) / ((1 + r)**term - 1)
        
        # Calculate DTI (Debt-to-Income) ratio
        monthly_income = income / 12
        existing_monthly_debt = (debt * 0.03) # assume 3% of total debt is paid monthly
        dti_ratio = round((existing_monthly_debt + monthly_payment) / monthly_income, 4)
        
        # Model risk score simulation
        # Risk score is probability of default, calculated based on credit score, DTI, income, employment years
        # Simulates a logistic regression or XGBoost probability
        logit = 2.5 - (cs - 600)/70.0 + (dti_ratio * 2.0) - (income / 100000.0)
        risk_score = float(1 / (1 + np.exp(-logit)))
        risk_score = round(max(0.001, min(risk_score, 0.999)), 4)
        
        # Decision status rules:
        # approved if risk score < 0.25 and DTI < 0.45 and credit score > 600
        # rejected if risk score > 0.65 or DTI > 0.55 or credit score < 500
        # otherwise pending review
        if risk_score < 0.28 and dti_ratio < 0.45 and cs >= 580:
            status = 'APPROVED'
        elif risk_score > 0.60 or dti_ratio > 0.55 or cs < 520:
            status = 'REJECTED'
        else:
            status = 'PENDING_REVIEW'
            
        # Backdate application date between 2024 and 2026
        days_ago = random.randint(1, 700)
        app_date = datetime.now() - timedelta(days=days_ago)
        app_date_str = app_date.strftime("%Y-%m-%d %H:%M:%S")
        
        loan_apps.append((cid, loan_amount, term, purpose, rate, dti_ratio, risk_score, status, app_date_str))
        
    cursor.executemany("""
        INSERT INTO loan_applications (customer_id, loan_amount, term_months, loan_purpose, interest_rate, dti_ratio, risk_score, decision_status, applied_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, loan_apps)
    conn.commit()
    print(f"Inserted {len(loan_apps)} loan application records.")
    
    # 3. Generate Transactions & Anomaly Detection
    transactions = []
    
    for cid, cs, income, debt in db_customers:
        # Generate 15-40 transactions per customer over the last 60 days
        num_tx = random.randint(15, 40)
        customer_avg_tx = random.uniform(20.0, 150.0)
        
        for _ in range(num_tx):
            category = random.choice(categories)
            location = random.choice(locations)
            
            # Transaction amount: exponential distribution centered around customer average
            amount = round(float(np.random.exponential(customer_avg_tx)), 2)
            amount = max(1.5, amount)
            
            # Anomaly/Fraud simulation logic:
            is_fraud = 0
            anomaly_score = 0.05
            
            # High amount outliers
            if amount > customer_avg_tx * 7.5 and random.random() > 0.4:
                anomaly_score = random.uniform(0.65, 0.95)
                # Some are flagged as fraud
                if anomaly_score > 0.85 and random.random() > 0.5:
                    is_fraud = 1
                    
            # Out of town travel anomalies
            if category == "Travel" and random.random() > 0.90:
                location = "International / Online"
                anomaly_score = max(anomaly_score, random.uniform(0.5, 0.85))
                if random.random() > 0.7:
                    is_fraud = 1
            
            # Add small random noise to anomaly score
            anomaly_score = round(max(0.01, min(anomaly_score + random.uniform(-0.05, 0.05), 0.99)), 4)
            
            tx_days_ago = random.uniform(0.1, 60.0)
            tx_time = datetime.now() - timedelta(days=tx_days_ago)
            tx_time_str = tx_time.strftime("%Y-%m-%d %H:%M:%S")
            
            transactions.append((cid, amount, category, location, tx_time_str, is_fraud, anomaly_score))
            
    cursor.executemany("""
        INSERT INTO transactions (customer_id, amount, merchant_category, location, transaction_time, is_fraud_flag, anomaly_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, transactions)
    conn.commit()
    print(f"Inserted {len(transactions)} transaction records.")
    conn.close()
    print("ETL Mock Data Generation completed successfully!")

if __name__ == '__main__':
    initialize_database()
    generate_mock_data()
