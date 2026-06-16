-- Analytical SQL queries for Fintech Credit Risk & Fraud detection

-- 1. Average Debt-to-Income (DTI) Ratio and Credit Score by Decision Status
SELECT 
    decision_status,
    COUNT(*) as total_applications,
    ROUND(AVG(dti_ratio), 4) as avg_dti,
    ROUND(AVG(risk_score), 4) as avg_model_risk_prob,
    ROUND(AVG(c.credit_score), 1) as avg_credit_score,
    ROUND(SUM(loan_amount), 2) as total_loan_volume
FROM loan_applications l
JOIN customers c ON l.customer_id = c.customer_id
GROUP BY decision_status;

-- 2. Fraud Transaction distribution and volume by Merchant Category
SELECT 
    merchant_category,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud_flag = 1 THEN 1 ELSE 0 END) as fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud_flag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_percentage,
    ROUND(SUM(amount), 2) as total_volume,
    ROUND(AVG(amount), 2) as avg_amount
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_percentage DESC;

-- 3. Loan approvals and volume by Credit Score Brackets
SELECT 
    CASE 
        WHEN c.credit_score >= 800 THEN '1. Exceptional (800-850)'
        WHEN c.credit_score >= 740 THEN '2. Very Good (740-799)'
        WHEN c.credit_score >= 670 THEN '3. Good (670-739)'
        WHEN c.credit_score >= 580 THEN '4. Fair (580-669)'
        ELSE '5. Poor (300-579)'
    END as credit_bracket,
    COUNT(*) as total_applications,
    SUM(CASE WHEN l.decision_status = 'APPROVED' THEN 1 ELSE 0 END) as approved_count,
    ROUND(100.0 * SUM(CASE WHEN l.decision_status = 'APPROVED' THEN 1 ELSE 0 END) / COUNT(*), 2) as approval_rate,
    ROUND(SUM(CASE WHEN l.decision_status = 'APPROVED' THEN l.loan_amount ELSE 0 END), 2) as approved_volume
FROM loan_applications l
JOIN customers c ON l.customer_id = c.customer_id
GROUP BY credit_bracket
ORDER BY credit_bracket ASC;

-- 4. Anomaly score outliers above threshold (anomaly_score > 0.7)
SELECT 
    t.transaction_id,
    c.first_name || ' ' || c.last_name as customer_name,
    t.amount,
    t.merchant_category,
    t.location,
    t.transaction_time,
    t.anomaly_score
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
WHERE t.anomaly_score > 0.7 OR t.is_fraud_flag = 1
ORDER BY t.anomaly_score DESC;

-- 5. Monthly trend analysis of approved loans and average interest rates
-- (Uses strftime for date grouping in SQLite)
SELECT 
    strftime('%Y-%m', applied_at) as application_month,
    COUNT(*) as total_applications,
    SUM(CASE WHEN decision_status = 'APPROVED' THEN 1 ELSE 0 END) as approved_loans,
    ROUND(AVG(interest_rate), 2) as avg_interest_rate,
    ROUND(SUM(CASE WHEN decision_status = 'APPROVED' THEN loan_amount ELSE 0 END), 2) as total_approved_amount
FROM loan_applications
GROUP BY application_month
ORDER BY application_month DESC;
