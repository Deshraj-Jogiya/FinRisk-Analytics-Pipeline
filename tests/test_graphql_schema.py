import os
import sqlite3

from api.graphql_schema import schema


def _seed_test_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, "
        "email TEXT, credit_score INTEGER, annual_income REAL, employment_years INTEGER, debt_amount REAL)"
    )
    conn.execute(
        "INSERT INTO customers VALUES (1, 'Jane', 'Doe', 'jane@example.com', 720, 85000.0, 5, 1200.0)"
    )
    conn.execute(
        "CREATE TABLE loan_applications (application_id INTEGER PRIMARY KEY, customer_id INTEGER, "
        "loan_amount REAL, term_months INTEGER, loan_purpose TEXT, interest_rate REAL, dti_ratio REAL, "
        "risk_score REAL, decision_status TEXT, applied_at TEXT)"
    )
    conn.execute(
        "INSERT INTO loan_applications VALUES (1, 1, 10000.0, 36, 'Home Improvement', 5.5, 0.2, 0.15, 'APPROVED', '2026-01-01')"
    )
    conn.commit()
    conn.close()


def test_customers_query_returns_real_row(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    _seed_test_db(str(db_path))
    monkeypatch.setattr("api.graphql_schema.DB_PATH", str(db_path))

    result = schema.execute_sync("{ customers(limit: 5) { customerId firstName creditScore } }")
    assert result.errors is None
    assert result.data["customers"] == [{"customerId": 1, "firstName": "Jane", "creditScore": 720}]


def test_loan_applications_by_status_filters_correctly(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    _seed_test_db(str(db_path))
    monkeypatch.setattr("api.graphql_schema.DB_PATH", str(db_path))

    result = schema.execute_sync(
        '{ loanApplicationsByStatus(decisionStatus: "APPROVED") { applicationId decisionStatus } }'
    )
    assert result.errors is None
    assert result.data["loanApplicationsByStatus"] == [{"applicationId": 1, "decisionStatus": "APPROVED"}]

    result = schema.execute_sync(
        '{ loanApplicationsByStatus(decisionStatus: "REJECTED") { applicationId } }'
    )
    assert result.data["loanApplicationsByStatus"] == []
