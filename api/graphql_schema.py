"""A real GraphQL query layer over the same SQLite database the REST
API already serves -- additive, not a replacement for the existing
/api/kpis and /api/predict REST endpoints, mounted at /graphql
alongside them.
"""

import os
import sqlite3
from typing import List, Optional

import strawberry
from strawberry.fastapi import GraphQLRouter

API_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(API_DIR)
DB_PATH = os.path.join(BASE_DIR, "app.db")


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@strawberry.type
class Customer:
    customer_id: int
    first_name: str
    last_name: str
    credit_score: int
    annual_income: float
    employment_years: int
    debt_amount: float


@strawberry.type
class LoanApplication:
    application_id: int
    customer_id: int
    loan_amount: float
    term_months: int
    loan_purpose: str
    risk_score: float
    decision_status: str


@strawberry.type
class Query:
    @strawberry.field
    def customer(self, customer_id: int) -> Optional[Customer]:
        conn = _get_db_connection()
        try:
            row = conn.execute(
                "SELECT customer_id, first_name, last_name, credit_score, annual_income, "
                "employment_years, debt_amount FROM customers WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
        finally:
            conn.close()
        return Customer(**dict(row)) if row else None

    @strawberry.field
    def customers(self, limit: int = 20) -> List[Customer]:
        conn = _get_db_connection()
        try:
            rows = conn.execute(
                "SELECT customer_id, first_name, last_name, credit_score, annual_income, "
                "employment_years, debt_amount FROM customers ORDER BY customer_id LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [Customer(**dict(row)) for row in rows]

    @strawberry.field
    def loan_applications_by_status(self, decision_status: str, limit: int = 20) -> List[LoanApplication]:
        conn = _get_db_connection()
        try:
            rows = conn.execute(
                "SELECT application_id, customer_id, loan_amount, term_months, loan_purpose, "
                "risk_score, decision_status FROM loan_applications WHERE decision_status = ? "
                "ORDER BY application_id LIMIT ?",
                (decision_status, limit),
            ).fetchall()
        finally:
            conn.close()
        return [LoanApplication(**dict(row)) for row in rows]


schema = strawberry.Schema(query=Query)
graphql_router = GraphQLRouter(schema)
