"""Rule-based fraud flags on card transactions, scored against the simulator's planted fraud.

The rules read only the customer, amount, time and location of each transaction. The simulator's labels (`is_fraud_flag` and
`fraud_pattern`) are used afterwards, only to measure how well the rules do.

Rules
- velocity: 5 or more transactions from one customer inside any 10-minute window (card testing).
- amount_spike: an amount more than 7 times the customer's median transaction.
- foreign_location: a transaction outside the customer's home market ("International / Online").
"""
import os
import sqlite3

import pandas as pd

VELOCITY_COUNT = 5
VELOCITY_WINDOW = pd.Timedelta(minutes=10)
SPIKE_MULTIPLE = 7.0
FOREIGN_LOCATION = "International / Online"

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE, "models", "fraud_rule_metrics.csv")


def velocity_flags(tx: pd.DataFrame) -> pd.Series:
    """True for every transaction that sits in a window of VELOCITY_COUNT or more of the same customer's transactions."""
    flags = pd.Series(False, index=tx.index)
    for _, group in tx.sort_values("transaction_time").groupby("customer_id"):
        times = group["transaction_time"].to_numpy()
        idx = group.index.to_numpy()
        start = 0
        for end in range(len(times)):
            while times[end] - times[start] > VELOCITY_WINDOW.to_timedelta64():
                start += 1
            if end - start + 1 >= VELOCITY_COUNT:
                flags.loc[idx[start:end + 1]] = True
    return flags


def amount_spike_flags(tx: pd.DataFrame) -> pd.Series:
    median = tx.groupby("customer_id")["amount"].transform("median")
    return tx["amount"] > SPIKE_MULTIPLE * median


def foreign_location_flags(tx: pd.DataFrame) -> pd.Series:
    return tx["location"] == FOREIGN_LOCATION


RULES = {"velocity": velocity_flags, "amount_spike": amount_spike_flags, "foreign_location": foreign_location_flags}
# The planted fraud pattern each rule is designed to catch
TARGET_PATTERN = {"velocity": "card_testing", "amount_spike": "amount_spike", "foreign_location": "foreign_location"}


def detect(tx: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rule for the given transactions (columns used: customer_id, amount, transaction_time, location)."""
    tx = tx[["customer_id", "amount", "transaction_time", "location"]].copy()
    tx["transaction_time"] = pd.to_datetime(tx["transaction_time"])
    return pd.DataFrame({name: rule(tx) for name, rule in RULES.items()}, index=tx.index)


def score(flags: pd.DataFrame, truth: pd.Series, pattern: pd.Series) -> pd.DataFrame:
    """Precision and recall of each rule and of any rule against the planted fraud, plus recall of the pattern each rule targets."""
    rows = []
    for name in list(RULES) + ["any_rule"]:
        flagged = flags.any(axis=1) if name == "any_rule" else flags[name]
        tp = int((flagged & truth).sum())
        target = pattern == TARGET_PATTERN.get(name)
        rows.append({
            "rule": name,
            "alerts": int(flagged.sum()),
            "true_fraud_caught": tp,
            "precision": round(tp / flagged.sum(), 4) if flagged.sum() else 0.0,
            "recall_all_fraud": round(tp / truth.sum(), 4) if truth.sum() else 0.0,
            "recall_own_pattern": round(int((flagged & target).sum()) / target.sum(), 4) if target.sum() else None,
        })
    return pd.DataFrame(rows)


def run_rules(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    tx = pd.read_sql_query("SELECT transaction_id, customer_id, amount, transaction_time, location, is_fraud_flag, fraud_pattern "
                           "FROM transactions", conn, index_col="transaction_id")
    flags = detect(tx)
    alerts = [(int(tid), rule, None) for rule in RULES for tid in flags.index[flags[rule]]]
    conn.execute("DELETE FROM fraud_alerts")
    conn.executemany("INSERT INTO fraud_alerts (transaction_id, rule, detail) VALUES (?, ?, ?)", alerts)
    conn.commit()
    conn.close()
    metrics = score(flags, tx["is_fraud_flag"] == 1, tx["fraud_pattern"])
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    print(f"Fraud rules raised {len(alerts):,} alerts on {len(tx):,} transactions; scored against the planted fraud:")
    print(metrics.to_string(index=False))
    return metrics


if __name__ == "__main__":
    run_rules(os.path.join(BASE, "app.db"))
