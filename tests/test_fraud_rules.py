"""Fraud rules on small hand-made transaction sets."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "etl"))
from fraud_rules import detect, score  # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=["customer_id", "amount", "transaction_time", "location"])


def test_velocity_flags_a_burst_of_five_inside_ten_minutes():
    t0 = pd.Timestamp("2026-01-01 12:00:00")
    burst = [(1, 5.0, t0 + pd.Timedelta(seconds=60 * k), "Phoenix, AZ") for k in range(5)]
    flags = detect(frame(burst))
    assert flags["velocity"].all()


def test_velocity_ignores_four_quick_charges_and_spread_out_history():
    t0 = pd.Timestamp("2026-01-01 12:00:00")
    four = [(1, 5.0, t0 + pd.Timedelta(seconds=60 * k), "Phoenix, AZ") for k in range(4)]
    spread = [(2, 40.0, t0 + pd.Timedelta(minutes=11 * k), "Phoenix, AZ") for k in range(6)]
    flags = detect(frame(four + spread))
    assert not flags["velocity"].any()


def test_velocity_counts_each_customer_separately():
    t0 = pd.Timestamp("2026-01-01 12:00:00")
    mixed = [(k % 2 + 1, 5.0, t0 + pd.Timedelta(seconds=30 * k), "Phoenix, AZ") for k in range(8)]  # 4 each
    assert not detect(frame(mixed))["velocity"].any()


def test_amount_spike_is_relative_to_the_customers_own_median():
    t0 = pd.Timestamp("2026-01-01")
    rows = [(1, 20.0, t0 + pd.Timedelta(days=k), "Phoenix, AZ") for k in range(9)] + [(1, 500.0, t0, "Phoenix, AZ")]
    rows += [(2, 400.0, t0 + pd.Timedelta(days=k), "Phoenix, AZ") for k in range(10)]  # high spender, no spike
    flags = detect(frame(rows))
    assert flags["amount_spike"].tolist() == [False] * 9 + [True] + [False] * 10


def test_foreign_location_rule():
    t0 = pd.Timestamp("2026-01-01")
    flags = detect(frame([(1, 10.0, t0, "International / Online"), (1, 10.0, t0, "Chicago, IL")]))
    assert flags["foreign_location"].tolist() == [True, False]


def test_score_reports_precision_and_recall():
    flags = pd.DataFrame({"velocity": [True, True, False, False], "amount_spike": [False, False, True, False],
                          "foreign_location": [False, False, False, False]})
    truth = pd.Series([True, False, True, True])
    pattern = pd.Series(["card_testing", None, "amount_spike", "foreign_location"])
    m = score(flags, truth, pattern).set_index("rule")
    assert m.loc["velocity", "precision"] == 0.5
    assert m.loc["velocity", "recall_own_pattern"] == 1.0
    assert m.loc["any_rule", "recall_all_fraud"] == round(2 / 3, 4)
