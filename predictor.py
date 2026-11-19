"""
predictor.py — Experimental staffing shortage prediction (synthetic data only).

This module is entirely independent of engine.py and the rule-based eligibility
workflow.  It provides an optional early-warning signal; it does NOT make staffing
decisions, approve unqualified employees, or override the eligibility rules.

Model:  Logistic Regression (scikit-learn)
Data:   data/shift_history.csv — 200 synthetic historical shifts (seed=42)
Split:  Chronological — first 160 rows train, last 40 rows test.
        Chronological ordering preserves temporal integrity (no future leakage).

All public functions return plain Python dicts/lists so the Streamlit layer
can render them without any scikit-learn dependency in app.py itself.
"""
from __future__ import annotations
from pathlib import Path
import csv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).parent / 'data' / 'shift_history.csv'

FEATURES = [
    'shift_duration_hrs',
    'role_encoded',
    'day_of_week',
    'start_hour',
    'available_staff_count',
    'staff_cert_valid',
    'staff_hours_ok',
]
LABEL = 'understaffed_risk'

# Chronological split: train on the first 80%, test on the last 20%
TRAIN_SIZE = 160   # rows 0-159
# TEST_SIZE  = 40  # rows 160-199

RISK_THRESHOLDS = {'low': 0.35, 'high': 0.65}  # probability → label mapping


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_dataset() -> tuple[list[list[float]], list[int]]:
    """Return (X, y) as plain Python lists from the history CSV."""
    X, y = [], []
    with DATA_FILE.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            X.append([float(row[feat]) for feat in FEATURES])
            y.append(int(row[LABEL]))
    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_evaluate() -> dict:
    """
    Train a Logistic Regression on the first 160 rows, evaluate on the last 40.

    Returns a dict with keys:
      model          – fitted sklearn LogisticRegression (for prediction)
      feature_names  – list of feature name strings
      train_size     – number of training rows
      test_size      – number of test rows
      metrics        – dict: precision, recall, f1, accuracy (macro avg)
      confusion_matrix – 2×2 list [[TN, FP], [FN, TP]]
      classes        – [0, 1] labels present in test set
      report_text    – plain-text classification report from sklearn
      both_classes_in_test – bool: True if both 0 and 1 appear in test labels
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        accuracy_score, confusion_matrix, classification_report,
    )

    X, y = _load_dataset()

    X_train, y_train = X[:TRAIN_SIZE], y[:TRAIN_SIZE]
    X_test,  y_test  = X[TRAIN_SIZE:], y[TRAIN_SIZE:]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    classes_in_test = sorted(set(y_test))
    both_classes = (0 in classes_in_test) and (1 in classes_in_test)

    # Use 'binary' averaging (positive class = 1 = at-risk)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    accuracy  = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    report_txt = classification_report(y_test, y_pred, labels=[0, 1],
                                       target_names=['Low risk (0)', 'At risk (1)'],
                                       zero_division=0)

    return {
        'model':                  model,
        'feature_names':          FEATURES,
        'train_size':             len(X_train),
        'test_size':              len(X_test),
        'metrics': {
            'precision': round(precision, 3),
            'recall':    round(recall,    3),
            'f1':        round(f1,        3),
            'accuracy':  round(accuracy,  3),
        },
        'confusion_matrix':       cm,
        'classes':                classes_in_test,
        'report_text':            report_txt,
        'both_classes_in_test':   both_classes,
    }


def predict_shift_risk(
    trained: dict,
    shift_duration_hrs: float,
    role: str,
    day_of_week: int,
    start_hour: int,
    available_staff_count: int,
    staff_cert_valid: int,
    staff_hours_ok: int,
) -> dict:
    """
    Predict shortage risk for a single upcoming shift.

    Parameters mirror the feature set; role is the string ('Lifeguard' /
    'Swim Instructor') and is encoded internally.

    Returns a dict with keys:
      risk_label    – 'Low', 'Medium', or 'High'
      probability   – float 0-1, probability of class 1 (at-risk)
      features_used – dict of the feature values passed to the model
    """
    role_encoded = 0 if role == 'Lifeguard' else 1
    feature_vector = [[
        float(shift_duration_hrs),
        float(role_encoded),
        float(day_of_week),
        float(start_hour),
        float(available_staff_count),
        float(staff_cert_valid),
        float(staff_hours_ok),
    ]]
    model = trained['model']
    prob = float(model.predict_proba(feature_vector)[0][1])  # P(at-risk)

    if prob < RISK_THRESHOLDS['low']:
        label = 'Low'
    elif prob < RISK_THRESHOLDS['high']:
        label = 'Medium'
    else:
        label = 'High'

    return {
        'risk_label':  label,
        'probability': round(prob, 3),
        'features_used': {
            'shift_duration_hrs':   shift_duration_hrs,
            'role_encoded':         role_encoded,
            'day_of_week':          day_of_week,
            'start_hour':           start_hour,
            'available_staff_count': available_staff_count,
            'staff_cert_valid':     staff_cert_valid,
            'staff_hours_ok':       staff_hours_ok,
        },
    }


def features_from_shift_and_staff(shift: dict, staff_list: list[dict]) -> dict:
    """
    Derive pre-shift ML features from the shift dict and staff list.
    Uses only information available before the shift starts.
    Returns a dict of keyword arguments ready to pass to predict_shift_risk().
    """
    from datetime import datetime, timedelta

    day = shift['date']
    fmt = '%Y-%m-%d %H:%M'
    start_dt = datetime.strptime(f"{day} {shift['start']}", fmt)
    end_dt   = datetime.strptime(f"{day} {shift['end']}",   fmt)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)

    shift_duration_hrs = (end_dt - start_dt).total_seconds() / 3600
    role = shift['role']
    day_of_week = start_dt.weekday()   # 0=Monday
    start_hour  = start_dt.hour

    # Count staff passing each stage of the pre-shift eligibility funnel.
    # NOTE: we do NOT use engine.eligible() here — that would pull in the
    # actual eligibility outcome and risk leaking the target.  Instead we
    # compute the same three intermediate counts independently.
    shift_date = start_dt.date()

    available = [s for s in staff_list
                 if s['role'] == role and s['available'].strip().lower() == 'yes']

    cert_valid = [s for s in available
                  if _cert_ok(s['cert_expires'], end_dt.date())]

    hours_ok = [s for s in cert_valid
                if _hours_budget_ok(s['hours_this_week'], s['max_weekly_hours'],
                                    shift_duration_hrs)]

    return {
        'shift_duration_hrs':    shift_duration_hrs,
        'role':                  role,
        'day_of_week':           day_of_week,
        'start_hour':            start_hour,
        'available_staff_count': len(available),
        'staff_cert_valid':      len(cert_valid),
        'staff_hours_ok':        len(hours_ok),
    }


# ---------------------------------------------------------------------------
# Private helpers for features_from_shift_and_staff
# ---------------------------------------------------------------------------

def _cert_ok(cert_expires_str: str, shift_end_date) -> bool:
    from datetime import date
    try:
        return date.fromisoformat(cert_expires_str) >= shift_end_date
    except ValueError:
        return False


def _hours_budget_ok(hours_str: str, max_str: str, shift_hrs: float) -> bool:
    try:
        return float(hours_str) + shift_hrs <= float(max_str)
    except ValueError:
        return False
