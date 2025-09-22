# ml_workloads.py
from __future__ import annotations
from typing import List, Dict
import numpy as np
from sklearn.ensemble import IsolationForest

def anomaly_flags(category_spend_by_month: Dict[str, List[float]], contamination=0.1):
    """
    Input: {"groceries":[... per-month spend ...], "dining":[...], ...}
    Output: per-category anomaly flags for the latest month.
    """
    out = {}
    for cat, series in category_spend_by_month.items():
        arr = np.array(series).reshape(-1, 1)
        if len(arr) < 6:  # need a little history
            out[cat] = {"is_anomaly": False, "score": 0.0}
            continue
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(arr[:-1])  # train on history except the latest
        score = model.decision_function(arr[-1].reshape(-1,1))[0]
        is_anom = model.predict(arr[-1].reshape(-1,1))[0] == -1
        out[cat] = {"is_anomaly": bool(is_anom), "score": float(score)}
    return out

def naive_forecast_monthly(series: List[float], horizon=3):
    """
    Super-light forecaster: last-3-month average for the next horizon months.
    Replace with SARIMA/Prophet if you want heavier deps.
    """
    if len(series) == 0:
        return [0.0] * horizon
    mean3 = np.mean(series[-3:]) if len(series) >= 3 else np.mean(series)
    return [float(mean3)] * horizon
