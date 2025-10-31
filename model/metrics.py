# model/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, brier_score_loss, roc_auc_score

def sharpe(returns: pd.Series, rf: float=0.0) -> float:
    ex = returns - rf/12.0
    return np.sqrt(12) * ex.mean() / ex.std(ddof=1)

def sortino(returns: pd.Series, rf: float=0.0) -> float:
    ex = returns - rf/12.0
    downside = ex[ex<0].std(ddof=1)
    return np.sqrt(12) * ex.mean() / (downside if downside>0 else np.nan)

def max_drawdown(returns: pd.Series) -> float:
    cum = (1+returns).cumprod()
    peak = cum.cummax()
    dd = (cum/peak)-1
    return dd.min()

def classification_metrics(y_true, y_pred, p_off=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion": confusion_matrix(y_true, y_pred)
    }
    if p_off is not None and set(np.unique(y_true)).issuperset({0,1,2}):
        # Brier on risk_off as one-vs-rest proxy
        out["brier_off"] = brier_score_loss((y_true==0).astype(int), p_off)
        try:
            out["auc_off"] = roc_auc_score((y_true==0).astype(int), p_off)
        except Exception:
            pass
    return out
