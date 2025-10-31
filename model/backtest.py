# model/backtest.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pgmpy.inference import DBNInference

ALLOC = {
    "risk_off":  {"equity": 0.20, "bond": 0.80},
    "risk_neut": {"equity": 0.75, "bond": 0.25},
    "risk_on":   {"equity": 1.00, "bond": 0.00},
}
THRESH = {"off":0.65, "on":0.35}
TCOST = 0.001  # 10 bps on the shift

def decide(p_off: float) -> str:
    if p_off > THRESH["off"]:
        return "risk_off"
    if p_off < THRESH["on"]:
        return "risk_on"
    return "risk_neut"

def step_return(decision: str, rets: Dict[str,float]) -> float:
    w = ALLOC[decision]
    gross = w["equity"]*rets["equity"] + w["bond"]*rets["bond"]
    return gross

def apply_tcost(prev_w: Dict[str,float], new_w: Dict[str,float]) -> float:
    shift = abs(prev_w["equity"]-new_w["equity"]) + abs(prev_w["bond"]-new_w["bond"])
    return TCOST * shift

def run_rolling(
    infer_factory,      # function(time_index)-> DBNInference configured for that train window
    df_evidence: pd.DataFrame,  # discretized features for inference at t
    df_returns: pd.DataFrame,   # columns: equity_ret, bond_ret
    dates: pd.DatetimeIndex
) -> pd.DataFrame:
    
    history = []
    prev_alloc = {"equity":0.60,"bond":0.40}  # start neutral-ish
    for t in dates:
        infer = infer_factory(t)
        ev = df_evidence.loc[t].dropna().to_dict()
        q = infer.query(variables=[("REGIME",1)], evidence={(k,0): int(v) for k,v in ev.items()})
        # assume state 0=risk_off, 1=neutral, 2=risk_on
        p_off = float(q[("REGIME",1)].values[0])
        decision = decide(p_off)
        # portfolio return applied for t+1 month (already aligned outside)
        r = step_return(decision, {"equity": df_returns.loc[t,"equity_ret"], "bond": df_returns.loc[t,"bond_ret"]})
        # costs
        cost = apply_tcost(prev_alloc, ALLOC[decision])
        net = r - cost
        history.append({"date": t, "p_off": p_off, "decision": decision, "gross": r, "t_cost": -cost, "net": net})
        prev_alloc = ALLOC[decision]
    return pd.DataFrame(history).set_index("date")
