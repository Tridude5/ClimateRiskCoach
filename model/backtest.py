# model/backtest.py
from __future__ import annotations
from typing import Callable, Dict, Any, Iterable
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# ---------- helpers ----------
def _node_for(model, var_name: str, ts: int):
    """Find a model node by (variable name, timeslice)."""
    for n in model.nodes():
        var = getattr(n, "variable", None)
        t   = getattr(n, "timeslice", None)
        if var is None or t is None:
            try:
                var, t = n  # tuple-like ('REGIME', 1)
            except Exception:
                continue
        if str(var) == str(var_name) and int(t) == int(ts):
            return n
    raise KeyError(f"Node not found: ({var_name}, {ts})")

def _prob_off_from_factor(factor, var_key):
    """Extract P(REGIME=0 | evidence) from an inference result."""
    st_names = getattr(factor, "state_names", None)
    if st_names and (var_key in st_names):
        order = st_names[var_key]
        if 0 in order:
            idx = order.index(0)
        elif "0" in order:
            idx = order.index("0")
        else:
            idx = 0
        return float(np.asarray(factor.values).ravel()[idx])
    return float(np.asarray(factor.values).ravel()[0])

def _tuple_evidence(ev_row: pd.Series) -> Dict[tuple, int]:
    """Evidence dict in tuple mode: {('VIX',0): val, ...}. ONLY base vars (no *_1)."""
    out: Dict[tuple, int] = {}
    for var, val in ev_row.items():
        try:
            out[(str(var), 0)] = int(val)
        except Exception:
            pass
    return out

def _node_evidence(ev_row: pd.Series, node0_map: Dict[str, Any]) -> Dict[Any, int]:
    """Evidence dict in node-object mode: {node0_obj_for('VIX'): val, ...}. ONLY base vars."""
    out: Dict[Any, int] = {}
    for var, val in ev_row.items():
        n0 = node0_map.get(str(var))
        if n0 is None:
            continue
        try:
            out[n0] = int(val)
        except Exception:
            pass
    return out

# ---------- strong guardrails ----------
def _ensure_lags_on(evidence: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee for every base column 'X' there exists 'X_1'.
    Create missing lags by shifting the base series. Drop first-row NaNs from the creation.
    Keep deterministic column order: base first, then lags.
    """
    ev = evidence.copy()
    ev.columns = [str(c).strip() for c in ev.columns]
    if "REGIME" in ev.columns:
        ev = ev.drop(columns=["REGIME"])

    cols = [str(c) for c in ev.columns]
    base_vars = [c for c in cols if not c.endswith("_1")]
    created = False
    for v in base_vars:
        lag = f"{v}_1"
        if lag not in ev.columns and v in ev.columns:
            ev[lag] = ev[v].shift(1)
            created = True
    if created:
        ev = ev.dropna()

    cols = [str(c) for c in ev.columns]
    base_vars = [c for c in cols if not c.endswith("_1")]
    lag_vars  = [c for c in cols if c.endswith("_1")]
    ev = ev.loc[:, base_vars + lag_vars]
    return ev

# ---------- main ----------
def run_rolling(
    retrain_fn: Callable[[int], Any],     # -> (infer, model) retrained up to train_end_idx
    evidence: pd.DataFrame,               # discretized features (REGIME absent). May include *_1, but not required.
    returns: pd.DataFrame,                # columns: equity_ret, bond_ret (aligned to evidence index)
    dates: Iterable[pd.Timestamp],        # test dates t — predict REGIME_t from X_{t-1}
    retrain_every: int = 12,              # annual re-train
    recent_only: int | None = None,       # speed-up during dev (e.g., last 360 months)
    progress_every: int = 100,
) -> pd.DataFrame:
    """
    For each test date t:
      - Use evidence at t-1 (ONLY base variable names, no *_1) to query P(REGIME_{t}|X_{t-1})
      - Choose decision from P(risk_off), and use realized returns at t

    This function is robust to missing *_1 columns:
      - It auto-creates lags from base variables if needed.
    """
    try:
        # 0) Make sure lags exist if later code tries to grab them; align returns afterwards.
        evidence = _ensure_lags_on(evidence)
        returns  = returns.copy()
        returns  = returns.loc[evidence.index]

        # 1) Split base vs lag; we will use ONLY base vars for inference
        cols = [str(c) for c in evidence.columns]
        base_vars = [c for c in cols if not c.endswith("_1")]
        # lag_vars  = [c for c in cols if c.endswith("_1")]  # not needed for inference

        # 2) Dates subset (optional)
        all_dates = list(pd.Index(dates))
        if recent_only and len(all_dates) > recent_only:
            all_dates = all_dates[-recent_only:]

        out_rows = []
        mode: str | None = None
        node0_map: Dict[str, Any] = {}
        regime1_key: Any = None
        current_infer = None
        current_model = None
        last_retrain_idx = None

        # Helper: ensure a model retrained up to train_end_idx (which corresponds to t-1)
        def ensure_model(t_i: int, train_end_idx: int):
            nonlocal last_retrain_idx, current_infer, current_model, mode, node0_map, regime1_key
            need_retrain = (
                current_infer is None
                or last_retrain_idx is None
                or (t_i == 1)
                or ((t_i - 1) % retrain_every == 0)
            )
            if need_retrain:
                current_infer, current_model = retrain_fn(train_end_idx)
                last_retrain_idx = train_end_idx
                mode = None
                node0_map = {}
                regime1_key = None

        idx_list = list(evidence.index)

        for t_i, t in enumerate(all_dates, start=1):
            # Find row of t and the previous row for evidence
            pos = evidence.index.get_loc(t)
            if isinstance(pos, slice):
                pos = pos.start
            prev_pos = pos - 1
            if prev_pos < 0:
                continue

            prev_t = idx_list[prev_pos]

            # --- Strictly pass ONLY base (t) variables for inference (timeslice 0)
            ev_row = evidence.loc[prev_t, base_vars]

            # (1) Ensure model is trained up to t-1
            ensure_model(t_i, train_end_idx=prev_pos)

            # (2) On first use after (re)train, detect mode
            if mode is None:
                try:
                    # tuple mode
                    test_ev = _tuple_evidence(ev_row)
                    test_q  = current_infer.query(variables=[("REGIME", 1)], evidence=test_ev)
                    _ = _prob_off_from_factor(test_q, ("REGIME", 1))
                    mode = "tuple"
                    regime1_key = ("REGIME", 1)
                except Exception:
                    # node-object mode
                    node0_map = {str(v): _node_for(current_model, str(v), 0) for v in base_vars}
                    reg1      = _node_for(current_model, "REGIME", 1)
                    test_ev   = _node_evidence(ev_row, node0_map)
                    test_q    = current_infer.query(variables=[reg1], evidence=test_ev)
                    _ = _prob_off_from_factor(test_q, reg1)
                    mode = "node"
                    regime1_key = reg1

            # (3) Query once per step
            if mode == "tuple":
                ev_map = _tuple_evidence(ev_row)
                q = current_infer.query(variables=[regime1_key], evidence=ev_map)
                p_off = _prob_off_from_factor(q, regime1_key)
            else:
                if not node0_map:
                    node0_map = {str(v): _node_for(current_model, str(v), 0) for v in base_vars}
                ev_map = _node_evidence(ev_row, node0_map)
                q = current_infer.query(variables=[regime1_key], evidence=ev_map)
                p_off = _prob_off_from_factor(q, regime1_key)

            # (4) Decision & realized return at t
            if p_off >= 0.5:
                decision, port_ret = "risk_off", returns.loc[t, "bond_ret"]
            elif p_off <= 0.33:
                decision, port_ret = "risk_on", returns.loc[t, "equity_ret"]
            else:
                decision  = "risk_neut"
                port_ret  = 0.5 * (returns.loc[t, "equity_ret"] + returns.loc[t, "bond_ret"])

            out_rows.append({
                "date": t,
                "p_off": p_off,
                "decision": decision,
                "equity_ret": returns.loc[t, "equity_ret"],
                "bond_ret": returns.loc[t, "bond_ret"],
                "net": port_ret,
            })

            if progress_every and (t_i % progress_every == 0):
                print(f"[backtest] {t_i}/{len(all_dates)} steps... (mode={mode})")

        return pd.DataFrame(out_rows).set_index("date")

    except Exception as e:
        # Dump rich context to artifacts before re-raising
        try:
            (ART / "backtest_exception.txt").write_text(
                "ERROR:\n"
                + str(e) + "\n\nTRACEBACK:\n"
                + traceback.format_exc()
            )
            # Try to log columns if available
            try:
                (ART / "backtest_evidence_cols_at_error.txt").write_text(str(list(evidence.columns)))
            except Exception:
                pass
        finally:
            raise
