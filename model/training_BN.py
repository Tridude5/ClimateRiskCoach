# model/training_BN.py
from __future__ import annotations
from typing import List, Tuple
import pandas as pd

# ---- pgmpy imports (robust across versions) ----
from pgmpy.estimators import HillClimbSearch, PC
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference

# Optional estimators
try:
    from pgmpy.estimators import MaximumLikelihoodEstimator
except Exception:
    MaximumLikelihoodEstimator = None
try:
    from pgmpy.estimators import BayesianEstimator
except Exception:
    BayesianEstimator = None

# Try to import scoring classes with maximum compatibility
BicScore = None
K2Score = None
BDeuScore = None
try:
    from pgmpy.estimators import BicScore
except Exception:
    try:
        from pgmpy.estimators import BICScore as BicScore
    except Exception:
        BicScore = None
try:
    from pgmpy.estimators import K2Score
except Exception:
    K2Score = None
try:
    from pgmpy.estimators import BDeuScore
except Exception:
    BDeuScore = None

from .dynamic_bayesian_net import build_dbn


def _pick_scoring(data: pd.DataFrame, prefer: str = "bic"):
    pref = (prefer or "bic").lower()
    if pref == "bic" and BicScore is not None:
        return BicScore(data)
    if pref in ("k2",) and K2Score is not None:
        return K2Score(data)
    if BicScore is not None:
        return BicScore(data)
    if K2Score is not None:
        return K2Score(data)
    if BDeuScore is not None:
        return BDeuScore(data)
    raise ImportError("No compatible pgmpy scoring class found (BIC/K2/BDeu).")


def learn_intraslice_edges(df_disc: pd.DataFrame, nodes: List[str]) -> List[Tuple[str, str]]:
    cols = [c for c in nodes if c in df_disc.columns]
    data = df_disc.loc[:, cols].dropna()
    scoring_method = _pick_scoring(data, prefer="bic")
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=scoring_method)
    return [(str(u), str(v)) for (u, v) in model.edges()]


def learn_pc_edges(df_disc: pd.DataFrame, nodes: List[str]) -> List[Tuple[str, str]]:
    cols = [c for c in nodes if c in df_disc.columns]
    data = df_disc.loc[:, cols].dropna()
    pc = PC(data)
    dag = pc.estimate(return_type="dag")
    return [(str(u), str(v)) for (u, v) in dag.edges()]


def _base_time_vars(df_in: pd.DataFrame) -> List[str]:
    out = []
    for c in df_in.columns:
        s = str(c).strip()
        if s.endswith("_1"):
            continue
        out.append(s)
    return out


def make_two_slice_training(df_disc: pd.DataFrame) -> pd.DataFrame:
    """
    Build (t, t+1) table with tuple columns. We'll prefer strict slice-0-first order
    to satisfy pgmpy variants that require *all* t=0 columns before any t=1 columns.
    """
    df = df_disc.copy()
    df.columns = [str(c).strip() for c in df.columns]
    vars_ = _base_time_vars(df)

    out_rows = []
    idx = df.index
    for t in range(len(idx) - 1):
        t0 = df.iloc[t]
        t1 = df.iloc[t + 1]
        row = {}
        # fill slice 0 then slice 1 for each var (we'll re-order globally below)
        for v in vars_:
            row[(v, 0)] = t0[v]
        for v in vars_:
            row[(v, 1)] = t1[v]
        out_rows.append(row)

    train_df = pd.DataFrame(out_rows, index=idx[1:])

    # int-cast where possible (discretized data)
    for col in train_df.columns:
        try:
            train_df[col] = train_df[col].astype("Int64").astype(int)
        except Exception:
            pass

    # Global order: ALL slice-0 first, then ALL slice-1 — this is the strictest requirement.
    t0_cols = [(v, 0) for v in vars_]
    t1_cols = [(v, 1) for v in vars_]
    train_df = train_df[t0_cols + t1_cols]
    return train_df


def _node_to_tuple(n) -> tuple:
    var = getattr(n, "variable", None)
    ts = getattr(n, "timeslice", None)
    if var is not None and ts is not None:
        return (str(var), int(ts))
    try:
        a, b = n
        return (str(a), int(b))
    except Exception:
        return (str(n), 0)


def fit_dbn_em(
    df_disc: pd.DataFrame,
    intraslice_edges: List[Tuple[str, str]],
    temporal_edges: List[Tuple[str, str]],
    max_iter: int = 150,
) -> tuple[DBN, DBNInference]:
    """
    Build a 2-slice DBN and fit CPDs with robust ordering:
      A) Try strict 'ALL t=0 then ALL t=1' tuple columns.
      B) If pgmpy still complains, align to dbn.nodes() but re-ordered so all t=0 precede t=1.
      C) As a final fallback, flatten to 'v_0'...'v_1' with t=0 first, then t=1.
    """
    dbn = build_dbn(intraslice_edges, temporal_edges)

    # A) strict tuple columns: all slice-0 first, then slice-1
    train_df_tuple = make_two_slice_training(df_disc)

    def _fit_with(df_for_fit: pd.DataFrame):
        try:
            return dbn.fit(df_for_fit)
        except TypeError:
            if MaximumLikelihoodEstimator is not None:
                return dbn.fit(df_for_fit, estimator=MaximumLikelihoodEstimator)
            if BayesianEstimator is not None:
                return dbn.fit(
                    df_for_fit,
                    estimator=BayesianEstimator,
                    prior_type="BDeu",
                    equivalent_sample_size=5,
                )
            raise

    # Attempt 1: tuple columns t0-first then t1
    try:
        _fit_with(train_df_tuple)
        return dbn, DBNInference(dbn)
    except Exception as e1:
        msg1 = str(e1)

    # B) Align to dbn node order but force t=0 before t=1 globally
    node_order_raw = list(dbn.nodes())
    node_order = [_node_to_tuple(n) for n in node_order_raw]
    t0 = [c for c in node_order if c in train_df_tuple.columns and c[1] == 0]
    t1 = [c for c in node_order if c in train_df_tuple.columns and c[1] == 1]
    aligned = train_df_tuple[t0 + t1]
    try:
        _fit_with(aligned)
        return dbn, DBNInference(dbn)
    except Exception as e2:
        msg2 = str(e2)

    # C) Flatten to string columns "v_0", "v_1" with t=0 first, then t=1
    flat_cols = [f"{v}_{ts}" for (v, ts) in (t0 + t1)]
    train_df_str = aligned.copy()
    train_df_str.columns = flat_cols
    _fit_with(train_df_str)  # raise if still impossible
    return dbn, DBNInference(dbn)
