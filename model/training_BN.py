# model/training_BN.py
from __future__ import annotations
from typing import List, Tuple
import pandas as pd

# ---- pgmpy imports (robust across versions) ----
from pgmpy.estimators import HillClimbSearch, PC
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference

# Optional estimators (some pgmpy versions require explicit estimator)
try:
    from pgmpy.estimators import MaximumLikelihoodEstimator
except Exception:
    MaximumLikelihoodEstimator = None
try:
    from pgmpy.estimators import BayesianEstimator
except Exception:
    BayesianEstimator = None

# Try to import the scoring classes with maximum compatibility
BicScore = None
K2Score = None
BDeuScore = None
try:
    from pgmpy.estimators import BicScore  # many builds
except Exception:
    try:
        from pgmpy.estimators import BICScore as BicScore  # some builds
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
    """
    Return a pgmpy scoring object available in this installation.
    Preference order: BIC -> K2 -> BDeu.
    """
    pref = (prefer or "bic").lower()
    if pref == "bic" and BicScore is not None:
        return BicScore(data)
    if pref in ("k2",) and K2Score is not None:
        return K2Score(data)

    # Fallbacks
    if BicScore is not None:
        return BicScore(data)
    if K2Score is not None:
        return K2Score(data)
    if BDeuScore is not None:
        return BDeuScore(data)

    raise ImportError(
        "No compatible pgmpy scoring class found. "
        "Tried BicScore/BICScore, K2Score, BDeuScore."
    )


def learn_intraslice_edges(df_disc: pd.DataFrame, nodes: List[str]) -> List[Tuple[str, str]]:
    """
    Score-based structure learning (Hill-Climb) among CURRENT-TIME nodes.
    Works even if df_disc has extra columns (e.g., *_1): we subset to `nodes`.
    """
    cols = [c for c in nodes if c in df_disc.columns]
    data = df_disc.loc[:, cols].dropna()
    scoring_method = _pick_scoring(data, prefer="bic")

    # IMPORTANT: pass scoring to estimate(), not to the constructor
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=scoring_method)
    return [(str(u), str(v)) for (u, v) in model.edges()]


def learn_pc_edges(df_disc: pd.DataFrame, nodes: List[str]) -> List[Tuple[str, str]]:
    """PC algorithm (constraint-based) — optional robustness check."""
    cols = [c for c in nodes if c in df_disc.columns]
    data = df_disc.loc[:, cols].dropna()
    pc = PC(data)
    dag = pc.estimate(return_type="dag")
    return [(str(u), str(v)) for (u, v) in dag.edges()]


def _base_time_vars(df_in: pd.DataFrame) -> List[str]:
    """
    Return base-time variable names from df_in by stripping any *_1 columns.
    Keeps 'REGIME' itself (as base-time target) if present.
    """
    out = []
    for c in df_in.columns:
        s = str(c).strip()
        if s.endswith("_1"):
            continue
        out.append(s)
    return out


def make_two_slice_training(df_disc: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (t, t+1) training frame for DBN parameter learning with columns EXACTLY as tuples:
      (var, 0) and (var, 1) for each base-time variable in df_disc.
    - If df_disc contains *_1 columns, they are ignored here.
    - Index is aligned so that row i uses slice t=i-1 for timeslice 0 and t=i for timeslice 1.
    """
    # Normalize column names and keep only base-time vars (ignore *_1)
    df = df_disc.copy()
    df.columns = [str(c).strip() for c in df.columns]
    vars_ = _base_time_vars(df)

    out_rows = []
    idx = df.index

    # Build paired rows (t -> timeslice 0, t+1 -> timeslice 1)
    for t in range(len(idx) - 1):
        row = {}
        t0 = df.iloc[t]
        t1 = df.iloc[t + 1]
        # slice t -> timeslice 0
        for v in vars_:
            row[(v, 0)] = t0[v]
        # slice t+1 -> timeslice 1
        for v in vars_:
            row[(v, 1)] = t1[v]
        out_rows.append(row)

    train_df = pd.DataFrame(out_rows, index=idx[1:])

    # Ensure integer/categorical values if possible (discretized inputs)
    for col in train_df.columns:
        try:
            train_df[col] = train_df[col].astype("Int64").astype(int)
        except Exception:
            # leave as-is if already categorical-like
            pass
    return train_df


def _node_to_tuple(n) -> tuple:
    """
    Convert a pgmpy DynamicNode or a tuple-like node to a canonical (var, timeslice) tuple.
    """
    var = getattr(n, "variable", None)
    ts = getattr(n, "timeslice", None)
    if var is not None and ts is not None:
        return (str(var), int(ts))
    try:
        a, b = n
        return (str(a), int(b))
    except Exception:
        # last resort: stringify
        return (str(n), 0)


def fit_dbn_em(  # keep original name so callers don't change
    df_disc: pd.DataFrame,
    intraslice_edges: List[Tuple[str, str]],
    temporal_edges: List[Tuple[str, str]],
    max_iter: int = 150,
) -> tuple[DBN, DBNInference]:
    """
    Build a 2-slice DBN (edges within t + edges from t→t+1) and fit CPDs.

    Key points:
    - We construct a proper two-slice training frame with tuple columns (var,0)/(var,1)
      from the **base-time** variables in df_disc. This avoids any dependency on *_1 columns.
    - This works whether df_disc is your base-time discretized table or your 2-slice table
      (we automatically ignore *_1 in the latter case).
    """
    # Build the DBN structure
    dbn = build_dbn(intraslice_edges, temporal_edges)

    # Build the two-slice training matrix from base-time vars only
    train_df = make_two_slice_training(df_disc)

    # Reorder columns to match the DBN's nodes exactly and validate presence
    node_order_raw = list(dbn.nodes())  # DynamicNode objects or tuples
    node_order = [_node_to_tuple(n) for n in node_order_raw]

    missing = [c for c in node_order if c not in train_df.columns]
    if missing:
        raise ValueError(f"Training frame missing columns for DBN nodes: {missing}")
    train_df = train_df[node_order]

    # Fit parameters: try plain fit; if version requires explicit estimator, fall back.
    try:
        dbn.fit(train_df)
    except TypeError:
        if MaximumLikelihoodEstimator is not None:
            dbn.fit(train_df, estimator=MaximumLikelihoodEstimator)
        elif BayesianEstimator is not None:
            dbn.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
        else:
            raise

    return dbn, DBNInference(dbn)
