# model/training_BN.py
from __future__ import annotations
from typing import List, Tuple
import pandas as pd

# ---- pgmpy imports (robust across versions) ----
from pgmpy.estimators import HillClimbSearch, PC
# ExpectationMaximization has been stable here:
from pgmpy.estimators import ExpectationMaximization
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference

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

    # Fallbacks in priority order
    if BicScore is not None:
        return BicScore(data)
    if K2Score is not None:
        return K2Score(data)
    if BDeuScore is not None:
        # BDeu needs an equivalent sample size; default is fine for structure search
        return BDeuScore(data)

    raise ImportError(
        "No compatible pgmpy scoring class found. "
        "Tried BicScore/BICScore, K2Score, BDeuScore. "
        "Consider: pip install 'pgmpy==0.1.24'"
    )


def learn_intraslice_edges(df_disc: pd.DataFrame, nodes: List[str]) -> List[Tuple[str, str]]:
    """
    Score-based structure learning (Hill-Climb).
    Uses the best-available scoring method for your pgmpy version.
    """
    data = df_disc[nodes].dropna()
    scoring_method = _pick_scoring(data, prefer="bic")
    hc = HillClimbSearch(data, scoring_method=scoring_method)
    model = hc.estimate()
    return list(model.edges())


def learn_pc_edges(df_disc: pd.DataFrame, nodes: List[str]) -> List[Tuple[str, str]]:
    """PC algorithm (constraint-based) — optional robustness check."""
    data = df_disc[nodes].dropna()
    pc = PC(data)
    dag = pc.estimate(return_type="dag")
    return list(dag.edges())


def make_two_slice_training(df_disc: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll df into (t, t+1) rows for EM on a 2-slice DBN.
    Output columns are MultiIndex: (variable, timeslice).
    """
    rows = []
    idx = df_disc.index
    for t in range(len(idx) - 1):
        r0 = df_disc.iloc[t].rename(lambda c: (c, 0))
        r1 = df_disc.iloc[t + 1].rename(lambda c: (c, 1))
        rows.append(pd.concat([r0, r1]))
    return pd.DataFrame(rows, index=idx[1:])


def fit_dbn_em(
    df_disc: pd.DataFrame,
    intraslice_edges: List[Tuple[str, str]],
    temporal_edges: List[Tuple[str, str]],
    max_iter: int = 150,
) -> tuple[DBN, DBNInference]:
    """
    Build a 2-slice DBN (edges within t + edges from t→t+1 REGIME/others),
    then fit CPDs with EM.
    """
    dbn = build_dbn(intraslice_edges, temporal_edges)  # from dynamic_bayesian_net.py
    train_df = make_two_slice_training(df_disc)
    em = ExpectationMaximization(dbn)
    em.fit(train_df, max_iter=max_iter)
    return dbn, DBNInference(dbn)
