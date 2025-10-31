# model/training_BN.py
from __future__ import annotations
import itertools
import pandas as pd
from typing import List, Tuple
from pgmpy.estimators import HillClimbSearch, BicScore, PC
from pgmpy.estimators import ExpectationMaximization
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
from .dynamic_bayesian_net import build_dbn 

def learn_intraslice_edges(df_disc: pd.DataFrame, nodes: List[str]) -> List[Tuple[str, str]]:
    """Score-based structure learning (Hill-Climb, BIC) on one time slice."""
    data = df_disc[nodes].dropna()
    hc = HillClimbSearch(data, scoring_method=BicScore(data))
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
