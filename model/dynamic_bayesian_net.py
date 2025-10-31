# model/dynamic_bayesian_net.py
from __future__ import annotations
import itertools
import pandas as pd
from typing import List, Tuple
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.estimators import HillClimbSearch, BicScore, PC
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import DBNInference

def learn_intraslice_structure(df_disc: pd.DataFrame, nodes: List[str]) -> List[Tuple[str,str]]:
  
    data = df_disc[nodes].dropna()
    hc = HillClimbSearch(data, scoring_method=BicScore(data))
    model = hc.estimate()
    return list(model.edges())

def pc_intraslice(df_disc: pd.DataFrame, nodes: List[str]) -> List[Tuple[str,str]]:
    data = df_disc[nodes].dropna()
    pc = PC(data)
    model = pc.estimate(return_type="dag")
    return list(model.edges())

def build_dbn(intraslice_edges: List[Tuple[str,str]],
              temporal_parents: List[Tuple[str,str]]) -> DBN:
   
    dbn = DBN()
    # Add nodes for slice 0
    nodes = sorted(set(list(itertools.chain(*intraslice_edges)) + list(itertools.chain(*temporal_parents))))
    dbn.add_nodes_from([(n, 0) for n in nodes])
    dbn.add_edges_from([((u,0),(v,0)) for (u,v) in intraslice_edges])

    # Add time-slice 1 nodes and temporal edges
    dbn.initialize_initial_state()  # prepare two-slice template
    for n in nodes:
        dbn.add_node((n,1))
    for (u,v) in temporal_parents:
        dbn.add_edge((u,0),(v,1))
    return dbn

def fit_em(dbn: DBN, df_disc: pd.DataFrame, state_names: dict|None=None, max_iter: int=200) -> DBN:
    em = ExpectationMaximization(dbn)
    em.fit(df_disc, max_iter=max_iter, state_names=state_names or {})
    return dbn

def make_infer(dbn: DBN) -> DBNInference:
    return DBNInference(dbn)
