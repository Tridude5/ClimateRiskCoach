# model/dynamic_bayesian_net.py
from __future__ import annotations
from typing import List, Tuple, Set
from pgmpy.models import DynamicBayesianNetwork as DBN

def build_dbn(
    intraslice_edges: List[Tuple[str, str]],
    temporal_parents: List[Tuple[str, str]],
) -> DBN:
    """
    Build a 2-slice Dynamic Bayesian Network template.

    Parameters
    ----------
    intraslice_edges : list of (u, v)
        Directed edges within the same time slice (for all variables), e.g. ("VIX", "REGIME").
        These edges are added for both t=0 and t=1.
    temporal_parents : list of (u, v)
        Directed edges from time t to time t+1, e.g. ("VIX", "REGIME") meaning (VIX_t -> REGIME_{t+1}).

    Returns
    -------
    DBN
        A pgmpy DynamicBayesianNetwork with nodes for t=0 and t=1 and the specified edges.
    """
    dbn = DBN()

    # Collect all variable names that appear in either edge list
    vars_set: Set[str] = set()
    for u, v in intraslice_edges + temporal_parents:
        vars_set.add(u)
        vars_set.add(v)

    # Add nodes for both timeslices
    for var in sorted(vars_set):
        dbn.add_nodes_from([(var, 0), (var, 1)])

    # Add intraslice edges for t=0 and t=1
    dbn.add_edges_from([((u, 0), (v, 0)) for (u, v) in intraslice_edges])
    dbn.add_edges_from([((u, 1), (v, 1)) for (u, v) in intraslice_edges])

    # Add temporal edges (t -> t+1)
    dbn.add_edges_from([((u, 0), (v, 1)) for (u, v) in temporal_parents])

    return dbn
