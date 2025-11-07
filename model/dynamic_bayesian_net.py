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

    IMPORTANT:
    - Add base variable names as nodes (e.g., "VIX", "REGIME"), NOT ("VIX", 0).
    - Then add edges using ((var, t), (var, t)) and ((u, 0), (v, 1)).
    """
    dbn = DBN()

    # Collect all base variable names
    vars_set: Set[str] = set()
    for u, v in intraslice_edges + temporal_parents:
        vars_set.add(u); vars_set.add(v)

    # Add base variables (pgmpy will manage timeslices)
    dbn.add_nodes_from(sorted(vars_set))

    # Intra-slice edges for t=0 and t=1
    dbn.add_edges_from([((u, 0), (v, 0)) for (u, v) in intraslice_edges])
    dbn.add_edges_from([((u, 1), (v, 1)) for (u, v) in intraslice_edges])

    # Temporal edges t -> t+1
    dbn.add_edges_from([((u, 0), (v, 1)) for (u, v) in temporal_parents])

    # ✅ finalize the two-slice template (required for some pgmpy builds)
    dbn.initialize_initial_state()

    return dbn
