# model/vizualize_network.py
from __future__ import annotations
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def dag_viz(edges, climate_nodes, market_nodes, df_disc, target="REGIME"):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Node importance via mutual information with target
    imp = {}
    cols = [c for c in df_disc.columns if c!=target and df_disc[c].notna().sum()>0]
    X = df_disc[cols].fillna(df_disc[cols].mode().iloc[0]).astype(int)
    y = df_disc[target].astype(int).fillna(df_disc[target].mode().iloc[0])
    mi = mutual_info_classif(X, y, discrete_features=True)
    for c,val in zip(cols, mi):
        imp[c] = val
    # normalize sizes
    svals = np.array([imp.get(n,0.01) for n in G.nodes()])
    svals = 300 + 5000*(svals / (svals.max() if svals.max()>0 else 1))

    # Edge thickness proxy: child MI with parent
    widths=[]
    for u,v in G.edges():
        if u in df_disc and v in df_disc:
            w = mutual_info_classif(df_disc[[u]].fillna(0).astype(int), df_disc[v].astype(int).fillna(0), discrete_features=True)[0]
        else:
            w = 0.05
        widths.append(1 + 6*w)

    colors = []
    for n in G.nodes():
        if n in climate_nodes: colors.append("green")
        elif n in market_nodes: colors.append("blue")
        elif n == "REGIME": colors.append("orange")
        else: colors.append("gray")

    pos = nx.spring_layout(G, seed=7)
    plt.figure(figsize=(10,8))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=svals, width=widths, arrows=True, alpha=0.9)
    plt.title("DBDN DAG: climate (green) / market (blue) / regime (orange)")
    plt.tight_layout(); plt.show()
