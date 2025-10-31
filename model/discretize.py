# model/discretize.py

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer

def kmeans_discretize(series: pd.Series, k: int = 3, random_state: int = 42) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.nunique() <= k:
        # Already discrete-ish; map to 0..k-1
        out = pd.qcut(series.rank(method="first"), q=k, labels=False, duplicates="drop")
        return out.astype("Int64")
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = pd.Series(km.fit_predict(s.to_numpy().reshape(-1,1)), index=s.index)
    # order clusters by centroid so 0=Low,1=Med,2=High
    order = np.argsort(km.cluster_centers_.ravel())
    mapping = {old:new for new, old in enumerate(order)}
    out = labels.map(mapping).reindex(series.index)
    return out.astype("Int64")

def equal_freq_discretize(series: pd.Series, k: int=3) -> pd.Series:
    # ensures ~20–40% per state; handles duplicates
    return pd.qcut(series.rank(method="first"), q=k, labels=False, duplicates="drop").astype("Int64")

def discretize_frame(df: pd.DataFrame, method: str="kmeans", k: int=3, cols: list[str]|None=None) -> pd.DataFrame:
    cols = cols or list(df.columns)
    out = df.copy()
    for c in cols:
        if method == "kmeans":
            out[c] = kmeans_discretize(out[c], k=k)
        elif method == "equalfreq":
            out[c] = equal_freq_discretize(out[c], k=k)
        else:
            raise ValueError("method must be 'kmeans' or 'equalfreq'")
    return out
