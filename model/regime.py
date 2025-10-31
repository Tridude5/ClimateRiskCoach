# model/regime.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# 1) Unsupervised base labels via GMM
def gmm_clusters(df: pd.DataFrame, features: list[str], n_states: int=3, seed: int=42) -> pd.Series:
    X = df[features].replace([np.inf,-np.inf], np.nan).dropna()
    gmm = GaussianMixture(n_components=n_states, covariance_type="full", random_state=seed, n_init=10)
    gmm.fit(X.to_numpy())
    z = pd.Series(gmm.predict(X.to_numpy()), index=X.index, name="gmm_state")
    # Optionally sort by mean return so state ordering is interpretable
    means = X.assign(ret=df['SPX_ret'].reindex(X.index)).groupby(z).mean()['ret'].sort_values()
    mapping = {old:i for i, old in enumerate(means.index)}  # 0=worst -> 2=best
    return z.map(mapping)

# 2) Financial rules refinement
def apply_rules(df: pd.DataFrame, vix_col="VIX", spread_chg_col="IGSpreadChg") -> pd.Series:
    # Risk-Off rule
    risk_off = (df[vix_col] > 25) | (df[spread_chg_col] > 0.40)   # 40 bps = 0.40%
    # Risk-On rule
    risk_on = (df[vix_col] < 15) & (df[spread_chg_col] < -0.20)   # -20 bps
    out = pd.Series(1, index=df.index, name="rule_state")         # 1 = Neutral default
    out[risk_off] = 0
    out[risk_on]  = 2
    return out

# 3) Hybrid + Hysteresis
def hybrid_regime(df: pd.DataFrame,
                  gmm_features: list[str],
                  vix_col="VIX",
                  spread_chg_col="IGSpreadChg",
                  hysteresis_months: int=1) -> pd.Series:
    """
    Returns 0=Risk-Off, 1=Neutral, 2=Risk-On monthly series.
    Hysteresis: a regime switch is confirmed only if the new label persists
    for (hysteresis_months+1) consecutive months; otherwise keep previous.
    Also: if two consecutive months of data missing, drop the span.
    """
    base = gmm_clusters(df, gmm_features).reindex(df.index)
    rules = apply_rules(df, vix_col, spread_chg_col)
    # Combine: rules override where defined; else use gmm
    hybrid = base.copy()
    hybrid.loc[rules.index] = np.where(rules.notna(), rules, base)
    # Handle missing spans ≥ 2 months
    mask_na = df[[vix_col, spread_chg_col]].isna().all(axis=1)
    hybrid[mask_na.rolling(2).sum() == 2] = np.nan  # drop spans with 2 consecutive NA months

    # Hysteresis smoothing
    smoothed = hybrid.copy()
    for t in range(1, len(hybrid)):
        prev = smoothed.iloc[t-1]
        cur  = smoothed.iloc[t]
        if pd.isna(cur):
            continue
        # confirm switch only if it persists h+1 months
        if cur != prev:
            window = hybrid.iloc[t: t + hysteresis_months + 1]
            if (window == cur).sum() < (hysteresis_months + 1):
                smoothed.iloc[t] = prev
    smoothed.name = "REGIME"
    return smoothed.astype("Int64")
