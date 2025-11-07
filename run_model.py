# run_model.py

from pathlib import Path
import pandas as pd

# --- 1) Data: ensure processed CSVs exist + build monthly table with 1M climate lag ---
from utils.get_data import ensure_processed_data, build_feature_table

# --- 2) Regime labeling: GMM + financial rules + hysteresis ---
from model.regime import hybrid_regime

# --- 3) Discretization (K-Means / equal-frequency) ---
from model.discretize import discretize_frame

# --- 4) Learn structure + fit EM on a 2-slice Dynamic BN ---
from model.training_BN import (
    learn_intraslice_edges,
    learn_pc_edges,           # optional robustness check
    fit_dbn_em,
)

# --- 5) Walk-forward backtest & metrics ---
from model.backtest import run_rolling
from model.metrics import sharpe, sortino, max_drawdown, classification_metrics

# --- 6) Visualization (DAG) ---
from model.visualize_network import dag_viz   # node size=importance; edge width=MI


def main():
    # ------------------------------------------------------------------
    # A) BUILD DATA
    # ------------------------------------------------------------------
    # Auto-create data_sources/processed/ if missing (converts GISTEMP; placeholders for others)
    ensure_processed_data()

    df = build_feature_table().copy()
    print(f"[INFO] Feature table ready: shape={df.shape}")

    # Required columns (rename here if your names differ)
    # Climate (lagged 1M inside build_feature_table()):
    climate_cols = ["TempAnom", "Carbon", "CPU"]
    # Market:
    market_cols  = ["VIX", "TermSpread", "IGSpreadChg", "SPX_ret", "Bond_ret"]

    # Sanity checks
    missing = [c for c in climate_cols + market_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in build_feature_table(): {missing}")

    # ------------------------------------------------------------------
    # B) LABEL REGIMES (0=Risk-Off, 1=Neutral, 2=Risk-On)
    #    Hybrid: GMM on selected features + financial rules + hysteresis
    # ------------------------------------------------------------------
    gmm_features = ["SPX_ret", "VIX", "IGSpreadChg"]  # adjust if needed
    df["REGIME"] = hybrid_regime(
        df,
        gmm_features,
        vix_col="VIX",
        spread_chg_col="IGSpreadChg",
        hysteresis_months=1,      # confirm switches
    )

    # Drop rows with missing target or returns
    df = df.dropna(subset=["REGIME", "SPX_ret", "Bond_ret"])

    # ------------------------------------------------------------------
    # C) DISCRETIZE FEATURES
    #    Primary: K-Means (k=3). To test equal-frequency, set method="equalfreq".
    # ------------------------------------------------------------------
    nodes = climate_cols + ["VIX", "TermSpread", "IGSpreadChg", "SPX_ret", "REGIME"]
    df_disc = discretize_frame(df[nodes], method="kmeans", k=3)

    # ------------------------------------------------------------------
    # D) STRUCTURE LEARNING (intra-slice) + TEMPORAL EDGES
    # ------------------------------------------------------------------
    # Hill-Climb (BIC); PC also available for robustness
    edges_hc = learn_intraslice_edges(df_disc, nodes)
    # edges_pc = learn_pc_edges(df_disc, nodes)   # optional: compare if you like

    # Forward-only t→t+1 edges into next REGIME (and optionally others)
    temporal_edges = [
        ("TempAnom", "REGIME"), ("Carbon", "REGIME"), ("CPU", "REGIME"),
        ("VIX", "REGIME"), ("TermSpread", "REGIME"), ("IGSpreadChg", "REGIME"),
        ("SPX_ret", "REGIME"),
    ]

    # ------------------------------------------------------------------
    # E) FIT EM & GET INFERENCE
    # ------------------------------------------------------------------
    dbn, infer = fit_dbn_em(
        df_disc=df_disc,
        intraslice_edges=edges_hc,
        temporal_edges=temporal_edges,
        max_iter=150,
    )

    # ------------------------------------------------------------------
    # F) ROLLING BACKTEST (simple version: reuse fitted model)
    #    Evidence at t → P(Risk-Off_{t+1}) → decision at t+1 with costs
    # ------------------------------------------------------------------
    evidence = df_disc.drop(columns=["REGIME"])
    returns  = df[["SPX_ret", "Bond_ret"]].rename(
        columns={"SPX_ret": "equity_ret", "Bond_ret": "bond_ret"}
    )

    # Start after 120-month warm-up (per your spec)
    start_idx = 120 if len(evidence) > 121 else 1
    dates = evidence.index[start_idx:]

    # Provide factory that returns the (already-fitted) inference object.
    # (If you later want *true* rolling re-fits, replace this with a function
    #  that slices the last 120 months up to date t, re-trains, and returns a new infer.)
    bt = run_rolling(lambda _t: infer, evidence, returns, dates)

    # ------------------------------------------------------------------
    # G) METRICS & OUTPUTS
    # ------------------------------------------------------------------
    perf = {
        "Sharpe":  sharpe(bt["net"]),
        "Sortino": sortino(bt["net"]),
        "MaxDD":   max_drawdown(bt["net"]),
    }
    print("\nPortfolio (DBDN Tactical) performance:")
    for k, v in perf.items():
        print(f"  {k:7s}: {v: .4f}")

    # Classification-style view (map decisions to 0/1/2 to compare with labels)
    dec_map = {"risk_off": 0, "risk_neut": 1, "risk_on": 2}
    y_pred = bt["decision"].map(dec_map)
    y_true = df.loc[bt.index, "REGIME"].astype(int)

    clf = classification_metrics(y_true, y_pred, p_off=bt["p_off"])
    print("\nRegime timing metrics:")
    for k, v in clf.items():
        if k != "confusion":
            print(f"  {k:12s}: {v}")
    print("  confusion:\n", clf["confusion"])

    # Save artifacts
    Path("artifacts").mkdir(exist_ok=True)
    bt.to_csv("artifacts/backtest_dynamic_bn.csv")
    pd.Series(edges_hc).to_csv("artifacts/intraslice_edges.csv", index=False)
    pd.Series(temporal_edges).to_csv("artifacts/temporal_edges.csv", index=False)
    df_disc.to_csv("artifacts/training_discretized.csv")

    # ------------------------------------------------------------------
    # H) DAG VISUALIZATION (intra-slice)
    # ------------------------------------------------------------------
    try:
        dag_viz(edges_hc, climate_nodes=climate_cols,
                market_nodes=["VIX","TermSpread","IGSpreadChg","SPX_ret"],
                df_disc=df_disc[nodes], target="REGIME")
    except Exception as e:
        print("DAG visualization skipped:", e)


if __name__ == "__main__":
    main()
