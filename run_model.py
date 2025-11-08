# run_model.py

from pathlib import Path
import pandas as pd
import traceback

# Logging + artifacts folder
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def log(msg: str): print(f"[OK] {msg}")
def step(msg: str): print(f"\n=== {msg} ===")

# 1) Data: ensure processed CSVs exist + build monthly table with 1M climate lag 
from utils.get_data import ensure_processed_data, build_feature_table

# 2) Regime labeling: GMM + financial rules + hysteresis 
from model.regime import hybrid_regime

# 3) Discretization (K-Means / equal-frequency)
from model.discretize import discretize_frame

# 4) Learn structure + fit EM on a 2-slice Dynamic BN 
from model.training_BN import (
    learn_intraslice_edges,
    learn_pc_edges,        
    fit_dbn_em,
)

# 5) Walk-forward backtest & metrics 
from model.backtest import run_rolling
from model.metrics import sharpe, sortino, max_drawdown, classification_metrics

# 6) Visualization (DAG)
from model.visualize_network import dag_viz   # node size=importance; edge width=MI


def main():
    # ------------------------------------------------------------------
    # A) BUILD DATA
    # ------------------------------------------------------------------
    step("Start run")
    ensure_processed_data()

    step("Build feature table")
    df = build_feature_table().copy()
    # normalize any odd header whitespace
    df.columns = [str(c).strip() for c in df.columns]
    log(f"feature table shape = {df.shape}")
    try:
        df.head(20).to_csv(ART / "feature_table_head.csv")
    except Exception:
        pass

    # Required columns
    climate_cols = ["TempAnom", "Carbon", "CPU"]
    market_cols  = ["VIX", "TermSpread", "IGSpreadChg", "SPX_ret", "Bond_ret"]

    # Sanity checks
    missing = [c for c in climate_cols + market_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in build_feature_table(): {missing}")

    # ------------------------------------------------------------------
    # B) LABEL REGIMES (0=Risk-Off, 1=Neutral, 2=Risk-On)
    # ------------------------------------------------------------------
    step("Regime labeling")
    gmm_features = ["SPX_ret", "VIX", "IGSpreadChg"]
    df["REGIME"] = hybrid_regime(
        df,
        gmm_features,
        vix_col="VIX",
        spread_chg_col="IGSpreadChg",
        hysteresis_months=1,
    )
    df = df.dropna(subset=["REGIME", "SPX_ret", "Bond_ret"])
    log(f"post-labeling rows = {len(df)}")

    # ------------------------------------------------------------------
    # C) DISCRETIZE FEATURES
    # ------------------------------------------------------------------
    step("Discretize")
    nodes = climate_cols + ["VIX", "TermSpread", "IGSpreadChg", "SPX_ret", "REGIME"]
    df_disc = discretize_frame(df[nodes], method="kmeans", k=3)
    # normalize again just in case the discretizer returns non-str names
    df_disc.columns = [str(c).strip() for c in df_disc.columns]
    log(f"discretized shape = {df_disc.shape}")
    try:
        df_disc.head(50).to_csv(ART / "discretized_sample.csv")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # C.5) MAKE 2-SLICE TABLE (ADD _1 LAGS FOR ALL NON-TARGET NODES)
    # ------------------------------------------------------------------
    step("Make 2-slice table (add _1 lags)")

    def add_lag1(df_in: pd.DataFrame, exclude=("REGIME",)) -> pd.DataFrame:
        df_out = df_in.copy()
        for c in df_out.columns:
            if c in exclude:
                continue
            df_out[f"{c}_1"] = df_out[c].shift(1)
        df_out = df_out.dropna()
        df_out.columns = [str(c).strip() for c in df_out.columns]
        return df_out

    df_dbn = add_lag1(df_disc, exclude=("REGIME",))
    log(f"dbn table shape = {df_dbn.shape}")
    try:
        df_dbn.head(50).to_csv(ART / "dbn_sample_with_lags.csv")
    except Exception:
        pass

    # Define exact variables (base t) and lags (for EM training sanity)
    vars_now = ["TempAnom","Carbon","CPU","VIX","TermSpread","IGSpreadChg","SPX_ret"]
    lags_now = [f"{v}_1" for v in vars_now]
    needed_cols = vars_now + lags_now + ["REGIME"]
    missing = [c for c in needed_cols if c not in df_dbn.columns]
    if missing:
        (ART / "missing_in_df_dbn.txt").write_text(str(missing))
        raise KeyError(f"DBN table missing columns: {missing}")

    # ------------------------------------------------------------------
    # D) STRUCTURE LEARNING (intra-slice) + TEMPORAL EDGES
    # ------------------------------------------------------------------
    step("Structure learning")
    edges_hc = learn_intraslice_edges(df_dbn, nodes)   # use 2-slice table for stability
    # edges_pc = learn_pc_edges(df_dbn, nodes)         
    log(f"intraslice edges = {len(edges_hc)}")

    temporal_edges = [
        ("TempAnom", "REGIME"), ("Carbon", "REGIME"), ("CPU", "REGIME"),
        ("VIX", "REGIME"), ("TermSpread", "REGIME"), ("IGSpreadChg", "REGIME"),
        ("SPX_ret", "REGIME"),
    ]

    pd.Series(edges_hc).to_csv(ART / "intraslice_edges.csv", index=False)
    pd.Series(temporal_edges).to_csv(ART / "temporal_edges.csv", index=False)

    # ------------------------------------------------------------------
    # E) ROLLING BACKTEST (120m train, 1m test; re-train every 12m)
    # ------------------------------------------------------------------
    try:
        step("Backtest (rolling 120m train, 1m test; retrain every 12m)")

        # Start from FULL 2-slice evidence (backtest will only use base vars for inference)
        evidence = df_dbn.drop(columns=["REGIME"]).copy()

        # Align returns to this final evidence index now; backtest will re-align after any safe lag creation
        returns  = (
            df.loc[evidence.index, ["SPX_ret", "Bond_ret"]]
              .rename(columns={"SPX_ret": "equity_ret", "Bond_ret": "bond_ret"})
        )

        # Helpful breadcrumbs
        try:
            (ART / "evidence_cols_final.txt").write_text(str(list(evidence.columns)))
        except Exception:
            pass
        try:
            from model import backtest as _mb
            print("[DEBUG] Using backtest from:", getattr(_mb.run_rolling, "__code__", None).co_filename)
        except Exception:
            pass

        # Start after 120-month warm-up so the first train window is full
        start_idx = 120 if len(evidence) > 121 else 1
        dates = evidence.index[start_idx:]

        def retrain_fn(train_end_idx: int):
            # window [train_end_idx-119, train_end_idx]
            win_start = max(0, train_end_idx - 119)
            train_slice = df_dbn.iloc[win_start:train_end_idx+1]  # includes *_1 columns

            # HARD ASSERT: the training window must include all lags + target
            need = set(vars_now + [f"{v}_1" for v in vars_now] + ["REGIME"])
            has  = set(train_slice.columns)
            miss = sorted(list(need - has))
            if miss:
                (ART / "missing_in_train_slice.txt").write_text(
                    "missing=" + str(miss) + "\ncols=" + str(list(train_slice.columns))
                )
                raise KeyError(f"Training window missing columns: {miss}")

            # learn intra-slice structure on window 
            edges_hc_local = learn_intraslice_edges(train_slice, nodes)

            # Fit DBN parameters on the same 2-slice window
            dbn_local, infer_local = fit_dbn_em(
                train_slice,                     # positional arg: 2-slice data with lags
                intraslice_edges=edges_hc_local,
                temporal_edges=temporal_edges,
                max_iter=150,
            )
            return infer_local, infer_local.model

        bt = run_rolling(
            retrain_fn=retrain_fn,
            evidence=evidence,    # full (includes *_1); backtest ignores lags for inference
            returns=returns,
            dates=dates,
            retrain_every=12,
            recent_only=None,
            progress_every=50,
        )

        bt.to_csv(ART / "backtest_dynamic_bn.csv")
        log(f"backtest rows = {len(bt)}")

        # ------------------------------------------------------------------
        # F) METRICS & OUTPUTS (robust to different run_rolling returns)
        # ------------------------------------------------------------------
        # Normalize bt: accept either DataFrame or tuple
        if isinstance(bt, tuple):
            # Expect (df_bt,) or (y_true, y_pred, p_off, returns_df) — adapt gracefully
            if len(bt) == 1 and isinstance(bt[0], pd.DataFrame):
                bt = bt[0]
            else:
                # Build a DataFrame shell if possible
                y_true_bt = None
                y_pred_bt = None
                p_off_bt  = None
                ret_bt    = None
                for obj in bt:
                    if isinstance(obj, pd.Series):
                        nm = getattr(obj, "name", "") or ""
                        if nm.lower() in ("y_true","true","regime_true"):
                            y_true_bt = obj
                        elif nm.lower() in ("y_pred","pred","decision"):
                            y_pred_bt = obj
                        elif nm.lower() in ("p_off","p_risk_off","prob_off"):
                            p_off_bt = obj
                        elif nm.lower() in ("net","returns","net_ret","portfolio_ret"):
                            ret_bt = obj
                    elif isinstance(obj, pd.DataFrame):
                        if "net" in obj.columns:
                            ret_bt = obj["net"]
                bt = pd.DataFrame()
                if ret_bt is not None:
                    bt["net"] = ret_bt
                if y_pred_bt is not None:
                    bt["decision"] = y_pred_bt
                if p_off_bt is not None:
                    bt["p_off"] = p_off_bt

        # ---------- Portfolio metrics ----------
        net_candidates = ["net", "portfolio_ret", "ret", "net_ret", "strategy_ret"]
        net_col = next((c for c in net_candidates if c in bt.columns), None)

        if net_col is not None:
            try:
                perf = {
                    "Sharpe":  sharpe(bt[net_col]),
                    "Sortino": sortino(bt[net_col]),
                    "MaxDD":   max_drawdown(bt[net_col]),
                }
                print("\nPortfolio (DBDN Tactical) performance:")
                for k, v in perf.items():
                    print(f"  {k:7s}: {v: .4f}")
            except Exception as _e:
                print("[WARN] Portfolio metric calc failed:", _e)
        else:
            print("[WARN] No net return column found in backtest output "
                  f"(looked for {net_candidates}). Skipping portfolio metrics.")

        # ---------- Classification metrics ----------
        # y_true from labeled df, aligned to bt index
        try:
            y_true = df.loc[bt.index, "REGIME"].astype(int)
        except Exception as _e:
            y_true = None
            print("[WARN] Could not align y_true (REGIME) to backtest index:", _e)

        # y_pred from bt["decision"] (numeric or string)
        y_pred = None
        if "decision" in bt.columns:
            ser = bt["decision"]
            if pd.api.types.is_numeric_dtype(ser):
                y_pred = ser.astype(int)
            else:
                dec_map = {"risk_off": 0, "risk_neut": 1, "risk_on": 2,
                           "off": 0, "neutral": 1, "on": 2}
                y_pred = ser.astype(str).str.lower().map(dec_map)
        elif "regime_pred" in bt.columns:
            y_pred = bt["regime_pred"].astype(int)

        # p_off best-effort
        p_off = None
        for cand in ["p_off", "p_risk_off", "prob_off"]:
            if cand in bt.columns:
                p_off = bt[cand].astype(float)
                break

        if (y_true is not None) and (y_pred is not None):
            try:
                clf = classification_metrics(y_true, y_pred, p_off=p_off)
                print("\nRegime timing metrics:")
                for k, v in clf.items():
                    if k != "confusion":
                        try:
                            print(f"  {k:12s}: {float(v):.4f}")
                        except Exception:
                            print(f"  {k:12s}: {v}")
                print("  confusion:\n", clf["confusion"])
            except Exception as _e:
                print("[WARN] Classification metrics failed:", _e)
        else:
            if y_true is None:
                print("[WARN] Missing y_true (REGIME) — cannot compute classification metrics.")
            if y_pred is None:
                print("[WARN] Missing y_pred/decision in backtest output — cannot compute classification metrics.")

    except Exception as e:
        (ART / "error.txt").write_text(str(e))
        (ART / "traceback.txt").write_text(traceback.format_exc())
        print("\n[ERROR] Stopped during backtest:", e)

    # ------------------------------------------------------------------
    # G) DAG VISUALIZATION (intra-slice)
    # ------------------------------------------------------------------
    step("DAG visualization")
    try:
        dag_viz(edges_hc, climate_nodes=climate_cols,
                market_nodes=["VIX","TermSpread","IGSpreadChg","SPX_ret"],
                df_disc=df_dbn[nodes], target="REGIME")  # consistent index/table
    except Exception as e:
        print("DAG visualization skipped:", e)

    step("Done")


if __name__ == "__main__":
    main()
