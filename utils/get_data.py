# utils/get_data.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data_sources/raw")
PROCESSED = Path("data_sources/processed")

REQUIRED = [
    "gistemp_anomaly_monthly.csv",
    "co2_maunaloa_monthly.csv",
    "cpu_monthly.csv",
    "vix_monthly.csv",
    "term_spread_monthly.csv",
    "ig_oas_monthly.csv",
    "spx_monthly_close.csv",
    "agg_bond_monthly_close.csv",
]

def ensure_processed_data() -> None:
    """
    Make sure all CSVs expected by build_feature_table() exist.
    - Converts NASA GISTEMP raw CSV to monthly anomaly file.
    - Creates simple placeholder series for the rest (so the pipeline runs).
      Replace these later with real data exports.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1) GISTEMP -> gistemp_anomaly_monthly.csv
    out_gistemp = PROCESSED / "gistemp_anomaly_monthly.csv"
    if not out_gistemp.exists():
        src = RAW / "GLB.Ts+dSST.csv"
        if not src.exists():
            raise FileNotFoundError(f"Missing raw file: {src}")

        # find header row
        with open(src) as f:
            for i, line in enumerate(f):
                if line.strip().startswith("Year"):
                    header_row = i
                    break

        df = pd.read_csv(src, skiprows=header_row)
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        dfm = df.melt(id_vars="Year", value_vars=months, var_name="Month", value_name="Anomaly")
        month_map = {m: i for i, m in enumerate(months, start=1)}
        dfm["Month_num"] = dfm["Month"].map(month_map)
        dfm["Date"] = pd.to_datetime(dict(year=dfm["Year"], month=dfm["Month_num"], day=1))
        dfm = dfm[["Date", "Anomaly"]].sort_values("Date").set_index("Date")
        dfm["Anomaly"] = pd.to_numeric(dfm["Anomaly"], errors="coerce")
        dfm.to_csv(out_gistemp)
        print(f"✅ Created {out_gistemp}")

    # 2) Placeholders for other required files (until you add real ones)
    dates = pd.read_csv(out_gistemp, parse_dates=[0], index_col=0).index
    placeholder = pd.DataFrame({"Value": np.linspace(1.0, 2.0, len(dates))}, index=dates)

    for name in REQUIRED:
        fp = PROCESSED / name
        if not fp.exists():
            (placeholder.copy()).to_csv(fp)
            print(f"🗂️  Placeholder created: {fp}")

def _read_series(filename: str, name: str) -> pd.Series:
    s = pd.read_csv(PROCESSED / filename, parse_dates=[0], index_col=0).squeeze("columns")
    s.name = name
    return s

def _to_month_end(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s

def build_feature_table() -> pd.DataFrame:
    """
    Returns monthly DataFrame with:
    ['TempAnom','Carbon','CPU','VIX','TermSpread','IGSpreadChg','SPX_ret','Bond_ret']
    Climate features are lagged by 1 month.
    """
    # Ensure inputs exist
    ensure_processed_data()

    temp = _read_series("gistemp_anomaly_monthly.csv", "TempAnom")
    co2  = _read_series("co2_maunaloa_monthly.csv", "Carbon")
    cpu  = _read_series("cpu_monthly.csv", "CPU")
    vix  = _read_series("vix_monthly.csv", "VIX")
    term = _read_series("term_spread_monthly.csv", "TermSpread")
    ig   = _read_series("ig_oas_monthly.csv", "IGSpread")
    spx  = _read_series("spx_monthly_close.csv", "SPX_px")
    bond = _read_series("agg_bond_monthly_close.csv", "BOND_px")

    series = [temp, co2, cpu, vix, term, ig, spx, bond]
    series = [_to_month_end(s) for s in series]
    df = pd.concat(series, axis=1)

    df["IGSpreadChg"] = df["IGSpread"].diff()
    df["SPX_ret"]     = df["SPX_px"].pct_change()
    df["Bond_ret"]    = df["BOND_px"].pct_change()

    for c in ["TempAnom", "Carbon", "CPU"]:
        df[c] = df[c].shift(1)  # 1-month climate lag

    keep = ["TempAnom","Carbon","CPU","VIX","TermSpread","IGSpreadChg","SPX_ret","Bond_ret"]
    return df[keep].dropna(how="any")
