import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


logger = logging.getLogger(__name__)

def download_silso_sunspot_daily(url: str = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv") -> pd.DataFrame:
    """
    Load the daily SILSO (Sunspot Index and Long-term Solar Observations) sunspot
    number dataset from the Royal Observatory of Belgium and return it as a 
    time-indexed pandas DataFrame.

    This function downloads the semicolon-separated daily total sunspot number
    dataset (Version 2.0), parses the date components, and constructs a 
    DatetimeIndex. Only essential fields are kept (year, month, day, 
    sunspot_number).

    Parameters
    ----------
    url : str, optional
        The URL of the SILSO daily total sunspot number CSV file. 
        Defaults to the V2.0 daily dataset published by the SIDC.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by UTC calendar date with one column:
            - sunspot_number (float)

        The index is a DatetimeIndex constructed from the year, month, and day
        fields. Invalid or missing sunspot entries may result in NaN values.

    Notes
    -----
    - The file is semicolon-separated and contains comment rows starting with '#'.
    - Columns in the original dataset include statistical uncertainties and 
      metadata, but only the essential fields are retained in this function.
    - The returned series represents the official daily international sunspot 
      number (1-day cadence).
    """
    
    logger.info("Downloading SILSO daily sunspot data from %s", url)

    # Load the semicolon-separated SILSO dataset
    df = pd.read_csv(
        url,
        sep=";",               # SILSO files use semicolon delimiters
        header=None,
        comment="#",           # skip comment/header lines
        names=["year", "month", "day", "decimal_date", "sunspot_number", 
               "std", "n_obs", "definitive"],
        usecols=["year", "month", "day", "sunspot_number"],
    )

    # Build the datetime index
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    # Index by date and keep only the sunspot number
    df = df.set_index("date")[["sunspot_number"]].astype(float)

    logger.info("SILSO sunspot data loaded: %d rows", len(df))

    return df


def download_penticton_f107_daily(start_year: int = 2004, end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch daily F10.7 cm radio flux from the Penticton (Canada) archive and return it
    as a time-indexed pandas DataFrame aligned to daily SILSO sunspot data.

    This pulls the canonical 20:00 UTC ("local noon") observation and uses the
    adjusted flux (scaled to 1 AU - Astronomical Unit), which is generally
    preferred for long-term comparisons across solar cycles.

    Parameters
    ----------
    start_year : int, default 2004
        First calendar year to query. Public Penticton daily pages begin on
        2004-10-28; earlier years will just yield no rows.
    end_year : int or None, default None
        Last calendar year to query. If None, uses the current UTC year.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with a DatetimeIndex named by UTC calendar date (daily cadence)
        and a single column:
            - f107_flux (float): F10.7 cm adjusted flux at 20:00 UT (sfu).

        The index may contain days with NaNs if Penticton did not publish a value.

    Notes
    -----
    - The Penticton daily table contains three nominal observation times per day:
      18:00, 20:00, 22:00 UT. The 20:00 UT record is the canonical daily value.
    - Units are solar flux units (sfu) where 1 sfu = 1e-22 W m^-2 Hz^-1.
    - "Adjusted" values are corrected to 1 AU to remove Sun-Earth distance effects.
    - Duplicate dates (rare) are de-duplicated by keeping the last occurrence.

    Raises
    ------
    ValueError
        If the expected Penticton table cannot be found on a given year page.
    """
    if end_year is None:
        end_year = pd.Timestamp.utcnow().year

    logger.info("Downloading Penticton F10.7 daily data from %d to %d", start_year, end_year,)

    frames: list[pd.DataFrame] = []

    for y in range(start_year, end_year + 1):
        url = f"https://spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-5-flux-en.php?year={y}"
        tables = pd.read_html(url) # read the whole HTML page

        # Go through all tables (if there are multiple ones) and
        # Find the main table which has "Observed Flux" among the column heading
        try:
            tbl = next(t for t in tables if any("Observed Flux" in str(c) for c in t.columns))
        except StopIteration as exc:
            logger.error("Could not locate Penticton flux table for year %d at %s", y, url)
            raise ValueError(f"Could not locate the Penticton flux table for year {y} at {url}") from exc

        # Normalize column names
        tbl.columns = [str(c).strip().lower().replace(" ", "_") for c in tbl.columns]

        # Keep only the canonical 20:00 UT row and the adjusted flux
        df = (
            tbl[tbl["time"].astype(str).str.startswith("20:00")]
            .loc[:, ["date", "adjusted_flux"]]
            .copy()
        )

        # Parse the date text into datetime - coerce: replace any invalid date with NaT
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Make the date the index and rename the adjusted flux column
        df = df.set_index("date").rename(columns={"adjusted_flux": "f107_flux"})

        # Ensure the flux column is numeric.
        df["f107_flux"] = pd.to_numeric(df["f107_flux"], errors="coerce")

        # Accumulate per-year slices
        frames.append(df)

    df_out = pd.concat(frames).sort_index()
    logger.info("Penticton F10.7 data loaded: %d rows from %d to %d", len(df_out), start_year, end_year)

    return df_out

