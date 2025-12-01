import logging
import pandas as pd
import numpy as np

from typing import Callable, Union




logger = logging.getLogger(__name__)


def count_consecutive_nan_runs(series: pd.Series) -> int:
    """
    Count how many runs of consecutive NaN values appear in a pandas Series.

    A run is defined as a contiguous group of two or more NaN values.
    Each group is counted once, regardless of length.

    Example:
        Input:  [1, NaN, NaN, 3, NaN, NaN, NaN, 5]
        Output: 2

    Parameters
    ----------
    series : pd.Series
        Input pandas Series to analyze.

    Returns
    -------
    int
        Number of consecutive NaN runs in the Series.
    """
    logger.debug("Counting consecutive NaN runs for column: %s", series.name)

    # Convert the Series into a NumPy array of booleans
    is_nan = series.isna().to_numpy()
    run_count: int = 0
    i: int = 0
    n: int = len(is_nan)

    while i < n:
        # If a NaN is found, then start counting the length of the run
        if is_nan[i]:
            start = i
            while i < n and is_nan[i]:
                i += 1
            run_length = i - start
            if run_length >= 2:
                run_count += 1
                logger.debug(
                    "Found NaN run in column '%s' from index %d to %d (length=%d)",
                    series.name,
                    start,
                    i - 1,
                    run_length,
                )
        else:
            i += 1

    logger.debug(
        "Total consecutive NaN runs for column '%s': %d",
        series.name,
        run_count,
    )

    return run_count


def find_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find NaN values in a DataFrame and log the summary

    The function checks the following:
    - Total number of NaN values in the entire DataFrame.
    - Number of multiple (>=2) consecutive NaNs in each column.
      A single run of consecutive NaNs in a column counts as one instance.
    - Number of rows with multiple (>=2) NaN values
      (these rows are potentially corrupted).

    Logging behavior:
    - Always logs a summary at INFO level with:
        * total number of NaNs,
        * total number of multi-NaN runs in columns,
        * total number of rows with multiple NaNs.
    - If the number of rows that contain at least one NaN is <= 15,
      all of those rows are logged at INFO level.
    - If more than 15 rows contain NaNs, only the summary is logged.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the rows that have at least one NaN value.
        (Returned in all cases, regardless of NaN count.)
    """

    logger.info("Starting NaN analysis for dataframe with shape: %s", df.shape)

    # Total number of NaNs in the whole DataFrame
    total_nans: int = int(df.isna().sum().sum())

    # Number of multi-NaN runs (consecutive NaNs) across all columns
    consecutive_nan_instances: int = 0
    for col in df.columns:
        consecutive_nan_instances += count_consecutive_nan_runs(df[col])

    # Number of rows that contain multiple NaNs (>=2)
    row_nan_counts: pd.Series = df.isna().sum(axis=1)
    rows_with_multiple_nans = (row_nan_counts >= 2)
    rows_with_multiple_nans_count: int = int(rows_with_multiple_nans.sum())

    # DataFrame with only rows that contain at least one NaN
    df_with_nans: pd.DataFrame = df[df.isna().any(axis=1)]
    affected_row_count: int = len(df_with_nans)


    # --- Logging ---
    logger.info(
        "NaN analysis summary: "
        "total_nans=%d, "
        "consecutive_nan_instances_in_columns=%d, "
        "rows_with_multiple_nans=%d, "
        "rows_with_any_nan=%d",
        total_nans,
        consecutive_nan_instances,
        rows_with_multiple_nans_count,
        affected_row_count,
    )

    # If there are fewer than 15 rows with NaNs log each
    # Detailed logging only when manageable
    if affected_row_count <= 15:
        logger.info("Logging rows containing NaN values:")
        for idx, row in df_with_nans.iterrows():
            logger.info("Row index=%s | values=%s", idx, row.to_dict())
    else:
        logger.info(
            "More than 15 rows contain NaN values. "
            "Skipping detailed output and returning filtered DataFrame only."
        )

    logger.info("NaN analysis completed.")
    return df_with_nans



def find_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find rows in a time-indexed DataFrame with the same date index.

    Duplicates are the same dates in the index. All occurrences of
    duplicate index values are returned (not just the extra copies).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose index is expected to be a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the rows whose index value appears
        more than once. If there are no duplicate dates, an empty
        DataFrame is returned.

    Raises
    ------
    TypeError
        If the DataFrame index is not a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a pandas.DatetimeIndex.")

    duplicated_mask = df.index.duplicated(keep=False)
    duplicates = df[duplicated_mask].sort_index()

    unique_dup_dates = duplicates.index.unique() if not duplicates.empty else []
    logger.info(
        "Duplicate date detection: %d duplicate rows across %d unique dates.",
        len(duplicates),
        len(unique_dup_dates),
    )

    if not duplicates.empty:
        logger.debug("Duplicate rows detected:\n%s", duplicates)

    return duplicates


# A resolver takes all the duplicate rows for a single date and returns exactly one row
Resolver = Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]

def fix_duplicate_dates(df: pd.DataFrame, resolver: Resolver) -> pd.DataFrame:
    """
    Deduplicate a time-indexed DataFrame

    For each date that appears more than once in the index, all rows for that
    date are passed to the provided 'resolver' function. The resolver 
    returns one row which will represent that date in the output.

    Common resolver strategies include:
    - keeping the first row,
    - keeping the last row,
    - taking the mean of numeric columns,
    - interpolating or applying a custom rule.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by date. Duplicates are defined as repeated index values.
        The index must be a DatetimeIndex.
    resolver : Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]
        Function that receives a sub-DataFrame containing all rows for a single
        date and must return exactly one row (as a DataFrame or Series) representing
        the resolved entry for that date.

    Returns
    -------
    pd.DataFrame
        A DataFrame without duplicate index entries, sorted by index.

    Raises
    ------
    TypeError
        If the DataFrame index is not a DatetimeIndex.
    ValueError
        If the resolver does not return exactly one row.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a pandas.DatetimeIndex.")

    # Detect duplicates first (for logging & early return)
    duplicates = find_duplicate_dates(df)

    if duplicates.empty:
        logger.info("No duplicate dates found. Returning DataFrame sorted by index.")
        return df.sort_index()

    resolver_name = getattr(resolver, "__name__", repr(resolver))
    logger.info(
        "Resolving duplicate dates using resolver '%s'. "
        "Duplicate rows: %d, unique duplicate dates: %d.",
        resolver_name,
        len(duplicates),
        duplicates.index.nunique(),
    )

    # Work on a sorted copy to keep deterministic behavior
    df_sorted = df.sort_index()
    parts: list[pd.DataFrame] = []

    # Group by index (date)
    for key, sub in df_sorted.groupby(level=0, sort=False):
        if len(sub) == 1:
            # Only a single row for this date, keep as-is
            parts.append(sub)
            continue

        logger.debug(
            "Applying resolver '%s' to date %s with %d rows.",
            resolver_name,
            key,
            len(sub),
        )

        resolved = resolver(sub)

        # Normalize Series to single-row DataFrame
        if isinstance(resolved, pd.Series):
            resolved = resolved.to_frame().T

        if not isinstance(resolved, pd.DataFrame):
            raise ValueError(
                "Resolver must return a pandas Series or DataFrame, "
                f"got {type(resolved)}."
            )

        if resolved.empty or len(resolved) != 1:
            raise ValueError(
                "Resolver must return exactly one row for each duplicated date. "
                f"Got {len(resolved)} rows for date {key}."
            )

        # Ensure the index for the resolved row is exactly the date key
        resolved.index = pd.DatetimeIndex([key])
        parts.append(resolved)

    out = pd.concat(parts).sort_index()
    logger.info(
        "Duplicate resolution completed. Original rows: %d, output rows: %d.",
        len(df),
        len(out),
    )
    return out


# ---------- Resolvers ----------
def keep_first(sub: pd.DataFrame) -> pd.DataFrame:
    """Return the first row for this date."""
    return sub.iloc[[0]]


def keep_last(sub: pd.DataFrame) -> pd.DataFrame:
    """Return the last row for this date."""
    return sub.iloc[[-1]]


def take_mean(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Take column-wise mean for numeric columns; for non-numeric keep first.
    Returns a single-row DataFrame.
    """
    num = sub.select_dtypes(include=[np.number]).mean(numeric_only=True)

    nonnum_df = sub.select_dtypes(exclude=[np.number])
    if not nonnum_df.empty:
        nonnum = nonnum_df.iloc[0]
        mixed = pd.concat([nonnum, num])
    else:
        mixed = num

    return mixed.to_frame().T




