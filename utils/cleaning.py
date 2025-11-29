import logging
import pandas as pd


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

