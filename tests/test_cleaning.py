import numpy as np
import pandas as pd
import pytest
import logging

from utils.cleaning import count_consecutive_nan_runs, find_nans


@pytest.mark.parametrize(
    "values, expected",
    [
        # 0) Empty series → no runs
        ([], 0),

        # 1) No NaNs at all
        ([1, 2, 3, 4], 0),

        # 2) Single NaN only → not a run (needs at least 2)
        ([1, np.nan, 2, 3], 0),

        # 3) One run of exactly 2 NaNs
        ([1, np.nan, np.nan, 4], 1),

        # 4) One run of more than 2 NaNs
        ([1, np.nan, np.nan, np.nan, 5], 1),

        # 5) Two separate runs
        #    [1, NaN, NaN, 3, NaN, NaN, NaN, 5] → 2 runs
        ([1, np.nan, np.nan, 3, np.nan, np.nan, np.nan, 5], 2),

        # 6) Run at the beginning
        ([np.nan, np.nan, 1, 2, 3], 1),

        # 7) Run at the end
        ([1, 2, 3, np.nan, np.nan], 1),

        # 8) All NaNs
        ([np.nan, np.nan, np.nan, np.nan], 1),

        # 9) Mixture of np.nan and None should all be treated as NaN
        ([1, None, np.nan, 2, None, None, 3], 2),

        # 10) Alternating NaNs and values → no run
        ([np.nan, 1, np.nan, 2, np.nan, 3], 0),
    ],
)
def test_count_consecutive_nan_runs_parametrized(values, expected):
    series = pd.Series(values, name="test_series")
    result = count_consecutive_nan_runs(series)
    assert result == expected


def test_series_with_non_numeric_values():
    """NaNs in a string/objects series should still be counted."""
    values = ["a", np.nan, np.nan, "b", "c", np.nan, np.nan]
    series = pd.Series(values, name="string_series")

    result = count_consecutive_nan_runs(series)
    # Two runs: positions 1–2 and 5–6
    assert result == 2


def test_single_long_run_mixed_types():
    """Check a long run with different non-NaN types around it."""
    values = [0, "x", np.nan, np.nan, np.nan, np.nan, 3.14, True]
    series = pd.Series(values, name="mixed_series")

    result = count_consecutive_nan_runs(series)
    assert result == 1



# -------------------- Tests for find_nans --------------------

def test_find_nans_no_nans(caplog):
    """
    DataFrame with no NaNs:
    - total_nans = 0
    - consecutive_nan_instances_in_columns = 0
    - rows_with_multiple_nans = 0
    - rows_with_any_nan = 0
    - returned DataFrame is empty
    """
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )

    with caplog.at_level(logging.INFO, logger="utils.cleaning"):
        result = find_nans(df)

    # Returned df should be empty
    assert result.empty

    # Check the summary log line
    summary = (
        "NaN analysis summary: "
        "total_nans=0, "
        "consecutive_nan_instances_in_columns=0, "
        "rows_with_multiple_nans=0, "
        "rows_with_any_nan=0"
    )
    assert summary in caplog.text

    # Should still log "Logging rows containing NaN values:", but nothing after it
    assert "Logging rows containing NaN values:" in caplog.text
    row_logs = [r for r in caplog.records if "Row index=" in r.message]
    assert len(row_logs) == 0


def test_find_nans_some_nans_under_15_rows(caplog):
    """
    DataFrame with NaNs in every row, but only one NaN per row:
    - total_nans = 4
    - consecutive_nan_instances_in_columns = 1 (column 'a' has one run of length 2)
    - rows_with_multiple_nans = 0 (no row has >= 2 NaNs)
    - rows_with_any_nan = 4
    - all rows returned
    - detailed rows should be logged (<= 15 rows)
    """
    df = pd.DataFrame(
        {
            "a": [1, np.nan, np.nan, 4],
            "b": [np.nan, 2, 3, np.nan],
        }
    )

    with caplog.at_level(logging.INFO, logger="utils.cleaning"):
        result = find_nans(df)

    # Returned df should contain all rows
    assert len(result) == 4
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))

    summary = (
        "NaN analysis summary: "
        "total_nans=4, "
        "consecutive_nan_instances_in_columns=1, "
        "rows_with_multiple_nans=0, "
        "rows_with_any_nan=4"
    )
    assert summary in caplog.text

    # All rows have at least one NaN → all should be logged individually
    row_logs = [r for r in caplog.records if "Row index=" in r.message]
    assert len(row_logs) == 4
    # Make sure it logged the header line too
    assert "Logging rows containing NaN values:" in caplog.text


def test_find_nans_rows_with_multiple_nans_and_consecutive_runs(caplog):
    """
    DataFrame with multiple NaNs per row and multiple consecutive NaN runs:
    df =
        a     b     c
    0  NaN   NaN   1
    1  NaN   2     NaN
    2  3     NaN   NaN
    3  4     NaN   5

    - total_nans = 7
    - column 'a': [NaN, NaN, 3, 4] → one run
      column 'b': [NaN, 2, NaN, NaN] → one run (rows 2–3)
      column 'c': [1, NaN, NaN, 5]   → one run (rows 1–2)
      => consecutive_nan_instances_in_columns = 3
    - row NaN counts: [2, 2, 2, 1] → rows_with_multiple_nans = 3
    - rows_with_any_nan = 4
    """
    df = pd.DataFrame(
        {
            "a": [np.nan, np.nan, 3.0, 4.0],
            "b": [np.nan, 2.0, np.nan, np.nan],
            "c": [1.0, np.nan, np.nan, 5.0],
        }
    )

    with caplog.at_level(logging.INFO, logger="utils.cleaning"):
        result = find_nans(df)

    # All rows contain at least one NaN
    assert len(result) == 4
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))

    summary = (
        "NaN analysis summary: "
        "total_nans=7, "
        "consecutive_nan_instances_in_columns=3, "
        "rows_with_multiple_nans=3, "
        "rows_with_any_nan=4"
    )
    assert summary in caplog.text

    # 4 rows with NaNs -> all 4 logged individually
    row_logs = [r for r in caplog.records if "Row index=" in r.message]
    assert len(row_logs) == 4
    assert "Logging rows containing NaN values:" in caplog.text


def test_find_nans_many_rows_triggers_summary_only(caplog):
    """
    DataFrame with > 15 rows containing NaNs:
    - ensure that the function:
        * logs the summary
        * logs the 'skipping detailed output' message
        * does NOT log each row individually
    """
    n_rows = 20
    df = pd.DataFrame(
        {
            "a": [np.nan] * n_rows,  # one long run of NaNs
            "b": list(range(n_rows)),
        }
    )

    with caplog.at_level(logging.INFO, logger="utils.cleaning"):
        result = find_nans(df)

    # All rows contain NaNs
    assert len(result) == n_rows
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))

    summary = (
        "NaN analysis summary: "
        f"total_nans={n_rows}, "
        "consecutive_nan_instances_in_columns=1, "  # one run in column 'a'
        "rows_with_multiple_nans=0, "
        f"rows_with_any_nan={n_rows}"
    )
    assert summary in caplog.text

    # Check that the "skipping detailed output" message is logged
    assert (
        "More than 15 rows contain NaN values. "
        "Skipping detailed output and returning filtered DataFrame only."
    ) in caplog.text

    # Ensure that individual rows are NOT logged
    row_logs = [r for r in caplog.records if "Row index=" in r.message]
    assert len(row_logs) == 0
    # And the small-rows header should NOT appear either
    assert "Logging rows containing NaN values:" not in caplog.text


def test_find_nans_mixed_none_and_nan(caplog):
    """
    Make sure None is treated as NaN consistently with np.nan.
    Also test that rows with exactly two NaNs are counted as 'multiple'.
    """
    df = pd.DataFrame(
        {
            "a": [1, None, 3, None],
            "b": [None, 2, None, 4],
            "c": [5, 6, None, None],
        }
    )
    # NaN map:
    # row0: a=1,   b=NaN, c=5   → 1 NaN
    # row1: a=NaN, b=2,   c=6   → 1 NaN
    # row2: a=3,   b=NaN, c=NaN → 2 NaNs
    # row3: a=NaN, b=4,   c=NaN → 2 NaNs
    # total_nans = 1 + 1 + 2 + 2 = 6
    # rows_with_multiple_nans = 2 (rows 2 and 3)
    # consecutive runs:
    #   a: [1, NaN, 3, NaN] → no run
    #   b: [NaN, 2, NaN, 4] → no run
    #   c: [5, 6, NaN, NaN] → one run (rows 2–3)
    #   => consecutive_nan_instances_in_columns = 1
    # rows_with_any_nan = 4 (all rows)

    with caplog.at_level(logging.INFO, logger="utils.cleaning"):
        result = find_nans(df)

    assert len(result) == 4
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))

    summary = (
        "NaN analysis summary: "
        "total_nans=6, "
        "consecutive_nan_instances_in_columns=1, "
        "rows_with_multiple_nans=2, "
        "rows_with_any_nan=4"
    )
    assert summary in caplog.text

    # All 4 rows have NaNs and affected_row_count <= 15 → each row logged
    row_logs = [r for r in caplog.records if "Row index=" in r.message]
    assert len(row_logs) == 4
    assert "Logging rows containing NaN values:" in caplog.text
