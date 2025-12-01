import logging
import pytest
from datetime import datetime

import pandas as pd
import numpy as np

import utils.cleaning as cleaning

from utils.cleaning import (
    find_duplicate_dates,
    keep_first,
    keep_last,
    take_mean,

)



def _dt_list(strs):
    """Helper to convert list of date strings to a DatetimeIndex."""
    return pd.to_datetime(strs)


# ---------- Basic validation / type checks ----------
def test_find_duplicate_dates_raises_if_index_not_datetime():
    df = pd.DataFrame(
        {"value": [1, 2, 3]},
        index=[0, 1, 2],  # RangeIndex, not DatetimeIndex
    )

    with pytest.raises(TypeError, match="Index must be a pandas.DatetimeIndex"):
        find_duplicate_dates(df)


# ---------- No duplicates ----------
def test_find_duplicate_dates_no_duplicates_returns_empty_and_logs_summary(caplog):
    idx = _dt_list(["2024-01-01", "2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"value": [10, 20, 30]}, index=idx)

    with caplog.at_level(logging.INFO, logger=cleaning.__name__):
        dup = find_duplicate_dates(df)

    # No duplicates -> empty DataFrame
    assert dup.empty
    assert isinstance(dup.index, pd.DatetimeIndex)

    # Summary log should report zeros
    expected_summary = (
        "Duplicate date detection: 0 duplicate rows across 0 unique dates."
    )
    assert expected_summary in caplog.text

    # No debug line about "Duplicate rows detected"
    assert "Duplicate rows detected:" not in caplog.text


# ---------- Simple duplicates ----------
def test_find_duplicate_dates_single_duplicate_date_two_rows():
    """
    One date appears twice, others are unique.
    Expect both rows for that date to be returned, sorted by index.
    """
    idx = _dt_list(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"value": [10, 20, 21, 30]}, index=idx)

    dup = find_duplicate_dates(df)

    expected_idx = _dt_list(["2024-01-02", "2024-01-02"])
    expected = pd.DataFrame({"value": [20, 21]}, index=expected_idx).sort_index()

    pd.testing.assert_frame_equal(dup, expected)


def test_find_duplicate_dates_non_contiguous_and_unsorted_index():
    """
    Duplicate dates that are not next to each other and index is unsorted.
    Function should still:
      - detect all duplicates using index.duplicated(keep=False)
      - return them sorted by index
    """
    idx = _dt_list(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01"])
    df = pd.DataFrame({"value": [30, 10, 20, 11]}, index=idx)

    dup = find_duplicate_dates(df)

    # What we expect: rows where index value is duplicated, then sorted by index
    expected = df[df.index.duplicated(keep=False)].sort_index()
    pd.testing.assert_frame_equal(dup, expected)


# ---------- Multiple duplicate dates ----------
def test_find_duplicate_dates_multiple_dates_and_log_counts(caplog):
    """
    Two different dates are duplicated:
      2024-01-01 appears twice
      2024-01-02 appears three times
      2024-01-03 is unique

    Expect:
      - total duplicate rows = 5
      - unique duplicated dates = 2
      - log message reflects this
    """
    idx = _dt_list(
        ["2024-01-01", "2024-01-01",
         "2024-01-02", "2024-01-02", "2024-01-02",
         "2024-01-03"]
    )
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6]}, index=idx)

    with caplog.at_level(logging.INFO, logger=cleaning.__name__):
        dup = find_duplicate_dates(df)

    assert len(dup) == 5
    assert dup.index.nunique() == 2

    expected_summary = (
        "Duplicate date detection: 5 duplicate rows across 2 unique dates."
    )
    assert expected_summary in caplog.text


def test_find_duplicate_dates_logs_debug_rows_when_duplicates_present(caplog):
    """
    When there are duplicates, function should log a DEBUG message
    with the duplicate rows printed.
    """
    idx = _dt_list(["2024-01-01", "2024-01-01"])
    df = pd.DataFrame({"value": [10, 20]}, index=idx)

    with caplog.at_level(logging.DEBUG, logger=cleaning.__name__):
        dup = find_duplicate_dates(df)

    # Ensure duplicates are correctly returned
    assert len(dup) == 2

    # Debug output about the duplicate rows should be present
    assert "Duplicate rows detected:" in caplog.text
    # And the actual frame content should be in the log text too
    assert "2024-01-01" in caplog.text
    assert "10" in caplog.text
    assert "20" in caplog.text


# ---------- Dtype / column preservation ----------
def test_find_duplicate_dates_preserves_all_columns_and_dtypes():
    """
    Make sure the returned DataFrame:
      - keeps all original columns
      - keeps their dtypes
      - doesn't modify values
    """
    idx = _dt_list(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        },
        index=idx,
    )

    dup = find_duplicate_dates(df)

    expected = df[df.index.duplicated(keep=False)].sort_index()
    pd.testing.assert_frame_equal(dup, expected)


# ---------- Test the resolver functions ----------
def test_keep_first_returns_first_row():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
    )

    result = keep_first(df)

    # Must be a DataFrame with exactly one row
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    # Should match the first row of the input
    expected = df.iloc[[0]]
    pd.testing.assert_frame_equal(result, expected)


def test_keep_last_returns_last_row():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        },
        index=pd.to_datetime(["2024-01-01"] * 3),
    )

    result = keep_last(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    expected = df.iloc[[-1]]
    pd.testing.assert_frame_equal(result, expected)


def test_keep_first_and_last_on_single_row_are_identical():
    df = pd.DataFrame(
        {"a": [42], "b": ["only"]},
        index=pd.to_datetime(["2024-01-01"]),
    )

    first = keep_first(df)
    last = keep_last(df)

    pd.testing.assert_frame_equal(first, last)
    pd.testing.assert_frame_equal(first, df)



def test_take_mean_mixed_numeric_and_non_numeric():
    df = pd.DataFrame(
        {
            "num1": [1.0, 3.0],
            "num2": [10, 20],
            "cat": ["foo", "bar"],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-01"]),
    )

    result = take_mean(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    # Non-numeric: take first row's values
    assert result.iloc[0]["cat"] == "foo"

    # Numeric: mean
    assert result.iloc[0]["num1"] == pytest.approx((1.0 + 3.0) / 2)
    assert result.iloc[0]["num2"] == pytest.approx((10 + 20) / 2)


def test_take_mean_only_numeric_columns():
    df = pd.DataFrame(
        {
            "num1": [1.0, 3.0, 5.0],
            "num2": [10, 20, 30],
        },
        index=pd.to_datetime(["2024-01-01"] * 3),
    )

    result = take_mean(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    assert result.iloc[0]["num1"] == pytest.approx((1.0 + 3.0 + 5.0) / 3)
    assert result.iloc[0]["num2"] == pytest.approx((10 + 20 + 30) / 3)


def test_take_mean_handles_all_nan_numeric_column():
    df = pd.DataFrame(
        {
            "num1": [np.nan, np.nan],
            "num2": [1.0, 3.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-01"]),
    )

    result = take_mean(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    # num1: mean of all-NaN → NaN
    assert np.isnan(result.iloc[0]["num1"])
    # num2: mean of [1, 3]
    assert result.iloc[0]["num2"] == pytest.approx(2.0)
