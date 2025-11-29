import numpy as np
import pandas as pd
import pytest

from utils.cleaning import count_consecutive_nan_runs


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
