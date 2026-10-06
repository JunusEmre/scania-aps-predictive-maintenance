"""Unit tests for data-quality helper functions."""

import pandas as pd
import pytest

from src.data.preprocess import (
    TARGET_COLUMN,
    class_distribution,
    duplicate_count,
    find_constant_columns,
    find_high_missing_columns,
    find_zero_heavy_columns,
    missingness_summary,
    split_features_target,
)


def _sample_frame() -> pd.DataFrame:
    """Ten-row frame for missingness / constant-column checks."""
    return pd.DataFrame(
        {
            "class": ["neg"] * 9 + ["pos"],
            "a": list(range(10)),
            "b": [0.0] * 10,
            "c": [1.0, 2.0, 1.0, 2.0, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
            "d": [1.0, 2.0, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )


def _class_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": ["neg", "neg", "pos", "neg"],
            "x": [1, 2, 3, 4],
        }
    )


def test_split_features_target_returns_copies():
    df = _sample_frame()
    original = df.copy()
    X, y = split_features_target(df)

    assert TARGET_COLUMN not in X.columns
    assert list(y) == list(df[TARGET_COLUMN])
    assert len(X) == len(df)
    pd.testing.assert_frame_equal(df, original)
    X.iloc[0, 0] = 999
    assert df.loc[0, "a"] != 999


def test_split_features_target_requires_class_column():
    with pytest.raises(ValueError, match="Target column"):
        split_features_target(pd.DataFrame({"x": [1]}))


def test_missingness_summary_sorts_by_percent_descending():
    df = _sample_frame()
    summary = missingness_summary(df)

    assert list(summary.columns) == ["missing_count", "missing_percent"]
    assert summary.loc["d", "missing_count"] == 8
    assert summary.loc["d", "missing_percent"] == pytest.approx(80.0)
    assert summary.index[0] == "d"


def test_find_high_missing_columns_respects_threshold():
    df = _sample_frame()
    high = find_high_missing_columns(df, threshold=0.75)
    assert high == ["d"]

    none = find_high_missing_columns(df, threshold=0.81)
    assert none == []


def test_find_high_missing_columns_validates_threshold():
    df = _sample_frame()
    with pytest.raises(ValueError, match="threshold"):
        find_high_missing_columns(df, threshold=1.5)


def test_find_constant_columns_ignores_target():
    df = _sample_frame()
    assert find_constant_columns(df) == ["b"]


def test_duplicate_count():
    df = pd.DataFrame({"class": ["neg", "neg"], "x": [1, 2]})
    assert duplicate_count(df) == 0

    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert duplicate_count(df_dup) == 1


def test_class_distribution_counts_and_percentages():
    df = _class_frame()
    dist = class_distribution(df)

    assert set(dist.index) == {"neg", "pos"}
    assert dist.loc["neg", "count"] == 3
    assert dist.loc["pos", "count"] == 1
    assert dist.loc["neg", "percent"] == pytest.approx(75.0)
    assert dist.loc["pos", "percent"] == pytest.approx(25.0)


def test_class_distribution_requires_target():
    with pytest.raises(ValueError, match="Target column"):
        class_distribution(pd.DataFrame({"x": [1]}))


def test_find_zero_heavy_columns():
    df = pd.DataFrame(
        {
            "class": ["neg"] * 5,
            "mostly_zero": [0, 0, 0, 0, 1],
            "mixed": [0, 1, 2, 3, 4],
        }
    )
    result = find_zero_heavy_columns(df, threshold=0.8)

    assert len(result) == 1
    assert result.loc[0, "column"] == "mostly_zero"
    assert result.loc[0, "zero_percent"] == pytest.approx(80.0)


def test_find_zero_heavy_columns_validates_threshold():
    df = _sample_frame()
    with pytest.raises(ValueError, match="threshold"):
        find_zero_heavy_columns(df, threshold=-0.1)
