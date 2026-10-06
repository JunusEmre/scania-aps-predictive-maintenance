"""Tests for APS CSV loading."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import (
    TARGET_COLUMN,
    load_aps_csv,
    load_test_data,
    load_training_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAINING_CSV = RAW_DIR / "aps_failure_training_set.csv"
TEST_CSV = RAW_DIR / "aps_failure_test_set.csv"

requires_real_aps_data = pytest.mark.skipif(
    not (TRAINING_CSV.is_file() and TEST_CSV.is_file()),
    reason="APS CSV files not found under data/raw/",
)


def _write_sample_aps_csv(path: Path) -> None:
    metadata = [f"copyright or metadata line {i}" for i in range(20)]
    lines = metadata + [
        "class,aa_000,ab_000",
        "neg,1,2",
        "pos,3,na",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_aps_csv_skips_metadata_parses_na_and_uses_class_column(tmp_path):
    csv_path = tmp_path / "sample_aps.csv"
    _write_sample_aps_csv(csv_path)

    df = load_aps_csv(csv_path)

    assert list(df.columns) == ["class", "aa_000", "ab_000"]
    assert TARGET_COLUMN in df.columns
    assert len(df) == 2
    assert df.loc[0, "class"] == "neg"
    assert pd.isna(df.loc[1, "ab_000"])


@requires_real_aps_data
def test_training_set_has_expected_row_count():
    train = load_training_data(RAW_DIR)
    assert len(train) == 60_000


@requires_real_aps_data
def test_test_set_has_expected_row_count():
    test = load_test_data(RAW_DIR)
    assert len(test) == 16_000


@requires_real_aps_data
def test_train_and_test_have_171_columns():
    train = load_training_data(RAW_DIR)
    test = load_test_data(RAW_DIR)
    assert train.shape[1] == 171
    assert test.shape[1] == 171


@requires_real_aps_data
def test_train_and_test_share_column_names():
    train = load_training_data(RAW_DIR)
    test = load_test_data(RAW_DIR)
    assert list(train.columns) == list(test.columns)


@requires_real_aps_data
def test_target_column_is_class():
    train = load_training_data(RAW_DIR)
    assert train.columns[0] == "class"


@requires_real_aps_data
def test_target_values_are_neg_and_pos_only():
    train = load_training_data(RAW_DIR)
    test = load_test_data(RAW_DIR)
    assert set(train["class"].unique()) == {"neg", "pos"}
    assert set(test["class"].unique()) == {"neg", "pos"}


@requires_real_aps_data
def test_target_has_no_missing_values():
    train = load_training_data(RAW_DIR)
    test = load_test_data(RAW_DIR)
    assert train["class"].notna().all()
    assert test["class"].notna().all()
