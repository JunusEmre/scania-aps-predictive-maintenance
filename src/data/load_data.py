"""Load Scania APS failure datasets from CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METADATA_HEADER_LINES = 20
NA_VALUES = ["na"]
TARGET_COLUMN = "class"

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TRAINING_FILE = "aps_failure_training_set.csv"
TEST_FILE = "aps_failure_test_set.csv"


def _resolve_raw_dir(raw_dir: Path | str | None) -> Path:
    if raw_dir is None:
        return DEFAULT_RAW_DIR
    return Path(raw_dir)


def load_aps_csv(path: Path | str) -> pd.DataFrame:
    """Load one APS CSV, skipping metadata lines and treating ``na`` as missing."""
    return pd.read_csv(
        path,
        skiprows=METADATA_HEADER_LINES,
        na_values=NA_VALUES,
        low_memory=False,
    )


def training_data_path(raw_dir: Path | str | None = None) -> Path:
    return _resolve_raw_dir(raw_dir) / TRAINING_FILE


def test_data_path(raw_dir: Path | str | None = None) -> Path:
    return _resolve_raw_dir(raw_dir) / TEST_FILE


def load_training_data(raw_dir: Path | str | None = None) -> pd.DataFrame:
    return load_aps_csv(training_data_path(raw_dir))


def load_test_data(raw_dir: Path | str | None = None) -> pd.DataFrame:
    return load_aps_csv(test_data_path(raw_dir))
