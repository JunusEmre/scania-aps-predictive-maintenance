import pandas as pd

from src.data.load_data import TARGET_COLUMN


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features from the target column.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def missingness_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return missing-value counts and percentages for every column.
    """

    missing_count = df.isna().sum()
    missing_percent = df.isna().mean() * 100

    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_percent,
        }
    )

    return summary.sort_values(
        "missing_percent",
        ascending=False,
    )


def find_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.75,
) -> list[str]:
    """
    Return columns where the fraction of missing values is
    greater than or equal to the chosen threshold.

    Example:
        threshold=0.75 means 75% or more missing.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    missing_fraction = df.isna().mean()

    return missing_fraction[
        missing_fraction >= threshold
    ].index.tolist()


def find_constant_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Find feature columns containing only one unique non-missing value.
    """

    feature_df = df.drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    )

    return [
        column
        for column in feature_df.columns
        if feature_df[column].nunique(dropna=True) <= 1
    ]


def find_zero_heavy_columns(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Find numerical features where at least `threshold` of
    non-missing observations are zero.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    feature_df = df.drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    )

    results = []

    for column in feature_df.columns:
        series = feature_df[column].dropna()

        if len(series) == 0:
            continue

        zero_fraction = (series == 0).mean()

        if zero_fraction >= threshold:
            results.append(
                {
                    "column": column,
                    "zero_percent": zero_fraction * 100,
                }
            )

    if not results:
        return pd.DataFrame(
            columns=["column", "zero_percent"]
        )

    return (
        pd.DataFrame(results)
        .sort_values("zero_percent", ascending=False)
        .reset_index(drop=True)
    )


def duplicate_count(
    df: pd.DataFrame,
) -> int:
    """
    Count completely duplicated rows.
    """

    return int(df.duplicated().sum())


def class_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise the target-class distribution.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    counts = df[TARGET_COLUMN].value_counts()
    percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
    )

    return pd.DataFrame(
        {
            "count": counts,
            "percent": percentages,
        }
    )