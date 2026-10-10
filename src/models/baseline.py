"""Logistic Regression baseline and simple reference classifiers."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CONSTANT_FEATURE_TO_DROP = "cd_000"
LOGISTIC_REGRESSION_MAX_ITER = 2000


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Remove selected columns from a feature matrix (pandas DataFrame)."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


def build_logistic_regression_pipeline(
    columns_to_drop: list[str] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Preprocessing + Logistic Regression for the APS baseline.

    Fitted preprocessing (imputer, scaler) must be learned on development
    training data only when used inside a proper train/validation workflow.
    """
    drop_cols = columns_to_drop or [CONSTANT_FEATURE_TO_DROP]
    return Pipeline(
        steps=[
            ("drop_constant", DropColumnsTransformer(columns=drop_cols)),
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=LOGISTIC_REGRESSION_MAX_ITER,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_dummy_reference_classifier() -> DummyClassifier:
    """Most-frequent class reference (not a real ML model)."""
    return DummyClassifier(strategy="most_frequent")
