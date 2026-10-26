"""Selected Random Forest pipeline (frozen settings from notebook 03)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.baseline import CONSTANT_FEATURE_TO_DROP, DropColumnsTransformer

SELECTED_RF_PARAMS = {
    "n_estimators": 110,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "class_weight": {0: 1, 1: 15},
    "max_features": "sqrt",
    "max_samples": 0.8,
    "n_jobs": 2,
}


def build_selected_random_forest_pipeline(
    columns_to_drop: list[str] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build the fixed Random Forest pipeline used for final evaluation."""
    drop_cols = columns_to_drop or [CONSTANT_FEATURE_TO_DROP]
    classifier_params = {**SELECTED_RF_PARAMS, "random_state": random_state}
    return Pipeline(
        steps=[
            ("drop_constant", DropColumnsTransformer(columns=drop_cols)),
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("classifier", RandomForestClassifier(**classifier_params)),
        ]
    )
