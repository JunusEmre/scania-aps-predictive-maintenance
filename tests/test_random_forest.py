"""Tests for the selected Random Forest pipeline."""

import numpy as np
import pandas as pd

from src.models.baseline import CONSTANT_FEATURE_TO_DROP
from src.models.random_forest import build_selected_random_forest_pipeline


def test_selected_random_forest_pipeline_fits_and_predicts_binary():
    X = pd.DataFrame(
        {
            "feat_a": [1.0, np.nan, 3.0, 4.0],
            "feat_b": [0.0, 1.0, 2.0, 3.0],
            CONSTANT_FEATURE_TO_DROP: [7.0, 7.0, 7.0, 7.0],
        }
    )
    y = np.array([0, 0, 1, 1])

    pipeline = build_selected_random_forest_pipeline(random_state=42)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
