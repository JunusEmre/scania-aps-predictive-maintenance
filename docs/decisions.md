# Project Decisions

This file records important project decisions and changes.

## Initial decisions

- Problem: identify APS-related failures from operational truck data.
- Primary user: maintenance / fleet decision-maker.
- Baseline model: Logistic Regression.
- Comparison model: Random Forest.
- Primary evaluation: Scania cost function.
- Supporting metrics: Precision, Recall, F1 and confusion matrix.
- Official test data reserved for a single final evaluation after settings were frozen.

## Decision log

### Data loading and quality

- Parse APS CSVs with 20 metadata lines skipped; treat `na` as missing.
- Target column: `class` (`neg` / `pos`); encode as 0/1 for modelling.
- Initial data-quality review on **training data only** (notebook `01_data_quality.ipynb`).

### Preprocessing (frozen for compared models)

- Drop constant feature `cd_000` inside model pipelines.
- Median imputation with missing-value indicators (`SimpleImputer`, fit on training data only within each workflow).
- Logistic Regression: `StandardScaler` + `class_weight="balanced"`.
- Keep high-missing and zero-heavy features for the baseline comparison (dropping them left as a possible later experiment).
- No SMOTE in the delivered baselines.

### Development protocol

- Notebooks `01`–`03` use **official training data only** for exploratory work, internal validation, and calibration.
- Stratified development/validation and calibration splits (see notebooks `02` and `03`).
- Thresholds chosen on calibration data, not on the official test set.

### Frozen model settings (before official test evaluation)

- **Logistic Regression threshold:** `0.460`
- **Selected Random Forest:** `n_estimators=110`, `max_depth=20`, `min_samples_leaf=2`, `class_weight={0: 1, 1: 15}`, `max_features="sqrt"`, `max_samples=0.8`, `n_jobs=2`, `random_state=42`
- **Random Forest threshold:** `0.105`
- Implemented in `src/models/baseline.py`, `src/models/random_forest.py`, and final evaluation notebook `04_model_evaluation.ipynb`.

### Final evaluation and recommendation

- Official test set used **once** for final metrics in `04_model_evaluation.ipynb` after settings above were fixed.
- **Primary business metric:** Scania cost `10 × FP + 500 × FN`.
- **Random Forest** recommended over Logistic Regression on official test cost and related metrics (see README and notebook `04`).

Add dated entries below when the group changes scope or methods after delivery.
