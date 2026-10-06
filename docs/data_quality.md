# Data quality review — Scania APS training set

This document summarises the **initial** data-quality review for the official **training** dataset only. No imputation, feature removal, or modelling decisions have been finalised. The official **test** set (`aps_failure_test_set.csv`) remains reserved for final evaluation and was not used for these exploratory findings.

## Dataset dimensions

| Item | Value |
|------|------:|
| Training observations | 60,000 |
| Columns (total) | 171 |
| Input features | 170 |
| Target column | `class` (`neg`, `pos`) |

The held-out test set contains 16,000 observations with the same 171 columns; it is not analysed here.

## Class imbalance

| Class | Count | Share |
|-------|------:|------:|
| `neg` | 59,000 | 98.33% |
| `pos` | 1,000 | 1.67% |

The positive (failure) class is heavily under-represented. **Ordinary accuracy can be misleading**: a naive model that always predicts `neg` would achieve about 98.3% accuracy while failing to detect failures. Group evaluation should prioritise metrics aligned with the Scania cost function and recall-oriented views discussed in the project brief, not accuracy alone.

## Missing values

- **Overall missing share among feature cells** (training features only): about **8.33%** of all feature values are missing (`na` in the raw files, parsed as NaN).
- Missingness is **uneven across features**: many columns have little or no missing data; a subset is dominated by missing values.

### Features with ≥ 75% missing (training)

Six features meet or exceed 75% missingness:

- `ab_000`, `bo_000`, `bp_000`, `bq_000`, `br_000`, `cr_000`

Several others approach this level (for example `bn_000` at about 73.3% missing) and may warrant the same discussion if the group lowers or adjusts the threshold.

The top features by missing percentage in the training set include `br_000`, `bq_000`, `bp_000`, `bo_000`, `ab_000`, and `cr_000` (all above 77% in the current local copy).

## Constant and zero-heavy features

- **Constant features** (single non-missing value, excluding target): **`cd_000`** appears constant in the training data.
- **Zero-heavy features** (≥ 95% of non-missing values equal to zero): **30** features meet this criterion; examples include `ch_000`, `as_000`, `au_000`, and `ag_000` (see notebook for the full ranked list).

Constant or near-constant columns may carry little discriminative signal but could still interact with other features after transformation; the group should decide whether to drop, keep, or revisit them after modelling experiments.

## Duplicate rows

- **Completely duplicated training rows:** **0**

## Anonymised feature limitation

Sensor and operational fields use opaque codes (for example `aa_000`, `ab_000`). **We do not assign real-world physical meanings** to these names in this project; quality review is statistical only (missingness, sparsity, variance), not domain interpretation.

## Test set policy

All exploratory quality checks and future preprocessing choices must be informed by **training data only**. The official test set must not drive imputation rules, feature selection, or threshold tuning.

## Open decisions for the group

The following still require explicit group agreement before any irreversible preprocessing:

1. **High-missing features** — drop all six (or more) columns above 75% missing, impute, or use models that handle missingness natively?
2. **Imputation strategy** — median/mean, zero, indicator flags, or multivariate imputation (fit on training only)?
3. **Constant column `cd_000`** — remove or retain?
4. **Zero-heavy features** — treat zeros as meaningful, collapse, or remove low-variance columns?
5. **Class imbalance** — resampling, class weights, threshold tuning, or cost-sensitive learning?
6. **Evaluation** — confirm primary metric (Scania cost function) and how the reserved test set will be used once.

Reproducible helpers live in `src/data/preprocess.py`; the interactive walkthrough is in `notebooks/01_data_quality.ipynb`.
