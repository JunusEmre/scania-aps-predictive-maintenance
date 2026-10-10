# Scania APS Predictive Maintenance

Cost-sensitive predictive maintenance using the Scania APS Failure dataset (UCI). This repository supports a group Data Science project: data loading, data-quality review, an internal-validation Logistic Regression baseline, documentation, and automated tests. **Random Forest comparison and official test-set evaluation are not implemented yet.**

## Project overview

Operational data from trucks is used to investigate whether patterns in sensor and usage signals can help **identify vehicles that may have APS-related failures**, so maintenance staff can **prioritise inspections**. The aim is **decision support**—not an automatic repair or maintenance order, and **not** a prediction of the exact time of a future breakdown.

## Intended user

The primary audience includes roles such as:

- fleet maintenance manager  
- maintenance engineer  
- reliability engineer  

Model output should support human judgment about which vehicles warrant closer attention, not replace it.

## Group members

- Tim Rydén  
- Yunus Capar  
- Elza Capar  
- Eliesa Dudus  

## Dataset

| Item | Detail |
|------|--------|
| Name | Scania APS Failure dataset |
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) |
| Training set | 60,000 observations |
| Test set | 16,000 observations |
| Features | 170 anonymised input variables |
| Target | Column `class` with values `neg` (no failure) and `pos` (failure) |
| Missing values | Present (`na` in raw files) |
| Class balance | Strongly imbalanced |

Raw data is **not** stored in Git. Download the official files and place them locally under:

`data/raw/`

Expected files:

- `aps_failure_training_set.csv`  
- `aps_failure_test_set.csv`  
- `aps_failure_description.txt`  

See also `data/README.md`.

## Initial data-quality findings (training set only)

The following were computed from the **official training data** using `notebooks/01_data_quality.ipynb` and helpers in `src/data/preprocess.py`. Full detail is in `docs/data_quality.md`.

| Finding | Verified value |
|---------|----------------|
| Class distribution | `neg` 59,000 (98.33%), `pos` 1,000 (1.67%) |
| Overall missing share (feature cells) | ~8.33% |
| Features with ≥ 75% missing | 6: `ab_000`, `bo_000`, `bp_000`, `bq_000`, `br_000`, `cr_000` |
| Constant feature(s) | `cd_000` |
| Zero-heavy features (≥ 95% of non-missing values are 0) | 30 features |
| Completely duplicated rows | 0 |

No preprocessing was applied during the initial data-quality review. For the Logistic Regression baseline, preprocessing is performed inside the model pipeline using training data only.

## Evaluation direction

Primary business-oriented metric: **Scania cost structure** (`src/evaluation/metrics.py`)

- false positive cost = 10  
- false negative cost = 500  

Supporting metrics include precision, recall, F1-score, PR-AUC (where applicable), and the confusion matrix.

Because roughly **98%** of training rows are `neg`, **ordinary accuracy is misleading** (a model that always predicts “no failure” can look accurate while missing failures). Cost-based and recall-oriented measures are more appropriate for this problem.

**Logistic Regression baseline** — implemented in `src/models/baseline.py` and `notebooks/02_eda_baseline.ipynb` (internal 80/20 validation on training data only).

**Random Forest** — planned (`notebooks/03_random_forest.ipynb`); not implemented yet.

### Internal validation results — not official test results

From a stratified **80% development / 20% validation** split of the official **training** data (`random_state=42`), on the held-out validation fold:

| Model | Accuracy | Precision | Recall | F1 | FP | FN | Business cost |
|-------|---------:|----------:|-------:|---:|---:|---:|--------------:|
| Dummy (most frequent) | 0.983 | 0.000 | 0.000 | 0.000 | 0 | 200 | 100,000 |
| Logistic Regression | 0.968 | 0.336 | 0.920 | 0.492 | 364 | 16 | 11,640 |

Logistic Regression PR-AUC (validation): **0.721**. These numbers are for workflow comparison only; the official test set was not used.

## Test-set boundary

The **official test set is reserved for final evaluation only.**

It must **not** be used for:

- preprocessing decisions  
- feature selection  
- model selection  
- hyperparameter tuning  
- classification-threshold selection  

Those choices should be made using **training data only** (e.g. cross-validation on the training set).

## Repository structure

```
scania-aps-predictive-maintenance/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── raw/                 (.gitkeep; local CSVs not in Git)
│   └── processed/           (.gitkeep)
│
├── notebooks/
│   ├── 01_data_quality.ipynb      (initial data-quality review)
│   ├── 02_eda_baseline.ipynb      (EDA + Logistic Regression baseline)
│   ├── 03_random_forest.ipynb     (placeholder)
│   └── 04_model_evaluation.ipynb  (placeholder)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── random_forest.py       (placeholder)
│   └── evaluation/
│       └── metrics.py
│
├── tests/
│   ├── test_data_loading.py
│   ├── test_preprocess.py
│   ├── test_metrics.py
│   └── test_baseline.py
│
├── reports/
│   ├── figures/
│   └── results/
│
├── docs/
│   ├── data_quality.md
│   ├── decisions.md
│   └── ai_usage.md
│
└── presentation/
```

| Folder | Purpose |
|--------|---------|
| `data/` | Local project data and instructions; raw files stay on disk only |
| `notebooks/` | Analysis and (later) modelling notebooks |
| `src/data/` | Reusable data loading and data-quality helper code |
| `src/models/` | Model code (`baseline.py` implemented; Random Forest placeholder) |
| `src/evaluation/` | Shared evaluation logic (Scania business cost, validation metrics) |
| `tests/` | Automated tests |
| `reports/` | Final figures and result exports |
| `docs/` | Decisions, data-quality write-up, AI usage notes |
| `presentation/` | Presentation material |

## Setup (Windows)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data setup

1. Download the official Scania APS dataset from UCI (or the course-provided copy).  
2. Place the three original files in `data/raw/`.  
3. Do **not** commit raw CSVs to GitHub (they are ignored via `.gitignore`).

## Running tests

With the virtual environment activated:

```powershell
python -m pytest -v
```

Current behaviour:

- **Unit tests** (small synthetic data) run without the full dataset.  
- **Integration tests** in `tests/test_data_loading.py` load the real CSVs when both training and test files exist under `data/raw/`.  
- Those integration tests **skip cleanly** when the files are missing (`pytest.mark.skipif`).

## Current project status

**Completed**

- [x] Repository and folder structure  
- [x] Dependencies listed in `requirements.txt`  
- [x] APS CSV loader (`src/data/load_data.py`)  
- [x] Data-quality helpers (`src/data/preprocess.py`)  
- [x] Automated tests (loading, preprocess, metrics, baseline)  
- [x] Initial data-quality notebook and `docs/data_quality.md`  
- [x] EDA + Dummy reference + Logistic Regression baseline (`02_eda_baseline.ipynb`, `src/models/baseline.py`, `src/evaluation/metrics.py`)  

**In progress**

- [ ] Preprocessing experiments (e.g. dropping high-missing features) — group discussion  

**Planned**

- [ ] Random Forest (`03_random_forest.ipynb`, `src/models/random_forest.py`)  
- [ ] Model comparison and **official** test-set evaluation (`04_model_evaluation.ipynb`)  
- [ ] Error analysis and final demonstration (`presentation/`)  

## Known limitations

- Strong **class imbalance** makes naive accuracy a poor success measure.  
- Missing values are currently handled with median imputation and missing-value indicators in the Logistic Regression baseline. The final preprocessing strategy may change after model comparison.  
- **Anonymised** feature names limit physical interpretation; findings are statistical, not causal.  
- The project does **not** establish the physical root cause of an APS failure from data alone.  
- Outputs are **decision support**, not autonomous maintenance actions.  

## Generative AI

Use of generative AI in this project (if any) is documented in [`docs/ai_usage.md`](docs/ai_usage.md).
