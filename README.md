# Scania APS Predictive Maintenance

Cost-sensitive predictive maintenance using the Scania APS Failure dataset (UCI). This repository supports a group Data Science project at an early stage: data loading, data-quality review, documentation, and automated tests. **Modelling and final evaluation are not implemented yet.**

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

No final preprocessing (imputation, dropping columns, etc.) has been applied yet.

## Evaluation direction (planned)

**Not started in code yet**—this section describes the group’s intended direction from `docs/decisions.md`.

Planned models:

- **Logistic Regression** — simple, interpretable baseline  
- **Random Forest** — non-linear comparison model  

Primary business-oriented metric: **Scania cost structure**

- false positive cost = 10  
- false negative cost = 500  

Supporting metrics may include precision, recall, F1-score, and the confusion matrix.

Because roughly **98%** of training rows are `neg`, **ordinary accuracy is misleading** (a model that always predicts “no failure” can look accurate while missing failures). Cost-based and recall-oriented measures are more appropriate for this problem.

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
│   ├── 02_eda_baseline.ipynb      (placeholder for later EDA / baseline)
│   ├── 03_random_forest.ipynb     (placeholder)
│   └── 04_model_evaluation.ipynb  (placeholder)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── baseline.py            (placeholder)
│   │   └── random_forest.py       (placeholder)
│   └── evaluation/
│       └── metrics.py             (placeholder)
│
├── tests/
│   ├── test_data_loading.py
│   └── test_preprocess.py
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
| `src/models/` | Model code (planned; files present but not implemented) |
| `src/evaluation/` | Evaluation logic (planned; not implemented) |
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
- [x] Automated tests (`tests/test_data_loading.py`, `tests/test_preprocess.py`)  
- [x] Initial data-quality notebook and `docs/data_quality.md`  

**In progress**

- [ ] Preprocessing strategy (group review; no irreversible steps committed in code)  

**Planned**

- [ ] Exploratory analysis and Logistic Regression baseline (`02_eda_baseline.ipynb`, `src/models/baseline.py`)  
- [ ] Random Forest (`03_random_forest.ipynb`, `src/models/random_forest.py`)  
- [ ] Model comparison and Scania cost evaluation (`04_model_evaluation.ipynb`, `src/evaluation/metrics.py`)  
- [ ] Error analysis and final demonstration (`presentation/`)  

## Known limitations

- Strong **class imbalance** makes naive accuracy a poor success measure.  
- Widespread **missing values** require an agreed imputation or modelling strategy.  
- **Anonymised** feature names limit physical interpretation; findings are statistical, not causal.  
- The project does **not** establish the physical root cause of an APS failure from data alone.  
- Outputs are **decision support**, not autonomous maintenance actions.  

## Generative AI

Use of generative AI in this project (if any) is documented in [`docs/ai_usage.md`](docs/ai_usage.md).
