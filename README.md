# Scania APS Predictive Maintenance

Cost-sensitive predictive maintenance using the Scania APS Failure dataset (UCI). This repository documents a completed group Data Science project: data pipeline, data-quality review, Logistic Regression and Random Forest baselines, cost-based comparison, and **one official test-set evaluation** with frozen model settings.

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

The Scania/UCI dataset is **not redistributed** in this repository; download it locally and comply with the [UCI / source terms](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks). This project repository does not include a separate `LICENSE` file for the codebase — confirm licensing with the course/group if required.

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

**Logistic Regression** — `src/models/baseline.py`, `notebooks/02_eda_baseline.ipynb` (internal validation).

**Random Forest** — `src/models/random_forest.py`, `notebooks/03_random_forest.ipynb` (calibration on training data; settings then frozen).

**Official test evaluation** — `notebooks/04_model_evaluation.ipynb` (frozen thresholds and hyperparameters; no further tuning).

### Internal validation results — not official test results

Stratified **80% development / 20% validation** on official **training** data only (`random_state=42`; see `02_eda_baseline.ipynb`):

| Model | Accuracy | Precision | Recall | F1 | FP | FN | Business cost |
|-------|---------:|----------:|-------:|---:|---:|---:|--------------:|
| Dummy (most frequent) | 0.983 | 0.000 | 0.000 | 0.000 | 0 | 200 | 100,000 |
| Logistic Regression | 0.968 | 0.336 | 0.920 | 0.492 | 364 | 16 | 11,640 |

Logistic Regression PR-AUC (internal validation): **~0.72**.

### FINAL RESULTS — OFFICIAL TEST SET

Single final evaluation on **16,000** official test rows (**375** `pos`), with thresholds **LR = 0.460**, **RF = 0.105** (from notebook `03`, frozen in notebook `04`). Cost: **10 × FP + 500 × FN**.

| Model | Accuracy | Precision | Recall | F1 | FP | FN | Business cost | PR-AUC |
|-------|---------:|----------:|-------:|---:|---:|---:|--------------:|-------:|
| Dummy baseline | 0.977 | 0.000 | 0.000 | 0.000 | 0 | 375 | 187,500 | 0.023 |
| Logistic Regression | 0.971 | 0.441 | 0.944 | 0.601 | 449 | 21 | 14,990 | 0.802 |
| Random Forest | 0.978 | 0.518 | 0.960 | 0.673 | 335 | 15 | **10,850** | 0.899 |

**Accuracy alone is misleading** (Dummy ≈ 97.7% while missing every failure). **Random Forest** has the lowest business cost among compared models (~**27.6%** lower than Logistic Regression on the test set), but still misses **15** failures and triggers **335** false alarms — **decision support**, not autonomous maintenance.

## Test-set boundary

During development, the **official test set was not used** for preprocessing, feature selection, model choice, hyperparameters, or thresholds. Those decisions were made on **training data only** (internal validation and calibration in notebooks `02` and `03`).

After settings were frozen, the official test set was used **once** for final evaluation in `04_model_evaluation.ipynb`. It must **not** be reused for further tuning or model selection.

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
│   ├── 03_random_forest.ipynb     (Random Forest + calibration)
│   └── 04_model_evaluation.ipynb  (official test evaluation)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── random_forest.py
│   └── evaluation/
│       └── metrics.py
│
├── tests/
│   ├── test_data_loading.py
│   ├── test_preprocess.py
│   ├── test_metrics.py
│   ├── test_baseline.py
│   └── test_random_forest.py
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
    └── Scania APS Gruppresentation.pptx
```

| Folder | Purpose |
|--------|---------|
| `data/` | Local project data and instructions; raw files stay on disk only |
| `notebooks/` | Analysis and (later) modelling notebooks |
| `src/data/` | Reusable data loading and data-quality helper code |
| `src/models/` | Model code (`baseline.py`, `random_forest.py`) |
| `src/evaluation/` | Shared evaluation logic (Scania business cost, validation metrics) |
| `tests/` | Automated tests |
| `reports/` | Final figures and result exports |
| `docs/` | Decisions, data-quality write-up, AI usage notes |
| `presentation/` | Group presentation material (includes final deck `Scania APS Gruppresentation.pptx`) |

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
- [x] Dependencies (`requirements.txt`)
- [x] APS data loader and data-quality helpers
- [x] Automated tests (loading, preprocess, metrics, baseline, Random Forest)
- [x] Data-quality notebook and `docs/data_quality.md`
- [x] EDA + Dummy reference + Logistic Regression (`02_eda_baseline.ipynb`)
- [x] Random Forest + calibration (`03_random_forest.ipynb`, `src/models/random_forest.py`)
- [x] Cost-based comparison and **official test evaluation** (`04_model_evaluation.ipynb`)
- [x] Decision and AI-usage documentation (`docs/decisions.md`, `docs/ai_usage.md`)
- [x] Final group presentation (`presentation/Scania APS Gruppresentation.pptx`)

**Not in this repository**

- [ ] Exported figures in `reports/` (folders reserved; optional exports not committed)

## Known limitations

- Strong **class imbalance** makes naive accuracy a poor success measure.  
- Missing values are handled with median imputation and missing-value indicators inside the model pipelines (training-fit only within each workflow).
- **Anonymised** feature names limit physical interpretation; findings are statistical, not causal.  
- The project does **not** establish the physical root cause of an APS failure from data alone.  
- Outputs are **decision support**, not autonomous maintenance actions.  

## Generative AI

Use of generative AI in this project (if any) is documented in [`docs/ai_usage.md`](docs/ai_usage.md).
