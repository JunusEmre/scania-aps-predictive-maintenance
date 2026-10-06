# Data

## Dataset

Scania APS Failure at Scania Trucks.

Source:
UCI Machine Learning Repository.

The project uses:

- Training set: 60,000 observations
- Test set: 16,000 observations
- 170 input variables
- Target: `class`

The dataset contains missing values and a strongly imbalanced target.

## Important

Raw data should be downloaded from the original source and placed locally in:

`data/raw/`

The raw dataset is not committed automatically to this repository.

Expected files:

- `aps_failure_training_set.csv`
- `aps_failure_test_set.csv`
- `aps_failure_description.txt`

## Data handling

The official test set must remain separate from model development.

Cleaning, imputation, feature decisions and model selection should be performed using training data only.

The test set is reserved for final evaluation.
