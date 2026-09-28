# Diabetes survey classification

This project explores a CSV of 520 survey responses and predicts the probability that a row has a `Positive` diabetes class from age, gender, and symptom answers. It is a learning project, **not a medical diagnosis tool**.

The source file is [`diabetes_data_upload.csv`](diabetes_data_upload.csv). It has 17 columns: 16 inputs and the `class` label (`Positive` or `Negative`).

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn
```

Run the workflow from the project root because the scripts use relative paths:

```bash
python eda.py
python cleaning.py
python modeling.py
```

`modeling.py` imports `cleaning.py` and `evaluation.py`, so running it performs data preparation, model comparison, and final evaluation. Running `cleaning.py` alone prints a short data summary.

## Workflow

| Script | What it does | Output |
| --- | --- | --- |
| `eda.py` | Counts rows, missing values, duplicate rows, and classes; charts class counts and symptom shares. | `doc/eda_results/` |
| `cleaning.py` | Encodes `Positive`/`Negative`, gender, and Yes/No answers as numbers; rejects missing or unexpected answers. | Data returned to `modeling.py`; summary when run alone |
| `modeling.py` | Reserves a stratified 20% test set, compares three models with five stratified training folds, and selects the lowest mean RMSE. | `doc/model_results/` |
| `evaluation.py` | Scores the chosen model on the held-out test set and plots a confusion matrix at a 0.5 probability cutoff. | `doc/evaluation_result/test_predictions.png` |

The models are logistic regression, random forest, and gradient boosting. MAE, RMSE, and R² compare predicted probabilities with the 0/1 class labels. Accuracy and the confusion matrix use a 0.5 cutoff. The test set is used only after model selection.

## Duplicate-row experiment

The CSV contains **269 exact duplicate rows**: 520 total rows and 251 distinct rows. The original class counts are 320 Positive and 200 Negative; after exact deduplication they are 173 Positive and 78 Negative.

In the current [`cleaning.py`](../cleaning.py), the `drop_duplicates()` block is **commented out for an experiment**. A run of `modeling.py` therefore uses all 520 rows. To run the deduplicated version, uncomment that block and rerun `python modeling.py`; it will print `Removed 269 exact duplicate rows; 251 rows remain.` The output files are overwritten on each run, so record or copy results from each version before switching.

Identical rows can appear across training, validation, and test splits when duplicates are retained. This can make performance look better than it would on distinct responses. Removing duplicates also changes the sample size and class balance, so a score difference alone does not measure the effect of leakage. The CSV does not identify respondents, so identical answers cannot be confirmed as repeat submissions by the same person.

## Saved results

The existing [`model_winner.txt`](model_results/model_winner.txt) reports random forest as the winner by five-fold mean RMSE, with held-out test MAE **0.209**, RMSE **0.292**, R² **0.605**, and accuracy **0.902**. These files reflect the last run and are overwritten when the code is rerun; check the duplicate setting in `cleaning.py` when interpreting them.

- [`cross_validation_results.csv`](model_results/cross_validation_results.csv): mean scores for each model
- [`model_comparison.png`](model_results/model_comparison.png): mean MAE and RMSE chart
- [`test_predictions.png`](evaluation_result/test_predictions.png): held-out test confusion matrix
