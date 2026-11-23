# Credit Card Fraud Detection

This project detects fraudulent credit-card transactions in a highly imbalanced binary-classification dataset. The selection objective is **validation F1 score**; PR-AUC is reported as a secondary ranking metric.

## Deliverables

- `fraud_detection_analysis.ipynb`: exploratory analysis and model comparison.
- `credit_fraud_train.py`: command-line training entry point.
- `credit_fraud_utils_data.py`: data validation and preparation helpers.
- `credit_fraud_utils_eval.py`: threshold selection, evaluation, and artifact helpers.
- `model.pkl`: final model dictionary with `model`, `threshold`, and `model_name` keys.

## Quick start

```powershell
python -m pip install -r requirements.txt
python credit_fraud_train.py --model xgboost
```

The training command writes `model.pkl` and `artifacts/validation_metrics.json`.

## Model artifact contract

`model.pkl` contains a dictionary. To generate labels for a future CSV, drop `Class` when it is present, preserve the feature-column order stored in `feature_columns`, calculate `model.predict_proba(features)[:, 1]`, and classify scores greater than or equal to `threshold` as fraud.

## Reproducibility notes

- The source files use a fixed random seed by default.
- The validation set is kept separate while choosing the probability threshold.
- Class imbalance is handled with class weights for the sklearn baselines and native gradient boosting for the selected model.
- `trainval.csv` currently duplicates `val.csv`, so it is intentionally not used as a combined training dataset.
