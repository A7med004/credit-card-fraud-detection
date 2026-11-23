# Credit Card Fraud Detection - Project Report

## Executive summary

The objective was to identify fraudulent transactions in an extremely imbalanced binary-classification dataset. The selected operating metric was F1 score, as required, because it balances fraud precision and recall. PR-AUC was reported as a secondary metric because it is more informative than ROC-AUC or accuracy when fraud is rare.

The final artifact is an XGBoost classifier with a validation-selected probability threshold. On the held-out validation set, it achieved **F1 = 0.8862** and **PR-AUC = 0.8543**. The threshold was **0.303188**, rather than the default 0.50.

## Data understanding

| Split | Rows | Fraud cases | Fraud rate |
| --- | ---: | ---: | ---: |
| Train | 170,884 | 305 | 0.178% |
| Validation | 56,960 | 90 | 0.158% |

The dataset has 30 input features: `Time`, `Amount`, and anonymized PCA features `V1` through `V28`. There are no missing values. The class distribution shows why raw accuracy is unsuitable: a classifier that always predicts non-fraud would appear highly accurate while detecting no fraud at all.

## Modelling approach

I kept the validation data separate from training during model selection. I explored the required logistic-regression, random-forest, and soft-voting approaches, along with extra trees and XGBoost. Tree ensembles are well suited to nonlinear interactions in the anonymized variables. The selected XGBoost configuration uses 400 trees, maximum depth 4, learning rate 0.05, row and feature subsampling, L2 regularization, and a fixed random seed.

For every candidate, the model produced fraud probabilities on the validation set. I selected the threshold that maximized F1 using the precision-recall curve instead of assuming 0.50 was the correct business threshold.

## Final validation performance

| Metric | Result |
| --- | ---: |
| F1 score | 0.8862 |
| PR-AUC | 0.8543 |
| Precision | 0.9610 |
| Recall | 0.8222 |
| Optimal threshold | 0.303188 |
| False positives | 3 |
| False negatives | 16 |

The model is conservative at the selected threshold: most transactions marked as fraud are genuine fraud cases, while it still recovers more than 82% of validation fraud.

## Lessons learned

1. Accuracy is not a useful model-selection metric for severe class imbalance.
2. Threshold tuning is an essential part of the classifier, not a post-processing detail.
3. Validation data must remain separate while comparing models and choosing a threshold.
4. PR-AUC evaluates the quality of probability ranking, while F1 evaluates the chosen operating point; both are needed to understand performance.
5. Reproducible artifacts require fixed random seeds, explicit feature order, saved metrics, and a documented prediction contract.

## Reproducibility and submission

Run the following command from the project directory:

```powershell
python -m pip install -r requirements.txt
python credit_fraud_train.py --model xgboost
```

The generated `model.pkl` is a dictionary with `model`, `threshold`, `model_name`, and `feature_columns`. A scoring program should order its input using `feature_columns`, call `model.predict_proba(features)[:, 1]`, and label scores greater than or equal to `threshold` as fraud.
