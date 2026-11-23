"""Train and save a fraud classifier optimized for validation F1 score.

Examples:
    python credit_fraud_train.py --model xgboost
    python credit_fraud_train.py --model random_forest --n-estimators 200 --max-depth 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from credit_fraud_utils_data import get_feature_columns, load_dataset, split_features_target
from credit_fraud_utils_eval import evaluate_probabilities, find_best_threshold, save_metrics, save_model_artifact


def build_model(args: argparse.Namespace) -> Any:
    """Build one of the candidate classifiers with reproducible parameters."""
    logistic = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, solver="liblinear", random_state=args.random_state)),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced_subsample",
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    if args.model == "logistic":
        return logistic
    if args.model == "random_forest":
        return random_forest
    if args.model == "voting":
        return VotingClassifier(
            estimators=[("random_forest", random_forest), ("logistic", logistic)],
            voting="soft",
            weights=[2.0, 1.5],
            n_jobs=args.n_jobs,
        )
    if args.model == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=max(args.n_estimators, 300),
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
    if args.model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("xgboost is required for --model xgboost. Install it with: pip install xgboost") from error
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=2.0,
            eval_metric="logloss",
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            tree_method="hist",
        )
    raise ValueError(f"Unsupported model: {args.model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a credit-card fraud classifier and save model.pkl.")
    parser.add_argument("--train-path", default="data/split/train.csv", help="Path to labelled training CSV.")
    parser.add_argument("--val-path", default="data/split/val.csv", help="Path to labelled validation CSV.")
    parser.add_argument("--model", choices=["logistic", "random_forest", "voting", "extra_trees", "xgboost"], default="xgboost")
    parser.add_argument("--output-path", default="model.pkl", help="Destination for the pickle artifact.")
    parser.add_argument("--metrics-path", default="artifacts/validation_metrics.json", help="Destination for validation metrics JSON.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees for tree-based sklearn models.")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth for RandomForestClassifier.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers used by supported models.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_frame = load_dataset(args.train_path)
    validation_frame = load_dataset(args.val_path)
    train_x, train_y = split_features_target(train_frame)
    validation_x, validation_y = split_features_target(validation_frame)

    model = build_model(args)
    model.fit(train_x, train_y)
    validation_probabilities = model.predict_proba(validation_x)[:, 1]
    threshold, _ = find_best_threshold(validation_y.to_numpy(), validation_probabilities)
    metrics = evaluate_probabilities(validation_y.to_numpy(), validation_probabilities, threshold)
    metrics.update(
        {
            "model_name": args.model,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "fraud_rate_train": float(train_y.mean()),
            "fraud_rate_validation": float(validation_y.mean()),
        }
    )

    save_model_artifact(model, threshold, args.model, get_feature_columns(train_frame), args.output_path)
    save_metrics(metrics, args.metrics_path)
    print(f"Saved {args.model} model to {Path(args.output_path).resolve()}")
    print(f"Validation F1: {metrics['f1']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Threshold: {metrics['threshold']:.6f}")


if __name__ == "__main__":
    main()
