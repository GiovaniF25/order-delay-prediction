"""Classify orders in batches to prioritize deliveries at risk of delay."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT = "order-delay-prediction"
FEATURES = ["distance_km", "queue_size", "rain", "weekend"]
TARGET = "delayed"


def validate(data, training):
    values = data[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Features must be finite and nonnegative.")
    queue_size = data["queue_size"].to_numpy(dtype=float)
    if (queue_size != np.floor(queue_size)).any():
        raise ValueError("queue_size must contain integers.")
    for column in ["rain", "weekend"] + ([TARGET] if training else []):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1.")


def fit_models(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, stratify=data[TARGET], random_state=42
    )
    if y_train.value_counts().reindex([0, 1], fill_value=0).min() < 5:
        raise ValueError("The training split must contain at least five examples of each class.")
    tree = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    search = GridSearchCV(
        tree,
        {"max_depth": [2, 3, 4, 6], "min_samples_leaf": [15, 35, 60],
         "criterion": ["gini", "entropy"]},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    ).fit(X_train, y_train)
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent").fit(X_train, y_train),
        "unrestricted_tree": tree.fit(X_train, y_train),
        "tuned_tree": search.best_estimator_,
    }
    metrics = {"best_parameters": search.best_params_, "test_rows": len(X_test)}
    for name, model in models.items():
        predicted = model.predict(X_test)
        metrics[name] = {
            "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
            "delay_f1": f1_score(y_test, predicted, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        }
    return search.best_estimator_, metrics


def predict(model, data):
    x = data[FEATURES]
    return {"delay_risk": model.predict_proba(x)[:, 1], "predicted_delay": model.predict(x)}


def read_csv(path, training=False):
    data = pd.read_csv(path)
    required = FEATURES + ([TARGET] if training else [])
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if data.empty or data[required].isna().any().any():
        raise ValueError("The CSV must contain rows and values in all required columns.")
    validate(data, training)
    return data


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train, evaluate, and save the model")
    train.add_argument("--data", type=Path, default=root / "data" / "training.csv")
    train.add_argument("--output", type=Path, default=root / "artifacts")
    prediction = commands.add_parser("predict", help="Generate predictions from a CSV")
    prediction.add_argument("--data", type=Path, required=True)
    prediction.add_argument("--model", type=Path, default=root / "artifacts" / "model.joblib")
    prediction.add_argument("--output", type=Path, default=root / "artifacts" / "predictions.csv")
    args = parser.parse_args()

    try:
        if args.command == "train":
            model, metrics = fit_models(read_csv(args.data, training=True))
            args.output.mkdir(parents=True, exist_ok=True)
            joblib.dump({"project": PROJECT, "model": model}, args.output / "model.joblib")
            report = json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False)
            (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
            print(report)
            print(f"Saved model and metrics to {args.output}")
        else:
            data = read_csv(args.data)
            bundle = joblib.load(args.model)
            if bundle["project"] != PROJECT:
                raise ValueError("The model belongs to another project.")
            result = data.assign(**predict(bundle["model"], data))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"{len(result)} predictions saved to {args.output}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
