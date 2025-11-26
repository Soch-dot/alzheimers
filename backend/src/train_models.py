"""
Train several ML models on the cleaned dataset and save the best one.
Run with: python src/train_models.py
"""

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def load_data() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    processed_path = base_dir / "data" / "processed" / "clinical_clean.csv"
    print("Loading cleaned data from:", processed_path)
    return pd.read_csv(processed_path)


def build_models() -> Dict[str, Pipeline]:
    scaler = ("scaler", StandardScaler())
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
    }
    pipelines = {
        name: Pipeline([scaler, ("clf", model)]) for name, model in models.items()
    }
    return pipelines


def evaluate_model(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(
            y_true, y_prob, multi_class="ovr"
        ),
    }


def train():
    df = load_data()
    X = df.drop(columns=["group"])
    y = df["group"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    results = {}
    best_model_name = None
    best_score = -np.inf

    for name, pipeline in models.items():
        print(f"\nTraining model: {name}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        metrics = evaluate_model(y_test, y_pred, y_prob)
        results[name] = metrics
        print("Metrics:", metrics)
        print("Classification report:")
        print(classification_report(y_test, y_pred))

        if metrics["roc_auc_ovr"] > best_score:
            best_score = metrics["roc_auc_ovr"]
            best_model_name = name

    print("\nBest model:", best_model_name, "ROC-AUC:", best_score)

    best_pipeline = models[best_model_name]
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)
    print("Saved best model to:", model_path)

    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    train()

