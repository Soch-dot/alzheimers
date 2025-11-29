"""
Train models using CLINICAL FEATURES ONLY on ORIGINAL dataset.
This gives us the best accuracy for the prototype.
Run with: python src/train_models_clinical_only.py
"""

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

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_original_data() -> pd.DataFrame:
    """Load and prepare ORIGINAL dataset with clinical features only."""
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / "data" / "raw" / "Dataset.csv"
    
    print("Loading ORIGINAL dataset (not merged) from:", raw_path)
    df = pd.read_csv(raw_path)
    
    # Rename columns
    rename_map = {
        "M/F": "sex",
        "Age": "age",
        "EDUC": "education_years",
        "MMSE": "mmse",
        "CDR": "cdr",  # Clinical Dementia Rating - KEY clinical feature!
        "SES": "ses",  # Socioeconomic Status - also clinical
        "Group": "group",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    
    # Encode
    df["sex"] = df["sex"].map({"M": 1, "F": 0})
    group_mapping = {"Nondemented": 0, "Converted": 1, "Demented": 2}
    df["group"] = df["group"].map(group_mapping)
    
    # Select clinical features (CDR is clinical assessment, not MRI!)
    clinical_features = ["age", "sex", "education_years", "mmse", "cdr", "ses", "group"]
    available_features = [f for f in clinical_features if f in df.columns]
    df = df[available_features].copy()
    
    # Fill missing values
    for col in ["mmse", "ses", "cdr"]:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled missing {col.upper()} with median={median_val:.2f}")
    
    # Feature engineering
    df["age_group"] = pd.cut(df["age"], bins=[0, 70, 75, 80, 85, 100], labels=[0, 1, 2, 3, 4])
    df["age_group"] = df["age_group"].astype(float).fillna(2).astype(int)
    
    df["mmse_category"] = pd.cut(df["mmse"], bins=[0, 17, 23, 30], labels=[0, 1, 2])
    df["mmse_category"] = df["mmse_category"].astype(float).fillna(1).astype(int)
    
    print(f"\nUsing {len(df.columns) - 1} features: {[c for c in df.columns if c != 'group']}")
    print(f"Total samples: {len(df)}")
    print(f"Target distribution:\n{df['group'].value_counts().sort_index()}")
    
    return df


def build_models() -> Dict[str, Pipeline]:
    """Build models with optimized hyperparameters."""
    scaler = ("scaler", StandardScaler())
    
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            C=1.0,
            solver="lbfgs"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "svm": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
            C=10.0,
            gamma="scale"
        ),
    }
    
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss"
        )
    
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
        "roc_auc_ovr": roc_auc_score(y_true, y_prob, multi_class="ovr"),
    }


def train():
    df = load_original_data()
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

    print("\n" + "=" * 80)
    print(f"Best model: {best_model_name} | ROC-AUC: {best_score:.4f} | Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("=" * 80)

    best_pipeline = models[best_model_name]
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved best model to: {model_path}")


if __name__ == "__main__":
    train()

