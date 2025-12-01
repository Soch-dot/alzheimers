"""
Train several ML models on the cleaned dataset and save the best one.
Uses CLINICAL FEATURES ONLY (age, sex, education_years, mmse) for now.
MRI features will be added in future updates.
Run with: python src/train_models.py
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

# Try to import XGBoost, use fallback if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to a dataset (used for both original and synthetic)."""
    # 1. Basic categories
    df["age_group"] = pd.cut(df["age"], bins=[0, 70, 75, 80, 85, 100], labels=[0, 1, 2, 3, 4])
    df["age_group"] = df["age_group"].astype(float).fillna(2).astype(int)
    
    df["mmse_category"] = pd.cut(df["mmse"], bins=[0, 17, 23, 30], labels=[0, 1, 2])
    df["mmse_category"] = df["mmse_category"].astype(float).fillna(1).astype(int)
    
    # 2. EDUCATION-ADJUSTED MMSE
    df["mmse_education_adjusted"] = df["mmse"] + (df["education_years"] - 12) * 0.8
    df["mmse_education_adjusted"] = df["mmse_education_adjusted"].clip(0, 30)
    df["low_education"] = (df["education_years"] < 10).astype(int)
    
    # 3. INTERACTION FEATURES
    df["education_x_mmse"] = df["education_years"] * df["mmse"]
    df["age_x_mmse"] = df["age"] * df["mmse"]
    df["cdr_x_mmse"] = df["cdr"] * df["mmse"]
    df["education_x_cdr"] = df["education_years"] * df["cdr"]
    
    # 4. CDR-BASED FEATURES
    df["cdr_severe"] = (df["cdr"] >= 1.0).astype(int)
    df["cdr_mild"] = ((df["cdr"] > 0.0) & (df["cdr"] < 1.0)).astype(int)
    df["cdr_normal"] = (df["cdr"] == 0.0).astype(int)
    df["cdr_zero_override"] = (df["cdr"] == 0.0).astype(int) * 20
    df["cdr_high_override"] = (df["cdr"] >= 1.0).astype(int) * 15
    df["cdr_converted_indicator"] = (df["cdr"] == 0.5).astype(int) * 5
    
    # 5. MMSE RELATIVE TO EDUCATION
    df["mmse_relative_to_education"] = df["mmse"] / (df["education_years"] + 1)
    
    # 6. COGNITIVE RISK SCORE
    df["cognitive_risk_score"] = (30 - df["mmse"]) + (df["cdr"] * 10)
    
    # 7. AGE-ADJUSTED RISK
    df["age_risk_factor"] = np.where(df["age"] > 80, 1.0, 
                                     np.where(df["age"] > 75, 0.5, 0.0))
    
    # 8. NONLINEAR TRANSFORMS
    df["age_squared"] = df["age"] ** 2
    df["mmse_squared"] = df["mmse"] ** 2
    df["cdr_squared"] = df["cdr"] ** 2
    df["education_squared"] = df["education_years"] ** 2
    
    # 9. MMSE BIAS CORRECTION
    for edu_level in df["education_years"].unique():
        mask = df["education_years"] == edu_level
        if mask.sum() > 1:
            mmse_subset = df.loc[mask, "mmse"]
            if mmse_subset.std() > 0:
                df.loc[mask, "mmse_zscore_by_education"] = (
                    (df.loc[mask, "mmse"] - mmse_subset.mean()) / mmse_subset.std()
                )
    
    if "mmse_zscore_by_education" in df.columns:
        df["mmse_zscore_by_education"] = df["mmse_zscore_by_education"].fillna(0)
    else:
        df["mmse_zscore_by_education"] = 0
    
    expected_mmse_by_edu = df.groupby("education_years")["mmse"].mean()
    df["mmse_expected"] = df["education_years"].map(expected_mmse_by_edu)
    df["mmse_bias_corrected"] = df["mmse"] - df["mmse_expected"]
    df["mmse_bias_corrected"] = df["mmse_bias_corrected"].fillna(0)
    
    # 10. ADDITIONAL INTERACTION FEATURES
    df["age_x_cdr"] = df["age"] * df["cdr"]
    df["ses_x_education"] = df["ses"] * df["education_years"]
    df["age_x_education"] = df["age"] * df["education_years"]
    
    # 11. CDR-BASED RISK MULTIPLIERS
    df["cdr_risk_multiplier"] = np.where(df["cdr"] == 0.0, 0.1,
                                         np.where(df["cdr"] == 0.5, 2.0, 5.0))
    
    df = df.fillna(df.median())
    return df


def load_original_data() -> pd.DataFrame:
    """Load and prepare ORIGINAL dataset with clinical features only."""
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / "data" / "raw" / "Dataset.csv"
    
    print("Loading ORIGINAL dataset (clinical features only) from:", raw_path)
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
    
    # Apply feature engineering
    df = apply_feature_engineering(df)
    
    print(f"\nUsing {len(df.columns) - 1} features: {[c for c in df.columns if c != 'group']}")
    print(f"Total samples: {len(df)}")
    print(f"Target distribution:\n{df['group'].value_counts().sort_index()}")
    
    return df


def load_data_with_synthetic() -> pd.DataFrame:
    """Load original data + synthetic adversarial cases and combine them."""
    # Load original data
    df_original = load_original_data()
    original_count = len(df_original)
    
    # Load synthetic data
    base_dir = Path(__file__).resolve().parents[1]
    synthetic_path = base_dir / "data" / "synthetic" / "synthetic_adversarial_cases.csv"
    
    if synthetic_path.exists():
        print(f"\nLoading SYNTHETIC adversarial cases from: {synthetic_path}")
        df_synthetic = pd.read_csv(synthetic_path)
        
        # Ensure synthetic data has same columns as original (before feature engineering)
        required_cols = ["age", "sex", "education_years", "mmse", "cdr", "ses", "group"]
        missing_cols = [col for col in required_cols if col not in df_synthetic.columns]
        if missing_cols:
            raise ValueError(f"Synthetic data missing required columns: {missing_cols}")
        
        # Apply same feature engineering to synthetic data
        df_synthetic = apply_feature_engineering(df_synthetic)
        
        # Combine datasets
        df_combined = pd.concat([df_original, df_synthetic], ignore_index=True)
        
        # Shuffle to mix original and synthetic
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        synthetic_count = len(df_synthetic)
        print(f"\nCombined dataset:")
        print(f"  Original: {original_count} samples")
        print(f"  Synthetic: {synthetic_count} samples")
        print(f"  Total: {len(df_combined)} samples")
        print(f"\nCombined target distribution:\n{df_combined['group'].value_counts().sort_index()}")
        
        return df_combined
    else:
        print(f"\nWARNING: Synthetic data not found at {synthetic_path}")
        print("Training with original data only. Converted recall may be low.")
        return df_original


def build_models() -> Dict[str, Pipeline]:
    """Build models with improved hyperparameters and better class weights."""
    scaler = ("scaler", StandardScaler())
    
    # Custom class weights - give Converted class more weight
    # This helps distinguish Converted from Demented
    # Format: {class_index: weight}
    # Higher weight = model pays more attention to that class
    # Increased Converted weight to help distinguish it from Demented
    # Also slightly favor Nondemented when CDR=0
    custom_class_weights = {0: 1.2, 1: 10.0, 2: 1.0}  # Converted gets 10x weight, Nondemented slightly higher
    
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight=custom_class_weights,  # Custom weights instead of "balanced"
            random_state=42,
            C=0.5,  # Lower C = more regularization (prevents MMSE overfitting)
            solver="lbfgs",
            penalty="l2"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,  # Slightly shallower to prevent overfitting
            min_samples_split=10,  # Higher = less overfitting
            min_samples_leaf=5,  # Higher = less overfitting
            class_weight=custom_class_weights,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt"  # Use sqrt of features (prevents overfitting)
        ),
        "svm": SVC(
            kernel="rbf",
            probability=True,
            class_weight=custom_class_weights,
            random_state=42,
            C=5.0,  # Lower C = more regularization
            gamma="scale"
        ),
    }
    
    if XGBOOST_AVAILABLE:
        # XGBoost with scale_pos_weight equivalent for multi-class
        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,  # Shallower to prevent overfitting
            learning_rate=0.05,  # Lower learning rate = more careful learning
            random_state=42,
            eval_metric="mlogloss",
            min_child_weight=3,  # Higher = less overfitting
            subsample=0.8,  # Use 80% of data per tree (prevents overfitting)
            colsample_bytree=0.8  # Use 80% of features (prevents overfitting)
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
        "roc_auc_ovr": roc_auc_score(
            y_true, y_prob, multi_class="ovr"
        ),
    }


def train():
    # Load data WITH synthetic adversarial cases
    df = load_data_with_synthetic()
    X = df.drop(columns=["group"])
    y = df["group"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Print class distribution in training set
    print(f"\nTraining set class distribution:")
    print(y_train.value_counts().sort_index())
    print(f"\nTest set class distribution:")
    print(y_test.value_counts().sort_index())

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

