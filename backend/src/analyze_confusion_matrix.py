"""
Analyze confusion matrix to identify class confusion patterns.
This helps identify where the model struggles, especially with Converted class.
Run with: python src/analyze_confusion_matrix.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    """Load and prepare data (same as training)."""
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / "data" / "raw" / "Dataset.csv"
    
    df = pd.read_csv(raw_path)
    
    # Rename and encode (same as train_models.py)
    rename_map = {
        "M/F": "sex",
        "Age": "age",
        "EDUC": "education_years",
        "MMSE": "mmse",
        "CDR": "cdr",
        "SES": "ses",
        "Group": "group",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df["sex"] = df["sex"].map({"M": 1, "F": 0})
    group_mapping = {"Nondemented": 0, "Converted": 1, "Demented": 2}
    df["group"] = df["group"].map(group_mapping)
    
    clinical_features = ["age", "sex", "education_years", "mmse", "cdr", "ses", "group"]
    available_features = [f for f in clinical_features if f in df.columns]
    df = df[available_features].copy()
    
    # Fill missing
    for col in ["mmse", "ses", "cdr"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Feature engineering (same as training)
    df["age_group"] = pd.cut(df["age"], bins=[0, 70, 75, 80, 85, 100], labels=[0, 1, 2, 3, 4])
    df["age_group"] = df["age_group"].astype(float).fillna(2).astype(int)
    df["mmse_category"] = pd.cut(df["mmse"], bins=[0, 17, 23, 30], labels=[0, 1, 2])
    df["mmse_category"] = df["mmse_category"].astype(float).fillna(1).astype(int)
    df["mmse_education_adjusted"] = df["mmse"] + (df["education_years"] - 12) * 0.8
    df["mmse_education_adjusted"] = df["mmse_education_adjusted"].clip(0, 30)
    df["low_education"] = (df["education_years"] < 10).astype(int)
    df["education_x_mmse"] = df["education_years"] * df["mmse"]
    df["age_x_mmse"] = df["age"] * df["mmse"]
    df["cdr_x_mmse"] = df["cdr"] * df["mmse"]
    df["education_x_cdr"] = df["education_years"] * df["cdr"]
    df["cdr_severe"] = (df["cdr"] >= 1.0).astype(int)
    df["cdr_mild"] = ((df["cdr"] > 0.0) & (df["cdr"] < 1.0)).astype(int)
    df["cdr_normal"] = (df["cdr"] == 0.0).astype(int)
    df["cdr_zero_override"] = (df["cdr"] == 0.0).astype(int) * 20
    df["cdr_high_override"] = (df["cdr"] >= 1.0).astype(int) * 15
    df["cdr_converted_indicator"] = (df["cdr"] == 0.5).astype(int) * 5
    df["mmse_relative_to_education"] = df["mmse"] / (df["education_years"] + 1)
    df["cognitive_risk_score"] = (30 - df["mmse"]) + (df["cdr"] * 10)
    df["age_risk_factor"] = np.where(df["age"] > 80, 1.0, np.where(df["age"] > 75, 0.5, 0.0))
    
    # Nonlinear transforms
    df["age_squared"] = df["age"] * df["age"]
    df["mmse_squared"] = df["mmse"] * df["mmse"]
    df["cdr_squared"] = df["cdr"] * df["cdr"]
    df["education_squared"] = df["education_years"] * df["education_years"]
    
    # MMSE bias correction
    expected_mmse_by_edu = df.groupby("education_years")["mmse"].mean()
    df["mmse_expected"] = df["education_years"].map(expected_mmse_by_edu)
    df["mmse_bias_corrected"] = df["mmse"] - df["mmse_expected"]
    df["mmse_bias_corrected"] = df["mmse_bias_corrected"].fillna(0)
    
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
    
    # Additional interactions
    df["age_x_cdr"] = df["age"] * df["cdr"]
    df["ses_x_education"] = df["ses"] * df["education_years"]
    df["age_x_education"] = df["age"] * df["education_years"]
    df["cdr_risk_multiplier"] = np.where(df["cdr"] == 0.0, 0.1,
                                         np.where(df["cdr"] == 0.5, 2.0, 5.0))
    
    df = df.fillna(df.median())
    return df


def analyze_confusion():
    """Analyze confusion matrix and class confusion patterns."""
    print("=" * 80)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 80)
    
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "best_model.pkl"
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)
    
    # Load data
    df = load_data()
    X = df.drop(columns=["group"])
    y = df["group"]
    
    # Ensure feature order matches training
    expected_features = [
        "age", "sex", "education_years", "mmse", "cdr", "ses",
        "age_group", "mmse_category", "mmse_education_adjusted", "low_education",
        "education_x_mmse", "age_x_mmse", "cdr_x_mmse", "education_x_cdr",
        "cdr_severe", "cdr_mild", "cdr_normal", "cdr_zero_override", "cdr_high_override",
        "cdr_converted_indicator", "mmse_relative_to_education", "cognitive_risk_score",
        "age_risk_factor", "age_squared", "mmse_squared", "cdr_squared", "education_squared",
        "mmse_zscore_by_education", "mmse_expected", "mmse_bias_corrected",
        "age_x_cdr", "ses_x_education", "age_x_education", "cdr_risk_multiplier"
    ]
    X = X[[col for col in expected_features if col in X.columns]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = ["Nondemented", "Converted", "Demented"]
    
    print("\n[CONFUSION MATRIX]")
    print("-" * 80)
    print("Rows = True labels, Columns = Predicted labels")
    print("\n" + " " * 20 + "Predicted")
    print(" " * 15 + "".join([f"{name:>15}" for name in class_names]))
    for i, name in enumerate(class_names):
        row_str = f"{name:>15} |"
        for j in range(3):
            row_str += f"{cm[i, j]:>15}"
        print(row_str)
    
    # Calculate percentages
    print("\n[CONFUSION MATRIX (Percentages)]")
    print("-" * 80)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print(" " * 20 + "Predicted")
    print(" " * 15 + "".join([f"{name:>15}" for name in class_names]))
    for i, name in enumerate(class_names):
        row_str = f"{name:>15} |"
        for j in range(3):
            row_str += f"{cm_percent[i, j]:>14.1f}%"
        print(row_str)
    
    # Class confusion analysis
    print("\n[CLASS CONFUSION ANALYSIS]")
    print("-" * 80)
    
    # Nondemented <-> Converted confusion
    nondem_to_conv = cm[0, 1]
    conv_to_nondem = cm[1, 0]
    print(f"\nNondemented -> Converted: {nondem_to_conv} cases ({cm_percent[0, 1]:.1f}%)")
    print(f"Converted -> Nondemented: {conv_to_nondem} cases ({cm_percent[1, 0]:.1f}%)")
    total_nondem_conv_confusion = nondem_to_conv + conv_to_nondem
    print(f"Total Nondemented <-> Converted confusion: {total_nondem_conv_confusion} cases")
    
    # Converted <-> Demented confusion
    conv_to_dem = cm[1, 2]
    dem_to_conv = cm[2, 1]
    print(f"\nConverted -> Demented: {conv_to_dem} cases ({cm_percent[1, 2]:.1f}%)")
    print(f"Demented -> Converted: {dem_to_conv} cases ({cm_percent[2, 1]:.1f}%)")
    total_conv_dem_confusion = conv_to_dem + dem_to_conv
    print(f"Total Converted <-> Demented confusion: {total_conv_dem_confusion} cases")
    
    # Nondemented <-> Demented confusion (should be rare)
    nondem_to_dem = cm[0, 2]
    dem_to_nondem = cm[2, 0]
    print(f"\nNondemented -> Demented: {nondem_to_dem} cases ({cm_percent[0, 2]:.1f}%)")
    print(f"Demented -> Nondemented: {dem_to_nondem} cases ({cm_percent[2, 0]:.1f}%)")
    total_nondem_dem_confusion = nondem_to_dem + dem_to_nondem
    print(f"Total Nondemented <-> Demented confusion: {total_nondem_dem_confusion} cases")
    
    # Classification report
    print("\n[CLASSIFICATION REPORT]")
    print("-" * 80)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Converted class specific analysis
    print("\n[CONVERTED CLASS ANALYSIS]")
    print("-" * 80)
    converted_true = (y_test == 1).sum()
    converted_pred = (y_pred == 1).sum()
    converted_correct = ((y_test == 1) & (y_pred == 1)).sum()
    converted_recall = converted_correct / converted_true if converted_true > 0 else 0
    converted_precision = converted_correct / converted_pred if converted_pred > 0 else 0
    
    print(f"True Converted cases: {converted_true}")
    print(f"Predicted Converted cases: {converted_pred}")
    print(f"Correctly predicted Converted: {converted_correct}")
    print(f"Converted Recall: {converted_recall:.2%}")
    print(f"Converted Precision: {converted_precision:.2%}")
    
    # Summary
    print("\n[SUMMARY]")
    print("-" * 80)
    if total_conv_dem_confusion > total_nondem_conv_confusion:
        print("WARNING: Model struggles more with Converted <-> Demented distinction")
        print("This is the hardest boundary to learn.")
    elif total_nondem_conv_confusion > total_conv_dem_confusion:
        print("WARNING: Model struggles more with Nondemented <-> Converted distinction")
        print("Converted cases are being confused with healthy patients.")
        print("This is the MAIN WEAKNESS - Converted class needs more training data!")
    else:
        print("Model shows balanced confusion patterns.")
    
    if converted_recall < 0.7:
        print(f"\nCAUTION: Converted recall is {converted_recall:.1%} - needs improvement")
    elif converted_recall < 0.85:
        print(f"\nGOOD: Converted recall is {converted_recall:.1%} - acceptable but could be better")
    else:
        print(f"\nEXCELLENT: Converted recall is {converted_recall:.1%} - very good!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_confusion()

