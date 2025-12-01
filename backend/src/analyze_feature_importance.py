"""
Analyze feature importance using SHAP values.
This helps identify if MMSE is still dominating the model.
Run with: python src/analyze_feature_importance.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")


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
    df["age_squared"] = df["age"] ** 2
    df["mmse_squared"] = df["mmse"] ** 2
    df["cdr_squared"] = df["cdr"] ** 2
    df["education_squared"] = df["education_years"] ** 2
    
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


def analyze_importance():
    """Analyze feature importance."""
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    if not SHAP_AVAILABLE:
        print("\nNote: SHAP not available. Using feature_importances_ for tree-based models.")
        print("Install SHAP for more detailed analysis: pip install shap\n")
    
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "best_model.pkl"
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)
    
    # Load data
    df = load_data()
    X = df.drop(columns=["group"])
    y = df["group"]
    
    # Ensure feature order matches training (same as train_models.py)
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
    
    # Reorder columns to match expected order
    X = X[[col for col in expected_features if col in X.columns]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get feature names (after ensuring correct order)
    feature_names = X.columns.tolist()
    
    # For tree-based models, use feature_importances_
    if hasattr(model.named_steps['clf'], 'feature_importances_'):
        print("\n[FEATURE IMPORTANCE] (Tree-based model)")
        print("-" * 80)
        importances = model.named_steps['clf'].feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(importance_df.to_string(index=False))
        
        # Check MMSE dominance
        mmse_features = [f for f in feature_names if 'mmse' in f.lower()]
        mmse_total_importance = importance_df[importance_df['feature'].isin(mmse_features)]['importance'].sum()
        total_importance = importances.sum()
        mmse_percentage = (mmse_total_importance / total_importance) * 100
        
        print(f"\nMMSE-related features total importance: {mmse_percentage:.2f}%")
        if mmse_percentage > 40:
            print("WARNING: MMSE is still dominating (>40%)")
        elif mmse_percentage > 25:
            print("CAUTION: MMSE is still high (>25%)")
        else:
            print("GOOD: MMSE dominance is under control (<25%)")
    
    # Try SHAP (if available and model supports it)
    try:
        print("\n[SHAP ANALYSIS]")
        print("-" * 80)
        
        # Use a sample for SHAP (faster)
        X_sample = X_test.sample(min(50, len(X_test)), random_state=42)
        
        # IMPORTANT: Apply scaler transformation first (model expects scaled data)
        X_sample_scaled = model.named_steps['scaler'].transform(X_sample)
        
        # Get the underlying classifier
        clf = model.named_steps['clf']
        
        # Check if it's XGBoost (has known issues with TreeExplainer for multi-class)
        is_xgboost = clf.__class__.__name__ == 'XGBClassifier'
        
        if is_xgboost:
            # For XGBoost multi-class, use KernelExplainer or Explainer (newer API)
            print("Using KernelExplainer for XGBoost (TreeExplainer has issues with multi-class)...")
            print("Note: This may take a minute...")
            
            # Use a smaller background dataset for KernelExplainer (faster)
            background = X_sample_scaled[:10]  # Use 10 samples as background
            
            # Create a wrapper function for the model - handle multi-class properly
            def model_predict(X):
                # XGBoost predict_proba returns shape (n_samples, n_classes)
                # For KernelExplainer, we need to flatten or handle each class separately
                proba = clf.predict_proba(X)
                # Return probabilities for all classes (KernelExplainer will handle it)
                return proba
            
            # Use KernelExplainer - explain one class at a time for multi-class
            # Explain class 1 (Converted) as it's the most challenging
            explainer = shap.KernelExplainer(lambda X: clf.predict_proba(X)[:, 1], background)
            shap_values_class1 = explainer.shap_values(X_sample_scaled[:10])
            
            # Also explain class 0 and 2 for comparison
            explainer_class0 = shap.KernelExplainer(lambda X: clf.predict_proba(X)[:, 0], background)
            shap_values_class0 = explainer_class0.shap_values(X_sample_scaled[:10])
            
            explainer_class2 = shap.KernelExplainer(lambda X: clf.predict_proba(X)[:, 2], background)
            shap_values_class2 = explainer_class2.shap_values(X_sample_scaled[:10])
            
            # Average absolute SHAP values across all classes
            shap_values = [shap_values_class0, shap_values_class1, shap_values_class2]
            shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            
            # Skip the standard processing below since we already computed shap_importance
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)
            
            print("\nTop 10 Most Important Features (SHAP - averaged across classes):")
            print(shap_df.head(10).to_string(index=False))
            
            # Check MMSE dominance in SHAP
            mmse_shap = shap_df[shap_df['feature'].isin(mmse_features)]['shap_importance'].sum()
            total_shap = shap_importance.sum()
            mmse_shap_pct = (mmse_shap / total_shap) * 100
            
            print(f"\nMMSE-related features SHAP importance: {mmse_shap_pct:.2f}%")
            
            # Compare with feature_importances_
            if hasattr(clf, 'feature_importances_'):
                print("\nComparison: SHAP vs Feature Importances (Top 5)")
                print("-" * 80)
                top5_shap = shap_df.head(5)
                top5_imp = importance_df.head(5)
                print("SHAP Top 5:")
                for idx, row in top5_shap.iterrows():
                    print(f"  {row['feature']}: {row['shap_importance']:.4f}")
                print("\nFeature Importances Top 5:")
                for idx, row in top5_imp.iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Skip the rest of the SHAP processing
            print("\n" + "=" * 80)
            return
        else:
            # For other tree models, use TreeExplainer (faster)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample_scaled)
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # Multi-class: average absolute SHAP values across all classes
            # shap_values is a list of arrays, one per class
            shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            # Single class or binary
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        print("\nTop 10 Most Important Features (SHAP):")
        print(shap_df.head(10).to_string(index=False))
        
        # Check MMSE dominance in SHAP
        mmse_shap = shap_df[shap_df['feature'].isin(mmse_features)]['shap_importance'].sum()
        total_shap = shap_importance.sum()
        mmse_shap_pct = (mmse_shap / total_shap) * 100
        
        print(f"\nMMSE-related features SHAP importance: {mmse_shap_pct:.2f}%")
        
        # Compare with feature_importances_
        if hasattr(clf, 'feature_importances_'):
            print("\nComparison: SHAP vs Feature Importances (Top 5)")
            print("-" * 80)
            top5_shap = shap_df.head(5)
            top5_imp = importance_df.head(5)
            print("SHAP Top 5:")
            for idx, row in top5_shap.iterrows():
                print(f"  {row['feature']}: {row['shap_importance']:.4f}")
            print("\nFeature Importances Top 5:")
            for idx, row in top5_imp.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("This is okay - feature_importances_ above still works.")
        print("Note: XGBoost multi-class models sometimes have compatibility issues with SHAP.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_importance()

