"""
Explore the Kaggle dataset and compare with current dataset.
Run with: python src/explore_kaggle_dataset.py
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    """Compare both datasets to understand structure."""
    base_dir = Path(__file__).resolve().parents[1]
    
    # Load both datasets
    current_path = base_dir / "data" / "raw" / "Dataset.csv"
    kaggle_path = base_dir / "data" / "raw" / "alzheimers_disease_data.csv"
    
    print("=" * 80)
    print("DATASET COMPARISON")
    print("=" * 80)
    
    # Current dataset
    print("\n[CURRENT DATASET] Dataset.csv:")
    print("-" * 80)
    df_current = pd.read_csv(current_path)
    print(f"Rows: {len(df_current):,}")
    print(f"Columns: {len(df_current.columns)}")
    print("\nColumns:")
    for i, col in enumerate(df_current.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\nTarget (Group) distribution:")
    print(df_current["Group"].value_counts())
    
    # Kaggle dataset
    print("\n\n[KAGGLE DATASET] alzheimers_disease_data.csv:")
    print("-" * 80)
    df_kaggle = pd.read_csv(kaggle_path)
    print(f"Rows: {len(df_kaggle):,}")
    print(f"Columns: {len(df_kaggle.columns)}")
    print("\nFirst 10 columns:")
    for i, col in enumerate(df_kaggle.columns[:10], 1):
        print(f"  {i:2d}. {col}")
    print(f"  ... and {len(df_kaggle.columns) - 10} more")
    
    if "Diagnosis" in df_kaggle.columns:
        print("\nTarget (Diagnosis) distribution:")
        print(df_kaggle["Diagnosis"].value_counts())
        print("\nDiagnosis unique values:")
        print(df_kaggle["Diagnosis"].unique()[:10])
    
    # Find common/similar features
    print("\n\n[FEATURE MAPPING]:")
    print("-" * 80)
    print("Common features (exact match):")
    common = set(df_current.columns) & set(df_kaggle.columns)
    print(f"  {common if common else 'None'}")
    
    print("\nSimilar features (need mapping):")
    mappings = {
        "Age": "Age",
        "MMSE": "MMSE",
        "EDUC": "EducationLevel",
        "M/F": "Gender",
    }
    for curr, kag in mappings.items():
        if curr in df_current.columns and kag in df_kaggle.columns:
            print(f"  [OK] {curr} (current) -> {kag} (kaggle)")
        elif curr in df_current.columns:
            print(f"  [X] {curr} (current) -> {kag} (kaggle) [MISSING in Kaggle]")
        elif kag in df_kaggle.columns:
            print(f"  [X] {curr} (current) -> {kag} (kaggle) [MISSING in Current]")
    
    # Check data types
    print("\n\n[DATA TYPES]:")
    print("-" * 80)
    print("Current dataset - key columns:")
    key_cols = ["Age", "MMSE", "CDR", "EDUC", "Group"]
    for col in key_cols:
        if col in df_current.columns:
            print(f"  {col}: {df_current[col].dtype}")
    
    print("\nKaggle dataset - key columns:")
    key_cols_kag = ["Age", "MMSE", "EducationLevel", "Gender", "Diagnosis"]
    for col in key_cols_kag:
        if col in df_kaggle.columns:
            print(f"  {col}: {df_kaggle[col].dtype}")
            if col in ["Age", "MMSE"]:
                print(f"    Range: {df_kaggle[col].min():.2f} - {df_kaggle[col].max():.2f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

