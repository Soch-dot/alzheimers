"""
Merge current dataset with Kaggle dataset.
This script:
1. Loads both datasets
2. Standardizes column names and formats
3. Maps Kaggle's binary diagnosis to 3-class format (if possible)
4. Combines compatible features
5. Saves merged dataset

Run with: python src/merge_datasets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_and_prepare_current(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare current dataset for merging."""
    # Create a copy
    df_curr = df.copy()
    
    # Rename columns to standard names
    rename_map = {
        "Subject ID": "subject_id",
        "MRI ID": "mri_id",
        "Group": "group",
        "Visit": "visit",
        "MR Delay": "mr_delay",
        "M/F": "sex",
        "Hand": "hand",
        "Age": "age",
        "EDUC": "education_years",
        "SES": "ses",
        "MMSE": "mmse",
        "CDR": "cdr",
        "eTIV": "etiv",
        "nWBV": "nwbv",
        "ASF": "asf",
    }
    df_curr.rename(columns=rename_map, inplace=True)
    
    # Encode categorical
    df_curr["sex"] = df_curr["sex"].map({"M": 1, "F": 0})
    df_curr["hand"] = df_curr["hand"].map({"R": 1, "L": 0})
    group_mapping = {"Nondemented": 0, "Converted": 1, "Demented": 2}
    df_curr["group"] = df_curr["group"].map(group_mapping)
    
    # Fill missing values
    for col in ["ses", "mmse"]:
        if col in df_curr.columns:
            median_val = df_curr[col].median()
            df_curr[col] = df_curr[col].fillna(median_val)
    
    # Add dataset source marker
    df_curr["dataset_source"] = "current"
    
    return df_curr


def load_and_prepare_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare Kaggle dataset for merging."""
    df_kag = df.copy()
    
    # Standardize column names
    rename_map = {
        "PatientID": "patient_id",
        "Age": "age",
        "Gender": "sex",  # 0=F, 1=M (check if this is correct)
        "EducationLevel": "education_years",
        "MMSE": "mmse",
        "Diagnosis": "diagnosis_raw",
    }
    df_kag.rename(columns={k: v for k, v in rename_map.items() if k in df_kag.columns}, inplace=True)
    
    # Handle Gender: Kaggle uses 0/1, we need to verify mapping
    # Assuming 0=Female, 1=Male (standard)
    if "sex" in df_kag.columns:
        # Keep as is if already 0/1, otherwise map
        pass
    
    # Map diagnosis to 3-class format
    # Kaggle likely has binary (0=No, 1=Yes) or text labels
    if "diagnosis_raw" in df_kag.columns:
        # Check what values exist
        unique_vals = df_kag["diagnosis_raw"].unique()
        print(f"\nKaggle Diagnosis unique values: {unique_vals[:10]}")
        
        # Try to map to 3-class system
        # Strategy: If binary, map 0->0 (Nondemented), 1->2 (Demented)
        # We'll lose "Converted" class from Kaggle, but that's okay
        if df_kag["diagnosis_raw"].dtype in [np.int64, np.float64]:
            # Binary numeric
            df_kag["group"] = df_kag["diagnosis_raw"].map({0: 0, 1: 2})
        else:
            # Text labels - try to map
            diagnosis_lower = df_kag["diagnosis_raw"].astype(str).str.lower()
            df_kag["group"] = 0  # Default to Nondemented
            df_kag.loc[diagnosis_lower.str.contains("dement|alzheim", na=False), "group"] = 2
            df_kag.loc[diagnosis_lower.str.contains("convert|mild", na=False), "group"] = 1
    
    # Fill missing MMSE if any
    if "mmse" in df_kag.columns:
        median_mmse = df_kag["mmse"].median()
        df_kag["mmse"] = df_kag["mmse"].fillna(median_mmse)
    
    # Add missing columns with NaN (for features only in current dataset)
    current_only_cols = ["visit", "mr_delay", "hand", "ses", "cdr", "etiv", "nwbv", "asf"]
    for col in current_only_cols:
        if col not in df_kag.columns:
            df_kag[col] = np.nan
    
    # Add dataset source marker
    df_kag["dataset_source"] = "kaggle"
    
    return df_kag


def merge_datasets() -> None:
    """Main merge function."""
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MERGING DATASETS")
    print("=" * 80)
    
    # Load datasets
    print("\n1. Loading datasets...")
    current_path = raw_dir / "Dataset.csv"
    kaggle_path = raw_dir / "alzheimers_disease_data.csv"
    
    df_current = pd.read_csv(current_path)
    df_kaggle = pd.read_csv(kaggle_path)
    
    print(f"   Current dataset: {len(df_current):,} rows")
    print(f"   Kaggle dataset: {len(df_kaggle):,} rows")
    
    # Prepare datasets
    print("\n2. Preparing datasets...")
    df_curr_clean = load_and_prepare_current(df_current)
    df_kag_clean = load_and_prepare_kaggle(df_kaggle)
    
    # Select common columns for merging
    common_cols = [
        "age",
        "sex",
        "education_years",
        "mmse",
        "group",
        "dataset_source",
    ]
    
    # Add optional columns if they exist
    optional_cols = ["visit", "mr_delay", "hand", "ses", "cdr", "etiv", "nwbv", "asf"]
    for col in optional_cols:
        if col in df_curr_clean.columns or col in df_kag_clean.columns:
            if col not in common_cols:
                common_cols.append(col)
    
    # Select only common columns
    df_curr_final = df_curr_clean[[c for c in common_cols if c in df_curr_clean.columns]].copy()
    df_kag_final = df_kag_clean[[c for c in common_cols if c in df_kag_clean.columns]].copy()
    
    # Ensure both have same columns (fill missing with NaN)
    all_cols = set(df_curr_final.columns) | set(df_kag_final.columns)
    for col in all_cols:
        if col not in df_curr_final.columns:
            df_curr_final[col] = np.nan
        if col not in df_kag_final.columns:
            df_kag_final[col] = np.nan
    
    # Reorder columns consistently
    col_order = ["group"] + [c for c in common_cols if c != "group"]
    df_curr_final = df_curr_final[col_order]
    df_kag_final = df_kag_final[col_order]
    
    # Merge
    print("\n3. Merging datasets...")
    df_merged = pd.concat([df_curr_final, df_kag_final], ignore_index=True)
    
    print(f"   Merged dataset: {len(df_merged):,} rows, {len(df_merged.columns)} columns")
    
    # Show distribution
    print("\n4. Target distribution:")
    print(df_merged["group"].value_counts().sort_index())
    
    print("\n5. Dataset source distribution:")
    print(df_merged["dataset_source"].value_counts())
    
    # Save
    output_path = processed_dir / "clinical_merged.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved merged dataset to: {output_path}")
    
    # Also save a version without dataset_source (for training)
    df_merged_train = df_merged.drop(columns=["dataset_source"])
    output_path_train = processed_dir / "clinical_clean.csv"
    df_merged_train.to_csv(output_path_train, index=False)
    print(f"[SUCCESS] Saved training dataset (without source) to: {output_path_train}")
    
    print("\n" + "=" * 80)
    print("MERGE COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the merged dataset")
    print("2. Run: python src/train_models.py")
    print("3. Compare metrics with previous model")


if __name__ == "__main__":
    merge_datasets()

