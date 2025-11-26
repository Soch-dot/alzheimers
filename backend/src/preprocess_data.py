"""
Clean the raw Dataset.csv and produce a model-ready CSV.
Run with: python src/preprocess_data.py
"""

from pathlib import Path

import pandas as pd


def preprocess() -> None:
    """Apply simple cleaning steps and save the result."""
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / "data" / "raw" / "Dataset.csv"
    processed_path = base_dir / "data" / "processed" / "clinical_clean.csv"

    print("=" * 60)
    print("Loading raw data from:", raw_path)
    df = pd.read_csv(raw_path)

    # 1. Rename columns to snake_case.
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
    df.rename(columns=rename_map, inplace=True)

    # 2. Fill missing values with median for numeric columns.
    for col in ["ses", "mmse"]:
        if col in df.columns:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled missing values in '{col}' with median={median_value:.2f}")

    # 3. Encode categorical columns.
    df["sex"] = df["sex"].map({"M": 1, "F": 0})
    df["hand"] = df["hand"].map({"R": 1, "L": 0})

    group_mapping = {"Nondemented": 0, "Converted": 1, "Demented": 2}
    df["group"] = df["group"].map(group_mapping)

    # 4. Drop ID columns (not useful for prediction).
    df.drop(columns=["subject_id", "mri_id"], inplace=True)

    print("\nCleaned data preview:")
    print(df.head())

    print("\nTarget counts after encoding (group column):")
    print(df["group"].value_counts())

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print("\nSaved cleaned data to:", processed_path)
    print("=" * 60)


if __name__ == "__main__":
    preprocess()

