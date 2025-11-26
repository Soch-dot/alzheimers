"""
Quick script to explore the Alzheimer's clinical dataset.

Run with: python src/explore_data.py
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    """Load the CSV, show basic structure, and summarize key columns."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset.csv"

    print("=" * 60)
    print("Loading dataset from:", data_path)
    df = pd.read_csv(data_path)

    print("\nFirst 5 rows (df.head()):")
    print(df.head())

    print("\nBasic info (df.info()):")
    df.info()

    print("\nSummary statistics (df.describe()):")
    print(df.describe())

    print("\nMissing values per column (df.isna().sum()):")
    print(df.isna().sum())

    # Show distribution of the target column (Group) so we know class balance.
    if "Group" in df.columns:
        print("\nDiagnosis counts (Group column):")
        print(df["Group"].value_counts())

    print("=" * 60)


if __name__ == "__main__":
    main()

