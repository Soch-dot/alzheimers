"""
Quick helper to load the trained model and make a sample prediction.
Run with: python src/predict_example.py
"""

from pathlib import Path

import joblib
import pandas as pd


def load_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
    print("Loading model from:", model_path)
    return joblib.load(model_path)


def build_sample_patient() -> pd.DataFrame:
    """Return a single-row DataFrame with feature names matching training data."""
    sample = {
        "visit": 1,
        "mr_delay": 0,
        "sex": 1,  # 1 = male, 0 = female
        "hand": 1,  # 1 = right, 0 = left
        "age": 75,
        "education_years": 12,
        "ses": 2,
        "mmse": 24,
        "cdr": 0.5,
        "etiv": 1700,
        "nwbv": 0.72,
        "asf": 1.1,
    }
    return pd.DataFrame([sample])


def main():
    model = load_model()
    patient_df = build_sample_patient()
    prediction = model.predict(patient_df)[0]
    probabilities = model.predict_proba(patient_df)[0]

    label_map = {0: "Nondemented", 1: "Converted", 2: "Demented"}
    print("\nInput patient data:")
    print(patient_df)
    print("\nPredicted class:", prediction, "-", label_map.get(prediction, "Unknown"))
    print("Class probabilities:")
    for idx, prob in enumerate(probabilities):
        print(f"  {idx} ({label_map[idx]}): {prob:.3f}")


if __name__ == "__main__":
    main()

