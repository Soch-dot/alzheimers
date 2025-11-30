"""
Test the trained model with hard test cases.
Run with: python test_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Load model
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "models" / "best_model.pkl"
test_cases_path = base_dir / "test_cases.json"

print("=" * 80)
print("TESTING MODEL WITH HARD CASES")
print("=" * 80)

# Load model
print(f"\nLoading model from: {model_path}")
model = joblib.load(model_path)

# Load test cases
print(f"Loading test cases from: {test_cases_path}")
with open(test_cases_path, "r") as f:
    test_data = json.load(f)

label_map = {0: "Nondemented", 1: "Converted", 2: "Demented"}

print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)

for i, test_case in enumerate(test_data["test_cases"], 1):
    print(f"\n[TEST CASE {i}] {test_case['name']}")
    print(f"Description: {test_case['description']}")
    print(f"Expected: {test_case['expected']}")
    print("-" * 80)
    
    # Extract clinical features
    patient = test_case["data"]
    clinical_data = {
        "age": patient["age"],
        "sex": patient["sex"],
        "education_years": patient["education_years"],
        "mmse": patient["mmse"],
        "cdr": patient["cdr"],
        "ses": patient["ses"],
    }
    
    # Feature engineering (same as training)
    df_temp = pd.DataFrame([clinical_data])
    
    # Age groups
    age_group = pd.cut([patient["age"]], bins=[0, 70, 75, 80, 85, 100], labels=[0, 1, 2, 3, 4])[0]
    if pd.isna(age_group):
        age_group = 2
    clinical_data["age_group"] = int(age_group)
    
    # MMSE categories
    mmse_cat = pd.cut([patient["mmse"]], bins=[0, 17, 23, 30], labels=[0, 1, 2])[0]
    if pd.isna(mmse_cat):
        mmse_cat = 1
    clinical_data["mmse_category"] = int(mmse_cat)
    
    # Create DataFrame
    df = pd.DataFrame([clinical_data])
    
    # Predict
    pred_class = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    
    # Display results
    predicted_label = label_map[pred_class]
    probs = {
        label_map[i]: float(probabilities[i]) * 100
        for i in range(len(probabilities))
    }
    
    print(f"Input Features:")
    print(f"  Age: {patient['age']}, Sex: {'Male' if patient['sex'] == 1 else 'Female'}")
    print(f"  Education: {patient['education_years']} years")
    print(f"  MMSE: {patient['mmse']:.1f} (Category: {clinical_data['mmse_category']})")
    print(f"  CDR: {patient['cdr']:.1f}")
    print(f"  SES: {patient['ses']:.1f}")
    
    print(f"\nPrediction:")
    print(f"  Predicted Class: {predicted_label} (Index: {pred_class})")
    print(f"  Probabilities:")
    for label, prob in probs.items():
        marker = ">>>" if label == predicted_label else "   "
        print(f"    {marker} {label:15s}: {prob:5.2f}%")
    
    # Check if prediction matches expectation
    expected_lower = test_case["expected"].lower()
    predicted_lower = predicted_label.lower()
    
    if expected_lower in predicted_lower or predicted_lower in expected_lower:
        print(f"  [CORRECT] Prediction matches expected outcome!")
    elif "or" in expected_lower:
        # Handle "Converted or Demented" cases
        if "converted" in expected_lower and pred_class in [1, 2]:
            print(f"  [CORRECT] Prediction is in expected range!")
        elif "demented" in expected_lower and pred_class == 2:
            print(f"  [CORRECT] Prediction matches expected outcome!")
        else:
            print(f"  [CHECK] Prediction may differ from expected")
    else:
        print(f"  [CHECK] Expected: {test_case['expected']}, Got: {predicted_label}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)

