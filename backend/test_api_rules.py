"""
Test the API with tricky cases to verify post-processing rules work.
Run with: python test_api_rules.py
Make sure API is running: uvicorn src.api:app --reload
"""

import json
import requests
from pathlib import Path

base_url = "http://127.0.0.1:8000/predict"
test_cases_path = Path(__file__).parent / "test_cases.json"

print("=" * 80)
print("TESTING API WITH POST-PROCESSING RULES")
print("=" * 80)
print("\nMake sure your API server is running!")
print("If not, run: uvicorn src.api:app --reload\n")

# Load test cases
with open(test_cases_path, "r") as f:
    test_data = json.load(f)

# Test the tricky cases (7, 9, 10, 11)
tricky_cases = [7, 9, 10, 11]  # 0-indexed, so -1

for idx in tricky_cases:
    test_case = test_data["test_cases"][idx - 1]  # Convert to 0-indexed
    print(f"\n[TEST CASE {idx}] {test_case['name']}")
    print(f"Description: {test_case['description']}")
    print(f"Expected: {test_case['expected']}")
    print("-" * 80)
    
    try:
        response = requests.post(base_url, json=test_case["data"], timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted: {result['predicted_class']}")
            print(f"Detection %: {result['detection_percentage']}%")
            print("Probabilities:")
            for label, prob in result['probabilities'].items():
                marker = ">>>" if label == result['predicted_class'] else "   "
                print(f"  {marker} {label:15s}: {prob*100:5.2f}%")
            
            # Check if correct
            expected_lower = test_case['expected'].lower()
            predicted_lower = result['predicted_class'].lower()
            
            if expected_lower in predicted_lower or predicted_lower in expected_lower:
                print("  [CORRECT] ✓")
            elif "or" in expected_lower:
                if "converted" in expected_lower and result['class_index'] in [1, 2]:
                    print("  [CORRECT] ✓")
                elif "demented" in expected_lower and result['class_index'] == 2:
                    print("  [CORRECT] ✓")
                else:
                    print(f"  [CHECK] Expected: {test_case['expected']}")
            else:
                print(f"  [CHECK] Expected: {test_case['expected']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API. Is it running?")
        print("Start it with: uvicorn src.api:app --reload")
        break
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)

