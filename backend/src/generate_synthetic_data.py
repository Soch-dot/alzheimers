"""
Generate synthetic adversarial cases to help model learn rule logic internally.
This reduces dependency on post-processing rules.
Run with: python src/generate_synthetic_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_cases() -> pd.DataFrame:
    """
    Generate synthetic adversarial cases that reflect rule logic.
    These cases will help the model internalize the rules.
    """
    synthetic_cases = []
    
    # Pattern 1: Young + CDR=0.5 + High MMSE → Converted
    # This is the "early conversion" pattern
    for _ in range(20):
        age = np.random.randint(55, 65)
        cdr = 0.5
        mmse = np.random.uniform(25, 29)
        education = np.random.randint(12, 18)
        sex = np.random.choice([0, 1])
        ses = np.random.uniform(2.0, 4.0)
        
        synthetic_cases.append({
            "age": age,
            "sex": sex,
            "education_years": education,
            "mmse": mmse,
            "cdr": cdr,
            "ses": ses,
            "group": 1  # Converted
        })
    
    # Pattern 2: Low Education + Low MMSE + CDR=0 → Nondemented
    # This is the "education bias" pattern
    for _ in range(20):
        age = np.random.randint(70, 85)
        cdr = 0.0
        mmse = np.random.uniform(18, 22)
        education = np.random.randint(5, 10)
        sex = np.random.choice([0, 1])
        ses = np.random.uniform(1.0, 3.0)
        
        synthetic_cases.append({
            "age": age,
            "sex": sex,
            "education_years": education,
            "mmse": mmse,
            "cdr": cdr,
            "ses": ses,
            "group": 0  # Nondemented
        })
    
    # Pattern 3: Very Old + Low nWBV + High MMSE → Demented (if we had nWBV)
    # For now, use: Very Old + CDR=0.5 + Moderate MMSE → Demented
    for _ in range(15):
        age = np.random.randint(82, 90)
        cdr = 0.5
        mmse = np.random.uniform(20, 24)
        education = np.random.randint(8, 14)
        sex = np.random.choice([0, 1])
        ses = np.random.uniform(1.5, 3.5)
        
        synthetic_cases.append({
            "age": age,
            "sex": sex,
            "education_years": education,
            "mmse": mmse,
            "cdr": cdr,
            "ses": ses,
            "group": 2  # Demented (progression from Converted)
        })
    
    # Pattern 4: CDR=0 + Any MMSE ≥ 20 → Nondemented (general rule)
    for _ in range(15):
        age = np.random.randint(65, 85)
        cdr = 0.0
        mmse = np.random.uniform(20, 30)
        education = np.random.randint(8, 16)
        sex = np.random.choice([0, 1])
        ses = np.random.uniform(1.5, 4.0)
        
        synthetic_cases.append({
            "age": age,
            "sex": sex,
            "education_years": education,
            "mmse": mmse,
            "cdr": cdr,
            "ses": ses,
            "group": 0  # Nondemented
        })
    
    # Pattern 5: CDR ≥ 1.0 → Demented (strong signal)
    for _ in range(15):
        age = np.random.randint(70, 88)
        cdr = np.random.choice([1.0, 1.5, 2.0])
        mmse = np.random.uniform(10, 25)  # Wide range - CDR is the signal
        education = np.random.randint(6, 16)
        sex = np.random.choice([0, 1])
        ses = np.random.uniform(1.0, 4.0)
        
        synthetic_cases.append({
            "age": age,
            "sex": sex,
            "education_years": education,
            "mmse": mmse,
            "cdr": cdr,
            "ses": ses,
            "group": 2  # Demented
        })
    
    # Pattern 6: Borderline Converted cases (CDR=0.5, moderate MMSE)
    # These are the hardest to classify
    for _ in range(20):
        age = np.random.randint(65, 80)
        cdr = 0.5
        mmse = np.random.uniform(22, 26)
        education = np.random.randint(10, 16)
        sex = np.random.choice([0, 1])
        ses = np.random.uniform(2.0, 4.0)
        
        # 60% Converted, 40% Demented (reflecting progression)
        group = np.random.choice([1, 2], p=[0.6, 0.4])
        
        synthetic_cases.append({
            "age": age,
            "sex": sex,
            "education_years": education,
            "mmse": mmse,
            "cdr": cdr,
            "ses": ses,
            "group": group
        })
    
    df_synthetic = pd.DataFrame(synthetic_cases)
    
    print(f"Generated {len(df_synthetic)} synthetic cases")
    print(f"\nClass distribution:")
    print(df_synthetic['group'].value_counts().sort_index())
    print("\nClass breakdown:")
    class_map = {0: "Nondemented", 1: "Converted", 2: "Demented"}
    for group_id, count in df_synthetic['group'].value_counts().sort_index().items():
        print(f"  {class_map[group_id]}: {count}")
    
    return df_synthetic


def save_synthetic_data():
    """Generate and save synthetic data."""
    base_dir = Path(__file__).resolve().parents[1]
    output_path = base_dir / "data" / "synthetic" / "synthetic_adversarial_cases.csv"
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_synthetic = generate_synthetic_cases()
    df_synthetic.to_csv(output_path, index=False)
    
    print(f"\nSaved synthetic data to: {output_path}")
    print(f"\nTo use this data:")
    print("1. Review the synthetic cases")
    print("2. Merge with original dataset in train_models.py")
    print("3. Retrain the model")
    print("\nThis will help the model learn rule logic internally!")


if __name__ == "__main__":
    save_synthetic_data()

