"""
EXPERIMENTAL calibration step — do not use in production, do not commit yet.

Fits a sigmoid (Platt) calibrator on TOP of the existing trained Random Forest
pipeline (backend/models/best_model.pkl) WITHOUT retraining or modifying it.

Calibration set construction
----------------------------
The existing training script (backend/src/train_clean_clinical_model.py) splits
Dataset.csv with:

    train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

where X has the fixed column order  [age, sex, education_years, mmse, ses].
We reproduce the IDENTICAL split call (same data loader, same arguments), so:

    - original training portion (80%) stays training data
    - original held-out portion (20%) becomes the EXPERIMENTAL calibration set

Because the held-out portion is reused for calibration (not reserved as a final
unbiased test), every metric printed for it is an EXPERIMENTAL/IN-SAMPLE
calibration diagnostic — it is NOT a final unbiased test result.

Sklearn API
-----------
Uses the current supported API for calibrating an already-fitted estimator:

    from sklearn.frozen import FrozenEstimator
    from sklearn.calibration import CalibratedClassifierCV

    CalibratedClassifierCV(estimator=FrozenEstimator(model), method="sigmoid")

With a FrozenEstimator, sklearn skips cross-validation and fits ONLY the
sigmoid calibrator on the data passed to .fit() — the Random Forest itself is
never refit (verified below by checking the RF's raw predict_proba is identical
before and after fitting the calibrator).
"""

import sys
import hashlib
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
MODELS = BASE / "models"
CAL_DIR = MODELS / "calibration"

sys.path.insert(0, str(SRC))
from train_clean_clinical_model import load_data  # noqa: E402  (shared, exact data prep)

ORIGINAL_MODEL = MODELS / "best_model.pkl"
CALIBRATED_MODEL = MODELS / "best_model_calibrated.pkl"
HASH_MARKER = CAL_DIR / "original_model_hash.txt"

FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
CLASS_NAMES = ["Nondemented", "Converted", "Demented"]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest().upper()


def main() -> None:
    CAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Confirm original model + record untouched hash -------------------
    print("=" * 72)
    print("STEP 1 — LOAD EXISTING MODEL (no retraining)")
    print("=" * 72)
    orig_hash_before = md5(ORIGINAL_MODEL)
    print(f"best_model.pkl   md5  = {orig_hash_before}")
    print(f"best_model.pkl   size = {ORIGINAL_MODEL.stat().st_size} bytes")

    model = joblib.load(ORIGINAL_MODEL)
    print(f"model type      = {type(model).__name__} ({type(model).__module__})")
    print(f"pipeline steps  = {[name for name, _ in model.steps]}")
    print(f"estimator       = {type(model.named_steps['clf']).__name__}")
    print(f"classes_        = {list(model.classes_)}  ->  {[CLASS_NAMES[c] for c in model.classes_]}")
    print(f"feature order   = {list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else '(via scaler/pipeline input)'}")
    print(f"has predict_proba = {hasattr(model, 'predict_proba')}")

    # --- 2. Reproduce the original split; held-out = calibration set ---------
    print("\n" + "=" * 72)
    print("STEP 2 — CALIBRATION SET (original held-out 20%)")
    print("=" * 72)
    df = load_data()
    X = df.drop("group", axis=1)
    y = df["group"]

    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"split used      = train_test_split(test_size=0.2, random_state=42, stratify=y)  [identical to training script]")
    print(f"training portion  = {X_train.shape[0]} samples (untouched, stays training)")
    print(f"CALIBRATION SET   = {X_cal.shape[0]} samples (EXPERIMENTAL — original held-out)")
    print(f"calibration class counts = {dict(y_cal.value_counts().sort_index())}  ->  "
          f"{[f'{CLASS_NAMES[c]}:{n}' for c, n in y_cal.value_counts().sort_index().items()]}")
    print(f"calibration features = {list(X_cal.columns)}")

    # --- 3. Fit calibrator on calibration set ONLY (FrozenEstimator) ---------
    print("\n" + "=" * 72)
    print("STEP 3 — FIT CALIBRATOR (sigmoid) ON CALIBRATION SET ONLY")
    print("=" * 72)
    raw_before = model.predict_proba(X_cal)  # sanity probe

    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="sigmoid",
    )
    calibrator.fit(X_cal, y_cal)

    raw_after = model.predict_proba(X_cal)
    unchanged = np.allclose(raw_before, raw_after)
    print(f"method                          = sigmoid (Platt scaling, one-vs-rest)")
    print(f"sklearn API                     = CalibratedClassifierCV(estimator=FrozenEstimator(model), method='sigmoid')")
    print(f"underlying RandomForest refit?  = {'YES (UNEXPECTED)' if not unchanged else 'NO — raw predict_proba identical before/after fit'}")
    print(f"calibrator classes_             = {list(calibrator.classes_)}")

    # --- 4. Save the NEW calibrated model (separate file) --------------------
    print("\n" + "=" * 72)
    print("STEP 4 — SAVE backend/models/best_model_calibrated.pkl")
    print("=" * 72)
    joblib.dump(calibrator, CALIBRATED_MODEL)
    cal_hash = md5(CALIBRATED_MODEL)
    print(f"calibrated model saved  = {CALIBRATED_MODEL}")
    print(f"calibrated model md5    = {cal_hash}")
    print(f"separate file?          = {'YES' if CALIBRATED_MODEL.resolve() != ORIGINAL_MODEL.resolve() else 'NO'}")

    orig_hash_after = md5(ORIGINAL_MODEL)
    print(f"original model md5 after= {orig_hash_after}  (unchanged: {orig_hash_after == orig_hash_before})")

    # Record the original hash marker for the test script's safety check.
    import time
    HASH_MARKER.write_text(
        f"original=backend/models/best_model.pkl\n"
        f"md5={orig_hash_after}\n"
        f"mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ORIGINAL_MODEL.stat().st_mtime))}\n",
        encoding="utf-8",
    )
    print(f"original-model hash marker written to {HASH_MARKER}")

    print("\nCalibration step complete. Run backend/scripts/test_calibration.py for the comparison.")


if __name__ == "__main__":
    main()
