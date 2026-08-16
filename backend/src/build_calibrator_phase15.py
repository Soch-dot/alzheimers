"""
Phase 15 -- build the production sigmoid calibrator (display-only).

Fits sigmoid (Platt) calibration on the FULL 112-training-subject OOF raw
probabilities, using the exact Phase 14-approved procedure:
  - StratifiedGroupKFold(5, shuffle=True, random_state=42) grouped by Subject ID
  - latest available visit by MR Delay per subject
  - exactly one OOF probability per dev subject from a model that did not train
    on that subject
  - underlying LogisticRegression never retrained

The calibrator maps raw binary screening probability -> calibrated display
probability. The raw 0.40 screening decision is NOT changed. The outer 38
subjects are never used to fit anything.

Outputs (production):
  backend/models/production/sigmoid_calibrator_v1.pkl
  backend/models/production/sigmoid_calibrator_v1.json
"""

from pathlib import Path
import hashlib
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.calibration import _SigmoidCalibration
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
import joblib

import sklearn

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"
PHASE14_OOF = Path(__file__).resolve().parents[1] / "models" / "experiments" / "phase14_recalibration" / "oof_raw_predictions.csv"
PROD_DIR = Path(__file__).resolve().parents[1] / "models" / "production"
ARTIFACT = PROD_DIR / "sigmoid_calibrator_v1.pkl"
METADATA = PROD_DIR / "sigmoid_calibrator_v1.json"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
SUBJECT_COL = "Subject ID"
LABEL_COL = "group"
BINARY_COL = "binary_target"
MR_DELAY_COL = "mr_delay"
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]

RAW_THRESHOLD = 0.40
N_FOLDS = 5
CV_RANDOM_STATE = 42
NP_SEED = 42
MAPPED_DISPLAY_BOUNDARY = 0.4082
MODEL_VERSION = "binary_lr_latest_visit_v1"
CALIBRATOR_VERSION = "sigmoid_calibrator_v1"

WINNER_HP = {"clf__C": 10.0, "clf__class_weight": "balanced"}


def md5_hex(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Age": "age", "M/F": "sex", "EDUC": "education_years",
        "MMSE": "mmse", "SES": "ses", "Group": LABEL_COL,
        "Visit": "raw_visit", "MR Delay": MR_DELAY_COL,
    })
    df["sex"] = df["sex"].map(SEX_MAP)
    df[LABEL_COL] = df[LABEL_COL].map(GROUP_MAP)
    df[BINARY_COL] = (df[LABEL_COL] != 0).astype(int)
    df = df.sort_values([SUBJECT_COL, MR_DELAY_COL]).reset_index(drop=True)
    return df


def latest_frame(subset):
    m = subset[subset[MR_DELAY_COL] == subset.groupby(SUBJECT_COL)[MR_DELAY_COL].transform("max")].copy()
    return m.reset_index(drop=True)


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
    ]).set_params(**WINNER_HP)


def main():
    np.random.seed(NP_SEED)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_subjects = sorted(split["train_subjects"])
    test_subjects = sorted(split["test_subjects"])
    assert len(set(train_subjects) & set(test_subjects)) == 0
    assert len(train_subjects) == 112 and len(test_subjects) == 38

    df = load_data()
    df_train = df[df[SUBJECT_COL].isin(train_subjects)]
    tr_latest = latest_frame(df_train)
    assert len(tr_latest) == 112

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    dev_meta = pd.DataFrame({"subject_id": tr_latest[SUBJECT_COL].to_numpy(),
                             "label": tr_latest[BINARY_COL].to_numpy()})
    folds = list(cv.split(dev_meta, dev_meta["label"], groups=dev_meta["subject_id"]))

    oof_parts = []
    for tr_idx, va_idx in folds:
        tr_subj = set(dev_meta.iloc[tr_idx]["subject_id"])
        va_subj = set(dev_meta.iloc[va_idx]["subject_id"])
        assert not (tr_subj & va_subj)
        tr_rows = tr_latest[tr_latest[SUBJECT_COL].isin(tr_subj)]
        va_rows = tr_latest[tr_latest[SUBJECT_COL].isin(va_subj)]
        est = make_pipeline()
        est.fit(tr_rows[FEATURES], tr_rows[BINARY_COL])
        va_rows = va_rows.copy()
        va_rows["oof_raw_prob"] = est.predict_proba(va_rows[FEATURES])[:, 1]
        oof_parts.append(va_rows[[SUBJECT_COL, BINARY_COL, "oof_raw_prob"]])
    oof = pd.concat(oof_parts, ignore_index=True)
    assert len(oof) == 112 and oof[SUBJECT_COL].nunique() == 112
    assert not set(oof[SUBJECT_COL]) & set(test_subjects)

    # Verify OOF matches the Phase 14 artifact exactly
    if PHASE14_OOF.exists():
        ref = pd.read_csv(PHASE14_OOF)
        ref_map = dict(zip(ref[SUBJECT_COL], ref["oof_raw_prob"]))
        for _, r in oof.iterrows():
            assert abs(float(r["oof_raw_prob"]) - float(ref_map[r[SUBJECT_COL]])) < 1e-12, \
                r[SUBJECT_COL]
        print("OOF verified: regenerated probabilities match Phase 14 oof_raw_predictions.csv exactly.")

    # Fit the final sigmoid calibrator on the full 112-subject OOF
    calibrator = _SigmoidCalibration()
    calibrator.fit(oof["oof_raw_prob"].to_numpy().reshape(-1, 1), oof[BINARY_COL].to_numpy())

    # Verify mapped display boundary at raw 0.40
    mapped = float(calibrator.predict(np.array([[RAW_THRESHOLD]]))[0])
    print("Raw 0.40 -> calibrated display boundary: %.6f (expected ~0.4082)" % mapped)
    assert abs(mapped - MAPPED_DISPLAY_BOUNDARY) < 0.01

    joblib.dump(calibrator, ARTIFACT)
    checksum = md5_hex(ARTIFACT)
    print("Calibrator artifact checksum:", checksum)

    metadata = {
        "calibrator_version": CALIBRATOR_VERSION,
        "calibration_method": "sigmoid",
        "fitted_on_subject_count": 112,
        "outer_test_subject_count": 38,
        "target_definition": "0 = Nondemented, 1 = Converted OR Demented",
        "raw_threshold": RAW_THRESHOLD,
        "mapped_display_boundary": round(mapped, 4),
        "model_version": MODEL_VERSION,
        "feature_order": FEATURES,
        "source_oof_artifact": "backend/models/experiments/phase14_recalibration/oof_raw_predictions.csv",
        "oof_procedure": ("StratifiedGroupKFold(5, shuffle=True, random_state=42) grouped by "
                          "Subject ID; latest available visit by MR Delay; one OOF prediction "
                          "per dev subject; model never trained on the predicted subject"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "artifact_checksum": checksum,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "semantics": ("Display-only calibration. Screening decision remains based on raw "
                      "probability threshold 0.40 (raw >= 0.40 -> positive). The calibrated "
                      "value is a model-estimated screening probability for display; it is "
                      "NOT clinically validated and is NOT a diagnosis."),
        "limitations": ["n=112 calibration subjects; n=38 outer test",
                        "not clinically validated", "not a diagnosis",
                        "calibration evaluated as display-only; decision uses raw probability"],
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Metadata written to", METADATA)


if __name__ == "__main__":
    main()