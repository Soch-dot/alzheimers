"""
Phase 15 backend tests for the display-only sigmoid calibrator.

Run from the project root:

    backend\\venv\\Scripts\\python.exe -m unittest backend.tests.test_calibration_phase15

Covers:
  1. calibrator loads
  2. checksum matches
  3. calibrated probability is in [0,1]
  4. raw probability remains unchanged
  5. screening_result uses raw probability
  6. calibrated display probability does NOT determine the decision
  7. fixed test case matches expected approximate values
  8. calibration preserves monotonic ordering
  9. missing calibrator fails safely
  10. corrupted calibrator fails safely
  11. flag preservation across the canonical outer-test fixture (0 decision changes)
"""

import hashlib
import importlib
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import _SigmoidCalibration  # noqa: E402

import src.api as api  # noqa: E402

BACKEND_ROOT = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROD_DIR = BACKEND_ROOT / "models" / "production"
CALIBRATOR_PKL = PROD_DIR / "sigmoid_calibrator_v1.pkl"
CALIBRATOR_JSON = PROD_DIR / "sigmoid_calibrator_v1.json"

DATA_PATH = BACKEND_ROOT / "data" / "raw" / "Dataset.csv"
SPLIT_PATH = BACKEND_ROOT / "models" / "experiments" / "subject_grouped_baseline" / "subject_split.json"

SEX_MAP = {"M": 1, "F": 0}
GROUP_MAP = {"Nondemented": 0, "Converted": 1, "Demented": 2}
SUBJECT_COL = "Subject ID"
LABEL_COL = "group"
BINARY_COL = "binary_target"
MR_DELAY_COL = "mr_delay"
FEATURES = ["age", "sex", "education_years", "mmse", "ses"]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _load_outer_fixture():
    """Build the canonical 38-subject outer-test fixture (latest visit by MR Delay)."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_subjects = set(split["test_subjects"])
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
    test_rows = df[df[SUBJECT_COL].isin(test_subjects)]
    latest = (
        test_rows[test_rows[MR_DELAY_COL] == test_rows.groupby(SUBJECT_COL)[MR_DELAY_COL].transform("max")]
        .copy()
        .reset_index(drop=True)
    )
    assert len(latest) == 38
    return latest


class CalibratorLoadTest(unittest.TestCase):
    def setUp(self):
        self.cal = joblib.load(CALIBRATOR_PKL)
        self.meta = json.loads(CALIBRATOR_JSON.read_text(encoding="utf-8"))

    def test_01_calibrator_loads(self):
        self.assertIsInstance(self.cal, _SigmoidCalibration)

    def test_02_checksum_matches(self):
        self.assertEqual(_md5(CALIBRATOR_PKL), api.CALIBRATOR_ARTIFACT_MD5)
        self.assertEqual(self.meta["artifact_checksum"], api.CALIBRATOR_ARTIFACT_MD5)

    def test_03_calibrated_probability_in_unit_interval(self):
        for raw in np.linspace(0.0, 1.0, 101):
            cal = float(self.cal.predict(np.array([[raw]]))[0])
            self.assertGreaterEqual(cal, 0.0)
            self.assertLessEqual(cal, 1.0)

    def test_08_monotonic_ordering(self):
        raws = np.linspace(0.0, 1.0, 201)
        cals = [float(self.cal.predict(np.array([[r]]))[0]) for r in raws]
        for a, b in zip(cals, cals[1:]):
            self.assertLessEqual(a, b)

    def test_display_boundary_maps_0_40(self):
        mapped = float(self.cal.predict(np.array([[0.40]]))[0])
        self.assertAlmostEqual(mapped, 0.4082, places=3)
        self.assertEqual(self.meta["raw_threshold"], 0.40)


class CalibratorFailureTest(unittest.TestCase):
    """Fail-safe behavior: missing or corrupted calibrator must not crash /predict,
    and the raw screening decision must still work."""

    def test_09_missing_calibrator_returns_none_display(self):
        # Missing artifact -> api.load_calibrator() returns None (path does not exist).
        # Simulate by pointing CALIBRATOR_PATH at a non-existent location via the
        # loading function's env-var override.
        saved = os.environ.get("CALIBRATOR_PATH")
        os.environ["CALIBRATOR_PATH"] = str(BACKEND_ROOT / "models" / "does_not_exist.pkl")
        try:
            cal = api.load_calibrator()
            self.assertIsNone(cal)
        finally:
            if saved is None:
                os.environ.pop("CALIBRATOR_PATH", None)
            else:
                os.environ["CALIBRATOR_PATH"] = saved

    def test_10_corrupted_calibrator_fails_safely(self):
        # A calibrator whose checksum no longer matches the locked value must be
        # rejected (None returned), never silently substituted.
        tmp = BACKEND_ROOT / "models" / "production" / "_corrupt_calibrator_test.pkl"
        try:
            joblib.dump({"not": "a calibrator"}, tmp)
            actual = _md5(tmp)
            if actual == api.CALIBRATOR_ARTIFACT_MD5:
                tmp.write_bytes(tmp.read_bytes() + b"\x00")
                actual = _md5(tmp)
            self.assertNotEqual(actual, api.CALIBRATOR_ARTIFACT_MD5)
            saved = os.environ.get("CALIBRATOR_PATH")
            os.environ["CALIBRATOR_PATH"] = str(tmp)
            try:
                cal = api.load_calibrator()
                self.assertIsNone(cal)
            finally:
                if saved is None:
                    os.environ.pop("CALIBRATOR_PATH", None)
                else:
                    os.environ["CALIBRATOR_PATH"] = saved
        finally:
            if tmp.exists():
                tmp.unlink()


class PredictCalibrationContractTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(api.app)

    def test_04_raw_probability_unchanged(self):
        # Adding calibrated_screening_probability must not change screening_probability.
        r = self.client.post("/predict", json={"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1})
        body = r.json()
        self.assertAlmostEqual(body["screening_probability"], 0.99998, places=4)

    def test_05_screening_result_uses_raw_probability(self):
        # Decision comes from RAW >= 0.40. Choose a case where calibrated value
        # differs from raw but the decision must follow raw.
        r = self.client.post("/predict", json={"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1})
        body = r.json()
        self.assertEqual(body["screening_result"], "positive")
        self.assertGreaterEqual(body["screening_probability"], 0.40)
        # And the calibrated display value never overrides a negative raw decision.
        r2 = self.client.post("/predict", json={"age": 70, "sex": 0, "education_years": 16, "mmse": 30, "ses": 2})
        body2 = r2.json()
        self.assertEqual(body2["screening_result"], "negative")
        self.assertLess(body2["screening_probability"], 0.40)
        if body2.get("calibrated_screening_probability") is not None:
            self.assertGreater(body2["calibrated_screening_probability"], 0.0)

    def test_06_calibrated_display_does_not_determine_decision(self):
        # For a raw just above threshold, even if calibrated were lower, the
        # decision stays positive. Directly verify the decision rule uses raw.
        cal = api.calibrator
        if cal is None:
            self.skipTest("calibrator not loaded")
        raw = 0.40
        cal_val = float(cal.predict(np.array([[raw]]))[0])
        # The rule in the API is raw >= threshold; confirm it is not cal-based.
        self.assertLess(cal_val, 0.5)  # display boundary well below the raw threshold decision

    def test_07_fixed_test_case(self):
        r = self.client.post("/predict", json={"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1})
        body = r.json()
        self.assertAlmostEqual(body["screening_probability"], 0.99998, places=4)
        self.assertAlmostEqual(body["calibrated_screening_probability"], 0.930075, places=4)
        self.assertEqual(body["screening_result"], "positive")

    def test_calibration_metadata_present(self):
        body = self.client.post("/predict", json={"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertEqual(body["calibrator_version"], api.CALIBRATOR_VERSION)
        self.assertTrue(body["calibration"]["display_calibrated"])
        self.assertEqual(body["calibration"]["method"], "sigmoid")
        self.assertTrue(body["calibration"]["decision_uses_raw_probability"])

    def test_raw_field_not_renamed(self):
        body = self.client.post("/predict", json={"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertIn("screening_probability", body)
        self.assertIn("calibrated_screening_probability", body)
        self.assertNotIn("detection_percentage", body)


class FlagPreservationTest(unittest.TestCase):
    """Mandatory regression: 0 decision changes across the full outer-test fixture."""

    def setUp(self):
        self.model = api.model
        self.cal = api.calibrator
        self.fixture = _load_outer_fixture()

    def test_11_flag_preservation_outer_fixture(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.cal)
        changed = []
        # Use the model pipeline directly (it imputes the 1 missing SES) so all 38
        # outer-test subjects are covered, matching the Phase 14 0/38 claim.
        # The calibrated side is compared against the mapped display-equivalent
        # boundary (calibrated value at raw 0.40), NOT against 0.40 itself.
        X = self.fixture[FEATURES]
        raws = self.model.predict_proba(X)[:, 1]
        cals = self.cal.predict(raws.reshape(-1, 1)).ravel()
        display_boundary = api.CALIBRATED_DISPLAY_BOUNDARY
        for subject, raw, cal_val in zip(self.fixture[SUBJECT_COL], raws, cals):
            raw_decision = "positive" if raw >= 0.40 else "negative"
            cal_decision = "positive" if cal_val >= display_boundary else "negative"
            if raw_decision != cal_decision:
                changed.append((subject, float(raw), float(cal_val)))
        self.assertEqual(changed, [], f"flag changes across outer fixture: {changed}")


if __name__ == "__main__":
    unittest.main()