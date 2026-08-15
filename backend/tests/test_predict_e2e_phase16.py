"""
Phase 16 — end-to-end /predict contract validation with realistic fixtures.

Run from project root:

    backend\\venv\\Scripts\\python.exe -m unittest backend.tests.test_predict_e2e_phase16

Fixtures are derived from real outer-test subjects so the borderline case is a
genuine decision boundary, not a synthetic guess:

  A. High-risk fixed case (74/M/10/23/1)          -> positive
  B. Typical negative (70/F/16/30/2)              -> negative
  C. Borderline raw near 0.40 (OAS2_0066-like)
     62/M/18/30/1  raw = 0.397968 (just below 0.40)
     calibrated crosses 0.40, decision MUST stay negative
  D. High MMSE (85/M/12/30/4)                     -> negative
  E. Missing SES: API schema rejects (422); pipeline imputes median.

For every fixture:
  - raw probability in [0,1]
  - calibrated probability in [0,1]
  - screening_result uses RAW probability (>= 0.40)
  - calibration does not change decision
  - model_version correct
  - calibrator_version correct
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from src.api import (  # noqa: E402
    SCREENING_MODEL_VERSION,
    SCREENING_THRESHOLD,
    CALIBRATOR_VERSION,
    app,
    calibrator,
    model,
)


class PredictE2EPhase16Test(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _cases(self):
        # (name, payload, expected_raw_result)
        return [
            ("A_high_risk", {"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}, "positive"),
            ("B_negative", {"age": 70, "sex": 0, "education_years": 16, "mmse": 30, "ses": 2}, "negative"),
            ("C_borderline", {"age": 62, "sex": 1, "education_years": 18, "mmse": 30, "ses": 1}, "negative"),
            ("D_high_mmse", {"age": 85, "sex": 1, "education_years": 12, "mmse": 30, "ses": 4}, "negative"),
        ]

    def test_all_fixtures(self):
        self.assertIsNotNone(model)
        self.assertIsNotNone(calibrator)
        for name, payload, expected in self._cases():
            with self.subTest(fixture=name):
                r = self.client.post("/predict", json=payload)
                self.assertEqual(r.status_code, 200, f"{name}: status {r.status_code}")
                body = r.json()

                raw = body["screening_probability"]
                cal = body["calibrated_screening_probability"]

                # raw and calibrated both in [0,1]
                self.assertGreaterEqual(raw, 0.0)
                self.assertLessEqual(raw, 1.0)
                self.assertGreaterEqual(cal, 0.0)
                self.assertLessEqual(cal, 1.0)

                # screening_result uses RAW threshold
                self.assertEqual(body["screening_result"],
                                 "positive" if raw >= SCREENING_THRESHOLD else "negative")
                # calibration metadata says decision uses raw
                self.assertTrue(body["calibration"]["decision_uses_raw_probability"])
                self.assertTrue(body["calibration"]["display_calibrated"])

                # model/calibrator versions
                self.assertEqual(body["model_version"], SCREENING_MODEL_VERSION)
                self.assertEqual(body["calibrator_version"], CALIBRATOR_VERSION)

                self.assertEqual(body["screening_result"], expected, name)

    def test_high_risk_fixed_case(self):
        r = self.client.post("/predict", json={"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1})
        body = r.json()
        self.assertAlmostEqual(body["screening_probability"], 0.99998, places=4)
        self.assertAlmostEqual(body["calibrated_screening_probability"], 0.930075, places=4)
        self.assertEqual(body["screening_result"], "positive")

    def test_borderline_calibrated_crosses_threshold_but_decision_stays_raw(self):
        """The hardest invariant: raw 0.397968 < 0.40 -> negative, even though the
        calibrated display value is above 0.40. The decision must remain negative."""
        r = self.client.post("/predict", json={"age": 62, "sex": 1, "education_years": 18, "mmse": 30, "ses": 1})
        body = r.json()
        raw = body["screening_probability"]
        cal = body["calibrated_screening_probability"]
        self.assertLess(raw, 0.40)
        self.assertGreater(cal, 0.40)  # display boundary, not a decision boundary
        self.assertEqual(body["screening_result"], "negative")

    def test_missing_ses(self):
        """Missing SES: the /predict schema requires ses (422) — that is the
        documented contract. The underlying pipeline imputes median SES, verified
        here directly so imputation still holds for the trained model."""
        import pandas as pd
        r = self.client.post("/predict", json={"age": 78, "sex": 1, "education_years": 12, "mmse": 24, "ses": None})
        self.assertEqual(r.status_code, 422)
        X = pd.DataFrame([{"age": 78, "sex": 1, "education_years": 12, "mmse": 24, "ses": None}])
        prob = float(model.predict_proba(X)[0][1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_calibration_never_changes_decision(self):
        for name, payload, _ in self._cases():
            with self.subTest(fixture=name):
                r = self.client.post("/predict", json=payload)
                body = r.json()
                raw = body["screening_probability"]
                expected = "positive" if raw >= SCREENING_THRESHOLD else "negative"
                self.assertEqual(body["screening_result"], expected, name)


if __name__ == "__main__":
    unittest.main()