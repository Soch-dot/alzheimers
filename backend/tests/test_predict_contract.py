"""
Backend tests for the versioned /predict binary screening contract
(experimental binary screening candidate).

Run from the project root:

    backend\\venv\\Scripts\\python.exe -m unittest backend.tests.test_predict_contract

Covers:
  A. valid positive case
  B. valid negative case
  C. boundary values
  D. numeric feature order
  E. missing/invalid request fields
  F. threshold exactly 0.40
  G. probability in [0,1]
  H. model_version
  I. screening target
  J. non-diagnostic metadata

These tests verify CONTRACT correctness, not a desired probability value.
No 99% assumption is made.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from src.api import (  # noqa: E402
    SCREENING_MODEL_VERSION,
    SCREENING_THRESHOLD,
    SCREENING_TARGET,
    CALIBRATOR_VERSION,
    CALIBRATED_DISPLAY_BOUNDARY,
    app,
)

FEATURE_KEYS = ["age", "sex", "education_years", "mmse", "ses"]


class PredictContractTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _post(self, payload):
        return self.client.post("/predict", json=payload)

    # A. valid positive case
    def test_valid_positive_case(self):
        r = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["screening_result"], "positive")
        self.assertEqual(body["predicted_class"], "dementia_related")
        self.assertGreaterEqual(body["screening_probability"], SCREENING_THRESHOLD)

    # B. valid negative case
    def test_valid_negative_case(self):
        r = self._post({"age": 70, "sex": 0, "education_years": 16, "mmse": 30, "ses": 2})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["screening_result"], "negative")
        self.assertEqual(body["predicted_class"], "nondemented")
        self.assertLess(body["screening_probability"], SCREENING_THRESHOLD)

    # C. boundary values (max/min valid ranges; should still be 200)
    def test_boundary_values(self):
        r = self._post({"age": 50, "sex": 0, "education_years": 6, "mmse": 0.0, "ses": 1})
        self.assertEqual(r.status_code, 200)
        r2 = self._post({"age": 100, "sex": 1, "education_years": 20, "mmse": 30.0, "ses": 5})
        self.assertEqual(r2.status_code, 200)

    # D. numeric feature order is fixed (regardless of JSON key order)
    def test_numeric_feature_order(self):
        # send keys in scrambled order; response features must be in canonical order
        r = self._post({"ses": 1, "mmse": 23, "sex": 1, "education_years": 10, "age": 74})
        self.assertEqual(r.status_code, 200)
        features = r.json()["features"]
        self.assertEqual(list(features.keys()), FEATURE_KEYS)
        # and the probability matches the canonical-order fixture from the frozen candidate
        self.assertAlmostEqual(r.json()["screening_probability"], 0.99998, places=4)

    # E. missing/invalid request fields
    def test_missing_field(self):
        r = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23})  # missing ses
        self.assertEqual(r.status_code, 422)

    def test_invalid_type(self):
        r = self._post({"age": "old", "sex": 1, "education_years": 10, "mmse": 23, "ses": 1})
        self.assertEqual(r.status_code, 422)

    def test_extra_field_ignored_or_rejected(self):
        r = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1, "cdr": 1.0})
        # extra fields are allowed by the model schema; contract only needs 200
        self.assertEqual(r.status_code, 200)

    # F. threshold exactly 0.40
    def test_threshold_exact_0_40(self):
        self.assertEqual(SCREENING_THRESHOLD, 0.40)

    # G. probability in [0,1]
    def test_probability_in_unit_interval(self):
        for payload in [
            {"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1},
            {"age": 70, "sex": 0, "education_years": 16, "mmse": 30, "ses": 2},
        ]:
            p = self._post(payload).json()["screening_probability"]
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    # H. model_version present and correct
    def test_model_version(self):
        body = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertEqual(body["model_version"], SCREENING_MODEL_VERSION)

    # I. screening target present
    def test_screening_target(self):
        body = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertEqual(body["screening_target"], SCREENING_TARGET)

    # J. non-diagnostic metadata
    def test_nondiagnostic_metadata(self):
        body = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertTrue(body["interpretation"]["not_a_diagnosis"])
        self.assertEqual(body["interpretation"]["label"], "Model-estimated screening probability")
        self.assertFalse(body["limitations"]["clinical_validation"])
        self.assertFalse(body["limitations"]["prospective_conversion_prediction"])

    def test_no_old_3class_fields_silently_present(self):
        """The old 3-class fields must not be silently reinterpreted as the new probability."""
        body = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertIn("screening_probability", body)
        self.assertNotIn("detection_percentage", body)
        # 3-class probabilities must stay absent (Phase 15)
        for old_field in ["nondemented_probability", "converted_probability",
                          "demented_probability", "detection_percentage"]:
            self.assertNotIn(old_field, body)

    # K. calibrated display field present with calibration metadata
    def test_calibrated_screening_probability_present(self):
        body = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertIn("calibrated_screening_probability", body)
        self.assertIsInstance(body["calibrated_screening_probability"], float)

    def test_calibrator_version_and_boundary(self):
        body = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertEqual(body["calibrator_version"], CALIBRATOR_VERSION)
        self.assertAlmostEqual(body["calibrated_threshold_equivalent"], CALIBRATED_DISPLAY_BOUNDARY, places=4)

    def test_calibration_metadata(self):
        body = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        cal = body["calibration"]
        self.assertTrue(cal["display_calibrated"])
        self.assertEqual(cal["method"], "sigmoid")
        self.assertTrue(cal["decision_uses_raw_probability"])

    def test_raw_field_not_renamed(self):
        body = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertEqual(list(body.keys()).count("screening_probability"), 1)
        self.assertEqual(list(body.keys()).count("calibrated_screening_probability"), 1)

    # L. screening_result must follow the RAW threshold, not the calibrated value
    def test_decision_follows_raw_threshold(self):
        # raw 0.99998 -> positive even though calibrated (0.930) is also high
        body = self._post({"age": 74, "sex": 1, "education_years": 10, "mmse": 23, "ses": 1}).json()
        self.assertGreaterEqual(body["screening_probability"], SCREENING_THRESHOLD)
        self.assertEqual(body["screening_result"], "positive")


if __name__ == "__main__":
    unittest.main()