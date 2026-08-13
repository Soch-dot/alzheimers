"""
Backend tests for deterministic MMSE Orientation-to-Time scoring and the
backend regression fix: the five orientation_time items must be scored
deterministically server-side and must NEVER reach the AI provider.

Run from the project root:

    backend\\venv\\Scripts\\python.exe -m unittest backend.tests.test_mmse_rules

Covers the task's required checks:
  1. all five orientation_time items correct for the current date (5/5)
  2. all five incorrect with guaranteed-wrong answers (0/5)
  3. semantic equivalents accepted (year words, season case, ordinal date,
     lowercase weekday, "the month of ...")
  4. zero AI provider calls for orientation_time items
  5. orientation_time keys excluded from the actual AI batch prompt
  6. missing AI result -> per-item error (never fabricated)
  7. deterministic-only batch succeeds even with AI_PROVIDER=none
  8. /predict unchanged (sanity import check)
"""

import os
import sys
import unittest
from datetime import datetime
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import ai_eval, mmse_rules  # noqa: E402
from src.ai_eval import (  # noqa: E402
    DETERMINISTIC_SECTIONS,
    MMSEBatchItem,
    MMSEEvaluateRequest,
    evaluate_mmse_batch,
)
from src.mmse_rules import (  # noqa: E402
    evaluate_orientation_time,
    _parse_date,
    _parse_year,
)

ORIENTATION_KEYS = ["year", "season", "date", "day", "month"]


def _batch(**kwargs):
    """Build an MMSEEvaluateRequest with item responses from kwargs."""
    items = {
        f"orientation_time.{key}": MMSEBatchItem(question="?", response=resp, expected="")
        for key, resp in kwargs.items()
    }
    return MMSEEvaluateRequest(items=items)


def _correct_answers(now):
    """All-correct responses for the given reference time (mirrors the server rule)."""
    if now.month in (12, 1, 2):
        season = "winter"
    elif now.month in (3, 4, 5):
        season = "spring"
    elif now.month in (6, 7, 8):
        season = "summer"
    else:
        season = "fall"
    return {
        "year": str(now.year),
        "season": season,
        "date": str(now.day),
        "day": now.strftime("%A"),
        "month": now.strftime("%B"),
    }


class EvaluationRulesTest(unittest.TestCase):
    """Unit tests for the deterministic rule engine (fixed reference date)."""

    NOW = datetime(2026, 8, 13, 10, 0, 0)  # Thursday 13 August 2026

    def test_all_five_correct(self):
        answers = _correct_answers(self.NOW)
        for key in ORIENTATION_KEYS:
            result = evaluate_orientation_time(key, answers[key], now=self.NOW)
            self.assertIsNotNone(result, key)
            self.assertTrue(result["correct"], key)
            self.assertEqual(result["score"], 1, key)
            self.assertEqual(result["confidence"], 1.0, key)

    def test_all_five_incorrect(self):
        wrong = {"year": "1999", "season": "winter", "date": "1", "day": "Monday", "month": "January"}
        for key in ORIENTATION_KEYS:
            result = evaluate_orientation_time(key, wrong[key], now=self.NOW)
            self.assertIsNotNone(result, key)
            self.assertFalse(result["correct"], key)
            self.assertEqual(result["score"], 0, key)

    def test_semantic_equivalents_accepted(self):
        equivalents = {
            "year": "twenty twenty-six",
            "season": "Summer",
            "date": "13th",
            "day": "thursday",
            "month": "the month of August",
        }
        for key in ORIENTATION_KEYS:
            result = evaluate_orientation_time(key, equivalents[key], now=self.NOW)
            self.assertTrue(result["correct"], f"{key}: {equivalents[key]}")
            self.assertEqual(result["score"], 1, key)

    def test_more_year_and_date_forms(self):
        self.assertEqual(_parse_year("2026"), 2026)
        self.assertEqual(_parse_year("two thousand twenty-six"), 2026)
        self.assertEqual(_parse_year("nineteen ninety-nine"), 1999)
        self.assertEqual(_parse_year("twenty twenty"), 2020)
        self.assertEqual(_parse_date("13"), 13)
        self.assertEqual(_parse_date("13th"), 13)
        self.assertEqual(_parse_date("thirteen"), 13)
        self.assertEqual(_parse_date("thirteenth"), 13)

    def test_unparseable_scores_incorrect_never_errors(self):
        result = evaluate_orientation_time("year", "I don't know", now=self.NOW)
        self.assertIsNotNone(result)
        self.assertFalse(result["correct"])
        result = evaluate_orientation_time("season", "banana", now=self.NOW)
        self.assertIsNotNone(result)
        self.assertFalse(result["correct"])

    def test_unknown_key_returns_none(self):
        self.assertIsNone(evaluate_orientation_time("bogus", "x", now=self.NOW))

    def test_season_autumn_equivalence(self):
        fall = datetime(2026, 10, 5)
        self.assertTrue(
            evaluate_orientation_time("season", "fall", now=fall)["correct"]
        )
        self.assertTrue(
            evaluate_orientation_time("season", "autumn", now=fall)["correct"]
        )

    def test_month_and_day_abbreviations(self):
        self.assertTrue(
            evaluate_orientation_time("month", "aug", now=self.NOW)["correct"]
        )
        self.assertTrue(
            evaluate_orientation_time("day", "Thu", now=self.NOW)["correct"]
        )


class BatchRoutingTest(unittest.TestCase):
    """Tests that the batch endpoint routes orientation_time deterministically."""

    def test_orientation_items_never_call_provider(self):
        answers = _correct_answers(datetime.now())
        with mock.patch.object(ai_eval, "_call_ollama") as call:
            call.side_effect = AssertionError("provider must not be called")
            outcome = evaluate_mmse_batch(_batch(**answers))
        self.assertEqual(len(outcome["items"]), 5)
        self.assertEqual(len(outcome["errors"]), 0)
        call.assert_not_called()

    def test_mixed_batch_excludes_orientation_from_model_prompt(self):
        answers = _correct_answers(datetime.now())
        naming = {"naming.wristwatch": MMSEBatchItem(question="What is this?", response="watch", expected="wristwatch")}
        items = {
            f"orientation_time.{key}": MMSEBatchItem(question="?", response=resp, expected="")
            for key, resp in answers.items()
        }
        items.update(naming)
        req = MMSEEvaluateRequest(items=items)

        captured = {}

        def fake_call(messages, timeout=None):
            captured["prompt"] = messages[1]["content"]
            return '{"items": {"naming.wristwatch": {"correct": true, "score": 1, "confidence": 0.99, "reason": "ok"}}}'

        with mock.patch.object(ai_eval, "_call_ollama", side_effect=fake_call) as call:
            outcome = evaluate_mmse_batch(req)

        call.assert_called_once()
        self.assertNotIn("orientation_time", captured["prompt"])
        self.assertIn("naming.wristwatch", captured["prompt"])
        self.assertEqual(len(outcome["items"]), 6)
        self.assertEqual(len(outcome["errors"]), 0)
        for key in outcome["items"]:
            if key.startswith("orientation_time."):
                self.assertEqual(outcome["items"][key]["confidence"], 1.0)

    def test_missing_ai_result_becomes_per_item_error(self):
        req = MMSEEvaluateRequest(
            items={
                "naming.pencil": MMSEBatchItem(question="What is this?", response="pencil", expected="pencil")
            }
        )
        with mock.patch.object(
            ai_eval,
            "_call_ollama",
            return_value='{"items": {}}',
        ):
            outcome = evaluate_mmse_batch(req)
        self.assertEqual(outcome["items"], {})
        self.assertIn("naming.pencil", outcome["errors"])

    def test_deterministic_only_succeeds_when_ai_disabled(self):
        answers = _correct_answers(datetime.now())
        with mock.patch.object(ai_eval, "AI_PROVIDER", "none"):
            outcome = evaluate_mmse_batch(_batch(**answers))
        self.assertEqual(len(outcome["items"]), 5)
        self.assertEqual(len(outcome["errors"]), 0)

    def test_ai_items_still_503_when_ai_disabled(self):
        req = MMSEEvaluateRequest(
            items={
                "naming.pencil": MMSEBatchItem(question="What is this?", response="pencil", expected="pencil")
            }
        )
        from fastapi import HTTPException

        with mock.patch.object(ai_eval, "AI_PROVIDER", "none"):
            with self.assertRaises(HTTPException) as ctx:
                evaluate_mmse_batch(req)
        self.assertEqual(ctx.exception.status_code, 503)

    def test_empty_request_422(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            evaluate_mmse_batch(MMSEEvaluateRequest(items={}))
        self.assertEqual(ctx.exception.status_code, 422)

    def test_section_sets_are_disjoint_and_cover_deterministic(self):
        self.assertIn("orientation_time", DETERMINISTIC_SECTIONS)
        self.assertNotIn("orientation_time", ai_eval.AI_SECTIONS)


class ContractTest(unittest.TestCase):
    """Sanity checks that untouched systems still import and behave."""

    def test_predict_import_and_math(self):
        from src.api import app  # noqa: F401
        routes = {r.path for r in app.routes}
        self.assertIn("/predict", routes)
        self.assertIn("/mmse/evaluate", routes)
        self.assertIn("/mmse/copying/evaluate", routes)

    def test_ai_eval_still_has_expected_symbols(self):
        self.assertTrue(callable(ai_eval.evaluate_mmse_batch))
        self.assertTrue(callable(ai_eval.parse_batch_result))


if __name__ == "__main__":
    unittest.main()
