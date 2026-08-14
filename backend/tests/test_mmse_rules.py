"""
Backend tests for deterministic MMSE scoring and the hybrid routing that keeps
every objectively scorable item off the AI provider.

Fully deterministic (never AI): orientation_time, attention_serial7,
attention_spell_world.
Hybrid (deterministic when safe, AMBIGUOUS -> AI otherwise): orientation_place,
registration, delayed_recall, naming, repetition.
AI-only: writing.

Run from the project root:

    backend\\venv\\Scripts\\python.exe -m unittest backend.tests.test_mmse_rules

Covers the task's required checks:
  1. all five orientation_time items correct for the current date (5/5)
  2. all five incorrect with guaranteed-wrong answers (0/5)
  3. semantic equivalents accepted (year words, season case, ordinal date, ...)
  4. zero AI provider calls for orientation_time / serial-7 / spell-world
  5. deterministic keys excluded from the actual AI batch prompt
  6. missing AI result -> per-item error (never fabricated)
  7. deterministic-only batch succeeds even with AI_PROVIDER=none
  8. /predict unchanged (sanity import check)
  9. all five serial-7 items correct (93/86/79/72/65) -> 5/5, zero AI calls
 10. mixed serial-7 (93, 86, 80, 72, 65) -> 4/5 (1 1 0 1 1)
 11. number-word serial-7 answers -> 5/5
 12. WORLD-backwards letters deterministic (D/L/R/O/W, case-insensitive) 5/5
 13. Orientation-to-Place deterministic (Maharashtra==maharashtra, " Mumbai ")
 14. Registration / Delayed Recall deterministic ("apple"/"an apple"; banana -> 0)
 15. Naming deterministic (watch -> wristwatch, pencil)
 16. Repetition deterministic ("no ifs ands or buts")
 17. zero AI calls for every clear hybrid-section answer
 18. genuinely ambiguous responses still reach the AI provider
 19. accuracy regression: correct/incorrect/semantic-equivalent sets
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
    FULLY_DETERMINISTIC_SECTIONS,
    MMSEBatchItem,
    MMSEEvaluateRequest,
    evaluate_mmse_batch,
)
from src.mmse_rules import (  # noqa: E402
    AMBIGUOUS,
    SERIAL_7_EXPECTED,
    evaluate_attention_serial7,
    evaluate_attention_spell_world,
    evaluate_delayed_recall,
    evaluate_naming,
    evaluate_orientation_place,
    evaluate_orientation_time,
    evaluate_registration,
    evaluate_repetition,
    _parse_date,
    _parse_number,
    _parse_year,
)

ORIENTATION_KEYS = ["year", "season", "date", "day", "month"]
SERIAL7_CORRECT = ["93", "86", "79", "72", "65"]
SPELL_WORLD_CORRECT = ["D", "L", "R", "O", "W"]
PLACE_CONFIG = {"state": "Maharashtra", "county": "Mumbai District", "town": "Mumbai", "building": "General Hospital", "floor": "2"}
REGISTRATION_OBJECTS = {"1": "Apple", "2": "Table", "3": "Penny"}


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


class Serial7RulesTest(unittest.TestCase):
    """Unit tests for the deterministic Serial-7s rule engine."""

    def test_all_five_correct(self):
        for i, expected in enumerate(SERIAL_7_EXPECTED, start=1):
            result = evaluate_attention_serial7(str(i), str(expected))
            self.assertIsNotNone(result, i)
            self.assertTrue(result["correct"], f"item {i}: {expected}")
            self.assertEqual(result["score"], 1, i)
            self.assertEqual(result["confidence"], 1.0, i)

    def test_incorrect_answers(self):
        wrong = {1: "92", 2: "85", 3: "80", 4: "71", 5: "60"}
        for i, value in wrong.items():
            result = evaluate_attention_serial7(str(i), value)
            self.assertIsNotNone(result, i)
            self.assertFalse(result["correct"], f"item {i}: {value}")
            self.assertEqual(result["score"], 0, i)

    def test_mixed_4_of_5(self):
        answers = {1: "93", 2: "86", 3: "80", 4: "72", 5: "65"}
        expected = {1: True, 2: True, 3: False, 4: True, 5: True}
        for i, value in answers.items():
            result = evaluate_attention_serial7(str(i), value)
            self.assertEqual(result["correct"], expected[i], f"item {i}: {value}")
        total = sum(
            evaluate_attention_serial7(str(i), v)["score"] for i, v in answers.items()
        )
        self.assertEqual(total, 4)

    def test_number_words(self):
        words = {1: "ninety-three", 2: "eighty-six", 3: "seventy-nine", 4: "seventy-two", 5: "sixty-five"}
        for i, value in words.items():
            result = evaluate_attention_serial7(str(i), value)
            self.assertTrue(result["correct"], f"item {i}: {value}")

    def test_number_word_parsing(self):
        self.assertEqual(_parse_number("93"), 93)
        self.assertEqual(_parse_number(" 93 "), 93)
        self.assertEqual(_parse_number("93"), 93)
        self.assertEqual(_parse_number("ninety three"), 93)
        self.assertEqual(_parse_number("ninety-three"), 93)

    def test_unparseable_scores_incorrect_never_errors(self):
        result = evaluate_attention_serial7("1", "I don't know")
        self.assertIsNotNone(result)
        self.assertFalse(result["correct"])

    def test_unknown_or_out_of_range_key_returns_none(self):
        self.assertIsNone(evaluate_attention_serial7("99", "93"))
        self.assertIsNone(evaluate_attention_serial7("abc", "93"))

    def test_zero_key_maps_to_first_item(self):
        result = evaluate_attention_serial7("0", "93")
        self.assertIsNotNone(result)
        self.assertTrue(result["correct"])


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

    def test_serial7_items_never_call_provider(self):
        items = {
            f"attention_serial7.{i}": MMSEBatchItem(question="?", response=v, expected="")
            for i, v in enumerate(SERIAL7_CORRECT, start=1)
        }
        req = MMSEEvaluateRequest(items=items)
        with mock.patch.object(ai_eval, "_call_ollama") as call:
            call.side_effect = AssertionError("provider must not be called")
            outcome = evaluate_mmse_batch(req)
        self.assertEqual(len(outcome["items"]), 5)
        self.assertEqual(len(outcome["errors"]), 0)
        call.assert_not_called()

    def test_mixed_serial7_batch_scores_4_of_5(self):
        items = {
            f"attention_serial7.{i}": MMSEBatchItem(question="?", response=v, expected="")
            for i, v in {1: "93", 2: "86", 3: "80", 4: "72", 5: "65"}.items()
        }
        req = MMSEEvaluateRequest(items=items)
        with mock.patch.object(ai_eval, "_call_ollama") as call:
            call.side_effect = AssertionError("provider must not be called")
            outcome = evaluate_mmse_batch(req)
        call.assert_not_called()
        scores = [outcome["items"][f"attention_serial7.{i}"]["score"] for i in range(1, 6)]
        self.assertEqual(scores, [1, 1, 0, 1, 1])
        self.assertEqual(sum(scores), 4)

    def test_clean_batch_sends_only_writing_to_model(self):
        answers = _correct_answers(datetime.now())
        serial7 = {
            f"attention_serial7.{i}": MMSEBatchItem(question="?", response=v, expected="")
            for i, v in enumerate(SERIAL7_CORRECT, start=1)
        }
        spelling = {
            f"attention_spell_world.{i}": MMSEBatchItem(question="?", response=v, expected="")
            for i, v in enumerate(SPELL_WORLD_CORRECT, start=1)
        }
        place = {
            f"orientation_place.{k}": MMSEBatchItem(question="?", response=v, expected=v)
            for k, v in PLACE_CONFIG.items()
        }
        registration = {
            f"registration.{k}": MMSEBatchItem(question="?", response=v, expected=v)
            for k, v in REGISTRATION_OBJECTS.items()
        }
        recall = {
            f"delayed_recall.{k}": MMSEBatchItem(question="?", response=v, expected=v)
            for k, v in REGISTRATION_OBJECTS.items()
        }
        naming = {
            "naming.wristwatch": MMSEBatchItem(question="What is this?", response="watch", expected="wristwatch"),
            "naming.pencil": MMSEBatchItem(question="What is this?", response="pencil", expected="pencil"),
        }
        repetition = {"repetition.phrase": MMSEBatchItem(question="Repeat", response="No ifs, ands, or buts.", expected="No ifs, ands, or buts.")}
        writing = {"writing.sentence": MMSEBatchItem(question="Write a sentence", response="The patient is here today.", expected="")}

        items = {}
        items.update({f"orientation_time.{key}": MMSEBatchItem(question="?", response=resp, expected="") for key, resp in answers.items()})
        items.update(serial7)
        items.update(spelling)
        items.update(place)
        items.update(registration)
        items.update(recall)
        items.update(naming)
        items.update(repetition)
        items.update(writing)
        req = MMSEEvaluateRequest(items=items)

        captured = {}

        def fake_call(messages, timeout=None):
            captured["prompt"] = messages[1]["content"]
            return '{"items": {"writing.sentence": {"correct": true, "score": 1, "confidence": 0.99, "reason": "ok"}}}'

        with mock.patch.object(ai_eval, "_call_ollama", side_effect=fake_call) as call:
            outcome = evaluate_mmse_batch(req)

        call.assert_called_once()
        self.assertEqual(len(outcome["items"]), 30)  # 29 deterministic + 1 AI (writing)
        self.assertEqual(len(outcome["errors"]), 0)
        self.assertIn("writing.sentence", captured["prompt"])
        for forbidden in ("orientation_time", "attention_serial7", "attention_spell_world",
                          "orientation_place", "registration", "delayed_recall",
                          "naming", "repetition"):
            self.assertNotIn(forbidden, captured["prompt"])
        for key in outcome["items"]:
            if key != "writing.sentence":
                self.assertEqual(outcome["items"][key]["confidence"], 1.0)

    def test_missing_ai_result_becomes_per_item_error(self):
        req = MMSEEvaluateRequest(
            items={
                "writing.sentence": MMSEBatchItem(question="Write a sentence", response="The man reads.", expected="")
            }
        )
        with mock.patch.object(
            ai_eval,
            "_call_ollama",
            return_value='{"items": {}}',
        ):
            outcome = evaluate_mmse_batch(req)
        self.assertEqual(outcome["items"], {})
        self.assertIn("writing.sentence", outcome["errors"])

    def test_deterministic_only_succeeds_when_ai_disabled(self):
        answers = _correct_answers(datetime.now())
        with mock.patch.object(ai_eval, "AI_PROVIDER", "none"):
            outcome = evaluate_mmse_batch(_batch(**answers))
        self.assertEqual(len(outcome["items"]), 5)
        self.assertEqual(len(outcome["errors"]), 0)

    def test_serial7_only_succeeds_when_ai_disabled(self):
        items = {
            f"attention_serial7.{i}": MMSEBatchItem(question="?", response=v, expected="")
            for i, v in enumerate(SERIAL7_CORRECT, start=1)
        }
        req = MMSEEvaluateRequest(items=items)
        with mock.patch.object(ai_eval, "AI_PROVIDER", "none"):
            outcome = evaluate_mmse_batch(req)
        self.assertEqual(len(outcome["items"]), 5)
        self.assertEqual(len(outcome["errors"]), 0)

    def test_ai_items_still_503_when_ai_disabled(self):
        req = MMSEEvaluateRequest(
            items={
                "writing.sentence": MMSEBatchItem(question="Write a sentence", response="The man reads.", expected="")
            }
        )
        from fastapi import HTTPException

        with mock.patch.object(ai_eval, "AI_PROVIDER", "none"):
            with self.assertRaises(HTTPException) as ctx:
                evaluate_mmse_batch(req)
        self.assertEqual(ctx.exception.status_code, 503)

    def test_ambiguous_item_still_503_when_ai_disabled(self):
        req = MMSEEvaluateRequest(
            items={
                "naming.wristwatch": MMSEBatchItem(question="What is this?", response="the thing used to tell time", expected="wristwatch")
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
        self.assertIn("attention_serial7", DETERMINISTIC_SECTIONS)
        self.assertIn("attention_spell_world", DETERMINISTIC_SECTIONS)
        self.assertIn("orientation_place", DETERMINISTIC_SECTIONS)
        self.assertIn("registration", DETERMINISTIC_SECTIONS)
        self.assertIn("delayed_recall", DETERMINISTIC_SECTIONS)
        self.assertIn("naming", DETERMINISTIC_SECTIONS)
        self.assertIn("repetition", DETERMINISTIC_SECTIONS)
        self.assertNotIn("writing", DETERMINISTIC_SECTIONS)
        # Fully deterministic sections can never reach the AI provider.
        self.assertTrue(FULLY_DETERMINISTIC_SECTIONS.isdisjoint(ai_eval.AI_SECTIONS))
        self.assertTrue({"orientation_time", "attention_serial7", "attention_spell_world"}
                        <= FULLY_DETERMINISTIC_SECTIONS)


class SpellWorldRulesTest(unittest.TestCase):
    """WORLD-backwards letters are fully deterministic."""

    def test_all_five_correct(self):
        for i, letter in enumerate(SPELL_WORLD_CORRECT, start=1):
            result = evaluate_attention_spell_world(str(i), letter)
            self.assertIsNotNone(result, i)
            self.assertTrue(result["correct"], f"item {i}: {letter}")
            self.assertEqual(result["score"], 1, i)
            self.assertEqual(result["confidence"], 1.0, i)

    def test_lowercase_accepted(self):
        for i, letter in enumerate(SPELL_WORLD_CORRECT, start=1):
            result = evaluate_attention_spell_world(str(i), letter.lower())
            self.assertTrue(result["correct"], f"item {i}: {letter.lower()}")

    def test_wrong_letter_item4(self):
        result = evaluate_attention_spell_world("4", "E")
        self.assertIsNotNone(result)
        self.assertFalse(result["correct"])
        self.assertEqual(result["score"], 0)

    def test_full_string_uses_position(self):
        result = evaluate_attention_spell_world("4", "DLROW")
        self.assertTrue(result["correct"])  # position 4 of "DLROW" is O

    def test_unknown_key_returns_none(self):
        self.assertIsNone(evaluate_attention_spell_world("99", "D"))


class HybridRulesTest(unittest.TestCase):
    """Orientation-to-Place, Registration, Delayed Recall, Naming, Repetition."""

    # --- Orientation to Place ---
    def test_place_correct_and_case_insensitive(self):
        self.assertTrue(evaluate_orientation_place("state", "Maharashtra", expected="Maharashtra")["correct"])
        self.assertTrue(evaluate_orientation_place("state", "maharashtra", expected="Maharashtra")["correct"])

    def test_place_whitespace_normalized(self):
        self.assertTrue(evaluate_orientation_place("town", " Mumbai ", expected="Mumbai")["correct"])

    def test_place_incorrect(self):
        self.assertFalse(evaluate_orientation_place("state", "Delhi", expected="Maharashtra")["correct"])
        self.assertFalse(evaluate_orientation_place("town", "Pune", expected="Mumbai")["correct"])

    def test_place_unconfigured_is_ambiguous(self):
        self.assertIs(evaluate_orientation_place("state", "Maharashtra", expected=""), AMBIGUOUS)

    def test_place_uncertain_is_ambiguous(self):
        self.assertIs(evaluate_orientation_place("state", "I don't know", expected="Maharashtra"), AMBIGUOUS)

    # --- Registration / Delayed Recall ---
    def test_registration_clear_correct(self):
        self.assertTrue(evaluate_registration("1", "apple", expected="Apple")["correct"])
        self.assertTrue(evaluate_registration("1", "an apple", expected="Apple")["correct"])
        self.assertTrue(evaluate_registration("2", "a table", expected="Table")["correct"])
        self.assertTrue(evaluate_registration("3", "Penny", expected="Penny")["correct"])

    def test_registration_incorrect(self):
        self.assertFalse(evaluate_registration("1", "banana", expected="Apple")["correct"])
        self.assertFalse(evaluate_registration("2", "chair", expected="Table")["correct"])

    def test_registration_ambiguous(self):
        self.assertIs(evaluate_registration("1", "that fruit", expected="Apple"), AMBIGUOUS)

    def test_delayed_recall_uses_same_objects(self):
        self.assertTrue(evaluate_delayed_recall("1", "apple", expected="Apple")["correct"])
        self.assertTrue(evaluate_delayed_recall("1", "an apple", expected="Apple")["correct"])
        self.assertFalse(evaluate_delayed_recall("1", "banana", expected="Apple")["correct"])

    # --- Naming ---
    def test_naming_clear_correct(self):
        self.assertTrue(evaluate_naming("wristwatch", "wristwatch", expected="wristwatch")["correct"])
        self.assertTrue(evaluate_naming("wristwatch", "watch", expected="wristwatch")["correct"])
        self.assertTrue(evaluate_naming("wristwatch", "a wristwatch", expected="wristwatch")["correct"])
        self.assertTrue(evaluate_naming("pencil", "pencil", expected="pencil")["correct"])

    def test_naming_incorrect(self):
        self.assertFalse(evaluate_naming("wristwatch", "pen", expected="wristwatch")["correct"])
        self.assertFalse(evaluate_naming("pencil", "pen", expected="pencil")["correct"])

    def test_naming_ambiguous(self):
        self.assertIs(
            evaluate_naming("wristwatch", "the thing used to tell time", expected="wristwatch"),
            AMBIGUOUS,
        )

    # --- Repetition ---
    def test_repetition_correct(self):
        phrase = "No ifs, ands, or buts."
        self.assertTrue(evaluate_repetition("phrase", phrase, expected=phrase)["correct"])
        self.assertTrue(evaluate_repetition("phrase", "no ifs ands or buts", expected=phrase)["correct"])

    def test_repetition_incorrect(self):
        self.assertFalse(evaluate_repetition("phrase", "Totally different words", expected="No ifs, ands, or buts.")["correct"])

    def test_repetition_partial_is_ambiguous(self):
        self.assertIs(evaluate_repetition("phrase", "no ifs", expected="No ifs, ands, or buts."), AMBIGUOUS)


class HybridBatchRoutingTest(unittest.TestCase):
    """Prove zero AI calls for clear hybrid answers and AI routing for ambiguous."""

    def _no_provider_call(self, items):
        req = MMSEEvaluateRequest(items=items)
        with mock.patch.object(ai_eval, "_call_ollama") as call:
            call.side_effect = AssertionError("provider must not be called")
            outcome = evaluate_mmse_batch(req)
        call.assert_not_called()
        return outcome

    def test_place_never_calls_provider(self):
        items = {
            f"orientation_place.{k}": MMSEBatchItem(question="?", response=v, expected=v)
            for k, v in PLACE_CONFIG.items()
        }
        outcome = self._no_provider_call(items)
        self.assertEqual(len(outcome["items"]), 5)
        self.assertEqual(len(outcome["errors"]), 0)

    def test_spell_world_never_calls_provider(self):
        items = {
            f"attention_spell_world.{i}": MMSEBatchItem(question="?", response=v, expected="")
            for i, v in enumerate(SPELL_WORLD_CORRECT, start=1)
        }
        outcome = self._no_provider_call(items)
        self.assertEqual(len(outcome["items"]), 5)
        self.assertEqual(len(outcome["errors"]), 0)

    def test_registration_clear_never_calls_provider(self):
        items = {
            f"registration.{k}": MMSEBatchItem(question="?", response="an apple" if k == "1" else v, expected=v)
            for k, v in REGISTRATION_OBJECTS.items()
        }
        outcome = self._no_provider_call(items)
        self.assertEqual(len(outcome["items"]), 3)
        self.assertEqual(len(outcome["errors"]), 0)

    def test_delayed_recall_clear_never_calls_provider(self):
        items = {
            f"delayed_recall.{k}": MMSEBatchItem(question="?", response="apple" if k == "1" else v, expected=v)
            for k, v in REGISTRATION_OBJECTS.items()
        }
        outcome = self._no_provider_call(items)
        self.assertEqual(len(outcome["items"]), 3)
        self.assertEqual(len(outcome["errors"]), 0)

    def test_naming_clear_never_calls_provider(self):
        items = {
            "naming.wristwatch": MMSEBatchItem(question="?", response="watch", expected="wristwatch"),
            "naming.pencil": MMSEBatchItem(question="?", response="pencil", expected="pencil"),
        }
        outcome = self._no_provider_call(items)
        self.assertEqual(len(outcome["items"]), 2)
        self.assertEqual(len(outcome["errors"]), 0)

    def test_repetition_clear_never_calls_provider(self):
        phrase = "No ifs, ands, or buts."
        items = {"repetition.phrase": MMSEBatchItem(question="Repeat", response=phrase, expected=phrase)}
        outcome = self._no_provider_call(items)
        self.assertEqual(len(outcome["items"]), 1)
        self.assertEqual(len(outcome["errors"]), 0)

    def test_ambiguous_items_are_sent_to_ai(self):
        items = {
            "naming.wristwatch": MMSEBatchItem(question="What is this?", response="the thing used to tell time", expected="wristwatch"),
            "registration.1": MMSEBatchItem(question="Object 1", response="that fruit", expected="Apple"),
            "writing.sentence": MMSEBatchItem(question="Write a sentence", response="The man reads.", expected=""),
            "orientation_time.year": MMSEBatchItem(question="?", response="2026", expected=""),
        }
        captured = {}

        def fake_call(messages, timeout=None):
            captured["prompt"] = messages[1]["content"]
            return (
                '{"items": {'
                '"naming.wristwatch": {"correct": true, "score": 1, "confidence": 0.95, "reason": "ok"}, '
                '"registration.1": {"correct": false, "score": 0, "confidence": 0.9, "reason": "no"}, '
                '"writing.sentence": {"correct": true, "score": 1, "confidence": 0.98, "reason": "ok"}}}'
            )

        with mock.patch.object(ai_eval, "_call_ollama", side_effect=fake_call) as call:
            outcome = evaluate_mmse_batch(MMSEEvaluateRequest(items=items))

        call.assert_called_once()
        prompt = captured["prompt"]
        for key in ("naming.wristwatch", "registration.1", "writing.sentence"):
            self.assertIn(key, prompt)
        self.assertNotIn("orientation_time", prompt)
        self.assertEqual(len(outcome["items"]), 4)  # 1 deterministic + 3 AI
        self.assertEqual(len(outcome["errors"]), 0)


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
