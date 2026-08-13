"""
Backend tests for the MMSE Question 11 vision evaluation service.

Synthetic images only (generated in memory with Pillow). No real patient
images are used. Run from the project root:

    backend\\venv\\Scripts\\python.exe -m unittest backend.tests.test_vision_eval

Covers the task's required checks:
  1. valid image accepted
  2. invalid file rejected
  3. oversized image rejected
  4. reference figure loaded from trusted server asset
  5. provider request construction (OpenAI-compatible multimodal payload)
  6. provider response parsed
  7. structured result validated
  8. invalid JSON rejected
  9. missing fields rejected
 10. score/correct mismatch rejected
 11. low-confidence flagged (review_required)
 12. timeout handled
 13. provider unavailable handled
 14. Q11 score maps to 0/1
 15. final MMSE remains 0-30 (backend: score is 0/1; total math unchanged)
 16. /predict unchanged (sanity import check)
 17. text MMSE endpoint unchanged (contract import check)
"""

import base64
import io
import json
import os
import sys
import unittest
from unittest import mock

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import vision_image, vision_eval  # noqa: E402
from src.vision_eval import (  # noqa: E402
    AI_CONFIDENCE_REVIEW_THRESHOLD,
    VisionProvider,
    VisionProviderError,
    evaluate_copying_image,
    parse_vision_result,
)

# ---------------------------------------------------------------------------
# Synthetic image helpers (in memory)
# ---------------------------------------------------------------------------
def make_two_overlapping_figures(size=(640, 480), label="overlap") -> bytes:
    """Draw two overlapping geometric figures (pentagon + rectangle-ish)."""
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.polygon([(120, 80), (360, 60), (420, 240), (240, 360), (80, 260)], fill=(255, 0, 0))
    d.polygon([(260, 120), (560, 140), (520, 340), (300, 320)], fill=(0, 0, 255))
    d.text((20, 20), label, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_blank_image(size=(640, 480)) -> bytes:
    img = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_data_url(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


class FakeProvider(VisionProvider):
    """Deterministic provider for unit tests; overrides the HTTP call."""

    name = "fake"
    base_url = "http://fake/v1"
    model = "fake-model"
    api_key = ""

    def __init__(self, content: str, error: VisionProviderError | None = None):
        self.content = content
        self.error = error
        self.last_messages = None
        self.last_payload = None

    def call(self, messages: list) -> str:
        self.last_messages = messages
        self.last_payload = self.build_payload(messages)
        if self.error:
            raise self.error
        return self.content


# ---------------------------------------------------------------------------
# Image validation (checks 1, 2, 3)
# ---------------------------------------------------------------------------
class TestImageValidation(unittest.TestCase):
    def test_valid_jpeg_accepted(self):
        data_url = vision_image.prepare_patient_image(make_two_overlapping_figures(), "image/jpeg")
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))

    def test_valid_png_accepted(self):
        data_url = vision_image.prepare_patient_image(
            make_png(make_two_overlapping_figures()), "image/png"
        )
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))

    def test_empty_body_rejected(self):
        with self.assertRaises(vision_image.VisionImageError):
            vision_image.prepare_patient_image(b"", "image/jpeg")

    def test_unsupported_mime_rejected(self):
        with self.assertRaises(vision_image.VisionImageError):
            vision_image.prepare_patient_image(make_two_overlapping_figures(), "application/pdf")

    def test_oversized_image_rejected(self):
        blob = b"\x00" * (vision_image.MAX_UPLOAD_BYTES + 1)
        with self.assertRaises(vision_image.VisionImageError):
            vision_image.prepare_patient_image(blob, "image/jpeg")

    def test_undecodable_bytes_rejected(self):
        with self.assertRaises(vision_image.VisionImageError):
            vision_image.prepare_patient_image(b"not an image at all", "image/jpeg")


# ---------------------------------------------------------------------------
# Reference figure (check 4)
# ---------------------------------------------------------------------------
class TestReferenceFigure(unittest.TestCase):
    def test_reference_loaded_from_server_asset(self):
        raw = vision_image.load_reference_figure_bytes()
        self.assertGreater(len(raw), 0)
        img = Image.open(io.BytesIO(raw))
        img.verify()  # raises if corrupted

    def test_reference_is_jpeg_not_duplicated(self):
        data_url = vision_image.prepare_reference_figure()
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        # The reference is loaded from the existing trusted asset path
        self.assertTrue("mmse-copying-figure" in vision_image._reference_path())


# ---------------------------------------------------------------------------
# Provider request construction (check 5, 6) and result validation (7-11)
# ---------------------------------------------------------------------------
class TestProviderAndValidation(unittest.TestCase):
    def test_payload_is_openai_compatible_multimodal(self):
        provider = FakeProvider(json.dumps({
            "correct": True, "score": 1, "confidence": 0.91, "reason": "ok"
        }))
        provider.call(provider.build_messages("data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"))
        self.assertIn("chat/completions", provider.chat_url())
        user_content = provider.last_messages[1]["content"]
        self.assertTrue(any(p["type"] == "text" for p in user_content))
        image_parts = [p for p in user_content if p["type"] == "image_url"]
        self.assertEqual(len(image_parts), 2)  # reference + patient
        self.assertEqual(image_parts[0]["image_url"]["url"], "data:image/jpeg;base64,AAA")

    def test_valid_result_parsed(self):
        res = parse_vision_result(
            '{"correct": true, "score": 1, "confidence": 0.91, "reason": "Both figures present."}'
        )
        self.assertTrue(res["correct"])
        self.assertEqual(res["score"], 1)
        self.assertAlmostEqual(res["confidence"], 0.91)

    def test_invalid_json_rejected(self):
        with self.assertRaises(ValueError):
            parse_vision_result("this is not json")

    def test_missing_fields_rejected(self):
        with self.assertRaises(ValueError):
            parse_vision_result('{"correct": true, "score": 1}')

    def test_score_correct_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            parse_vision_result(
                '{"correct": true, "score": 0, "confidence": 0.9, "reason": "x"}'
            )

    def test_confidence_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            parse_vision_result(
                '{"correct": true, "score": 1, "confidence": 1.5, "reason": "x"}'
            )

    def test_low_confidence_flagged(self):
        res = evaluate_copying_image_with_fake(
            FakeProvider(json.dumps({
                "correct": True, "score": 1,
                "confidence": AI_CONFIDENCE_REVIEW_THRESHOLD - 0.1,
                "reason": "uncertain",
            }))
        )
        self.assertTrue(res["review_required"])

    def test_high_confidence_not_flagged(self):
        res = evaluate_copying_image_with_fake(
            FakeProvider(json.dumps({
                "correct": True, "score": 1,
                "confidence": 0.95,
                "reason": "certain",
            }))
        )
        self.assertFalse(res["review_required"])


def evaluate_copying_image_with_fake(provider: FakeProvider) -> dict:
    """Run evaluate_copying_image but with a deterministic fake provider."""
    with mock.patch.object(vision_eval, "get_provider", return_value=provider):
        return evaluate_copying_image(make_two_overlapping_figures(), "image/jpeg")


# ---------------------------------------------------------------------------
# End-to-end through the fake provider (check 6, 14)
# ---------------------------------------------------------------------------
class TestEndToEnd(unittest.TestCase):
    def test_end_to_end_score_maps_to_01(self):
        for correct, expected_score in ((True, 1), (False, 0)):
            provider = FakeProvider(json.dumps({
                "correct": correct,
                "score": expected_score,
                "confidence": 0.88,
                "reason": "synthetic",
            }))
            res = evaluate_copying_image_with_fake(provider)
            self.assertEqual(res["score"], expected_score)
            self.assertIs(res["correct"], correct)

    def test_provider_timeout_handled(self):
        provider = FakeProvider("", error=VisionProviderError("timeout", "provider timed out"))
        with self.assertRaises(VisionProviderError) as ctx:
            evaluate_copying_image_with_fake(provider)
        self.assertEqual(ctx.exception.kind, "timeout")

    def test_provider_unavailable_handled(self):
        provider = FakeProvider(
            "", error=VisionProviderError("unavailable", "cannot reach ollama provider")
        )
        with self.assertRaises(VisionProviderError) as ctx:
            evaluate_copying_image_with_fake(provider)
        self.assertEqual(ctx.exception.kind, "unavailable")

    def test_invalid_model_output_handled(self):
        provider = FakeProvider("totally not json")
        with self.assertRaises(VisionProviderError) as ctx:
            evaluate_copying_image_with_fake(provider)
        self.assertEqual(ctx.exception.kind, "invalid")

    def test_provider_selection_ollama_default(self):
        provider = vision_eval.get_provider()
        self.assertEqual(provider.name, "ollama")
        self.assertIn("/v1", provider.base_url)


# ---------------------------------------------------------------------------
# Contract sanity (checks 15, 16, 17)
# ---------------------------------------------------------------------------
class TestUnchangedContracts(unittest.TestCase):
    def test_predict_still_importable(self):
        from src.api import app  # noqa: F401
        routes = {r.path for r in app.routes}
        self.assertIn("/predict", routes)
        self.assertIn("/mmse/evaluate", routes)
        self.assertIn("/mmse/copying/evaluate", routes)


if __name__ == "__main__":
    unittest.main()
