"""
MMSE Question 11 — vision-assisted figure copying evaluation service.

Layering:
    POST /mmse/copying/evaluate
        -> vision_image.prepare_patient_image (validate/normalize, in-memory)
        -> evaluate_copying_image (this module)
            -> provider adapter (ONE configured provider)
                Ollama (local, OpenAI-compatible /v1/chat/completions)
                Gemini (OpenAI-compatible endpoint)
                OpenAI (native OpenAI-compatible API)
            -> normalized structured result {correct, score, confidence, reason}
            -> confidence/review flag

Design rules:
  - ONE provider runs per assessment (VISION_PROVIDER=ollama|gemini|openai).
    No provider voting, no automatic multi-provider fallback yet.
  - The provider abstraction is built around an OpenAI-compatible
    chat/completions contract with multimodal content:
        user content = [ {type: text, text: ...},
                         {type: image_url, image_url: {url: data:image/jpeg;base64,...}},
                         {type: image_url, image_url: {url: data:image/jpeg;base64,...}} ]
    Provider-specific adapters only handle: base URL, model, auth header,
    endpoint path, request payload construction, and response extraction.
    All shared logic (MMSE copying criterion prompt, normalized schema,
    validation, confidence/review, error normalization, timeout) lives here.
  - Structured output is validated strictly: malformed output => NO score,
    a controlled error, and the item is never silently scored.
  - Timeouts use a dedicated VISION_TIMEOUT (default 120s) — NOT the text-MMSE
    OLLAMA_BATCH_TIMEOUT / frontend 180s budget.
  - Confidence is a model signal, not clinical certainty. confidence <
    AI_CONFIDENCE_REVIEW_THRESHOLD (0.7) => review_required = true.
  - Errors are normalized to friendly strings; raw diagnostics (stack traces,
    OS errors, HTTP internals) are for backend logs only.
  - Patient images are never persisted, never logged, never returned.

Config (backend env only; never in React):
    VISION_PROVIDER        (ollama|gemini|openai; default ollama)
    OLLAMA_BASE_URL        (default http://127.0.0.1:11434)
    OLLAMA_VISION_MODEL    (default gemma3)
    GEMINI_API_KEY         (backend-only)
    GEMINI_VISION_MODEL    (default gemini-2.0-flash)
    GEMINI_BASE_URL        (OpenAI-compatible Gemini endpoint)
    OPENAI_API_KEY         (backend-only)
    OPENAI_VISION_MODEL    (default gpt-4o-mini)
    OPENAI_BASE_URL        (default https://api.openai.com/v1)
    VISION_TIMEOUT         (default 120)
    AI_CONFIDENCE_REVIEW_THRESHOLD (default 0.7)

Only stdlib is used for HTTP (urllib), so no new dependencies are added.
"""

import json
import os
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

from src.vision_image import (
    VisionImageError,
    prepare_patient_image,
    prepare_reference_figure,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config (backend-only secrets)
# ---------------------------------------------------------------------------
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "ollama").strip().lower()
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT", "120"))
AI_CONFIDENCE_REVIEW_THRESHOLD = float(
    os.getenv("AI_CONFIDENCE_REVIEW_THRESHOLD", "0.7")
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "gemma3")

GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
).rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Prompt: the ONLY MMSE copying criterion (do not invent clinical criteria)
# ---------------------------------------------------------------------------
VISION_SYSTEM_PROMPT = (
    "You are evaluating Question 11 of the Mini-Mental State Examination "
    "(MMSE): the examiner asks the patient to copy a drawing. The reference "
    "figure is made of two overlapping geometric figures. You are NOT making "
    "a diagnosis and you have no clinical authority. "
    "You will receive two images: the first is the official reference figure "
    "(the exact stimulus the patient was asked to copy) and the second is the "
    "patient's attempt. Judge the patient's attempt against the reference "
    "based ONLY on task-relevant geometric properties: the required geometric "
    "structure of each figure (e.g. sides/angles/vertices), the presence of "
    "both figures, and the required overlap/intersection between them. "
    "Do NOT score artistic quality, handwriting, penmanship, aesthetics, "
    "paper cleanliness, color, arbitrary line thickness, neatness, or how "
    "'professional' the drawing looks. "
    'Respond with ONLY a JSON object matching exactly this schema: '
    '{"correct": boolean, "score": 0 or 1, "confidence": number between 0 and '
    '1, "reason": short string}. score must be 1 when correct is true and 0 '
    "when correct is false. Set confidence below 0.7 whenever you are "
    "genuinely uncertain. Do not include any text outside the JSON object."
)

VISION_USER_PROMPT = (
    "First image: the official MMSE reference figure. Second image: the "
    "patient's attempt. Evaluate whether the patient's attempt satisfies the "
    "MMSE copying criterion (geometric structure, presence of both figures, "
    "and the required overlap/intersection). Return the JSON object."
)


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------
class VisionProviderError(Exception):
    """
    A controlled provider/assessment failure.

    kind:
      "unavailable"  -> provider unreachable / not configured
      "timeout"      -> VISION_TIMEOUT exceeded
      "invalid"      -> model returned unparseable/invalid structured output
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


# ---------------------------------------------------------------------------
# Provider abstraction (OpenAI-compatible chat/completions core)
# ---------------------------------------------------------------------------
class VisionProvider:
    """Base adapter. Subclasses supply base_url/model/api_key/endpoint."""

    name: str = "unknown"
    base_url: str = ""
    model: str = ""
    api_key: str = ""

    def build_messages(self, reference_data_url: str, patient_data_url: str) -> list:
        return [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": reference_data_url}},
                    {"type": "image_url", "image_url": {"url": patient_data_url}},
                ],
            },
        ]

    def build_payload(self, messages: list) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256,
        }
        return payload

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def call(self, messages: list) -> str:
        """Send one multimodal chat request; return the model's content string."""
        payload = self.build_payload(messages)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.chat_url(), data=data, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=VISION_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # Do not leak the provider's raw body into the UI; log it dev-side.
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:500]
            except Exception:
                pass
            raise VisionProviderError(
                "unavailable",
                f"{self.name} returned HTTP {exc.code}",
            ) from exc
        except urllib.error.URLError as exc:
            raise VisionProviderError(
                "unavailable",
                f"cannot reach {self.name} provider: {exc.reason}",
            ) from exc
        except TimeoutError as exc:
            raise VisionProviderError("timeout", "provider timed out") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise VisionProviderError(
                "unavailable", f"{self.name} returned non-JSON"
            ) from exc
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise VisionProviderError(
                "unavailable", f"{self.name} returned an empty response"
            )
        return content


class OllamaVisionProvider(VisionProvider):
    name = "ollama"
    base_url = f"{OLLAMA_BASE_URL}/v1"
    model = OLLAMA_VISION_MODEL
    api_key = ""


class GeminiVisionProvider(VisionProvider):
    name = "gemini"
    base_url = GEMINI_BASE_URL
    model = GEMINI_VISION_MODEL
    api_key = GEMINI_API_KEY


class OpenAIVisionProvider(VisionProvider):
    name = "openai"
    base_url = OPENAI_BASE_URL
    model = OPENAI_VISION_MODEL
    api_key = OPENAI_API_KEY


def get_provider() -> VisionProvider:
    """Return the single configured provider adapter."""
    provider = VISION_PROVIDER
    if provider == "ollama":
        return OllamaVisionProvider()
    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise VisionProviderError(
                "unavailable", "Gemini vision unavailable: GEMINI_API_KEY is not set."
            )
        return GeminiVisionProvider()
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise VisionProviderError(
                "unavailable", "OpenAI vision unavailable: OPENAI_API_KEY is not set."
            )
        return OpenAIVisionProvider()
    raise VisionProviderError(
        "unavailable",
        f"Unsupported VISION_PROVIDER: {provider}. "
        "Use ollama, gemini, or openai.",
    )


# ---------------------------------------------------------------------------
# Structured-output validation (shared; not duplicated per provider)
# ---------------------------------------------------------------------------
def _parse_json_object(text: str) -> dict:
    """Best-effort parse of the model's JSON output. Raise ValueError."""
    text = text.strip()
    if text.startswith("```"):
        lines = [
            line for line in text.splitlines() if not line.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()

    def _load(raw: str) -> dict:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("model output is not a JSON object")
        return data

    try:
        return _load(text)
    except (json.JSONDecodeError, ValueError):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("model did not return valid JSON")
        try:
            return _load(text[start : end + 1])
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("model did not return valid JSON") from exc


def parse_vision_result(content: str) -> dict:
    """
    Validate the model's normalized Q11 result. Raise ValueError on anything
    invalid so no score is produced for malformed output.
    """
    data = _parse_json_object(content)

    correct = data.get("correct")
    if not isinstance(correct, bool):
        raise ValueError("'correct' must be a boolean")

    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("'score' must be 0 or 1")
    if float(score) not in (0.0, 1.0) or int(round(float(score))) != (1 if correct else 0):
        raise ValueError("'score' must match 'correct' (1 if correct else 0)")

    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("'confidence' must be a number")
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("'confidence' must be between 0 and 1")

    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("'reason' must be a non-empty string")

    return {
        "correct": correct,
        "score": 1 if correct else 0,
        "confidence": round(confidence, 4),
        "reason": reason.strip()[:500],
    }


# ---------------------------------------------------------------------------
# Public entry point used by api.py (Q11 vision evaluation)
# ---------------------------------------------------------------------------
def evaluate_copying_image(
    patient_image: bytes, content_type: str | None = None
) -> dict:
    """
    Evaluate a patient's copy of the MMSE figure against the trusted server
    reference. Returns the normalized structured result:

        {"correct": bool, "score": 0|1, "confidence": 0..1, "reason": str,
         "review_required": bool}

    Raises VisionImageError for bad uploads and VisionProviderError for
    provider failures. Patient images are processed in memory only.
    """
    patient_data_url = prepare_patient_image(patient_image, content_type)
    reference_data_url = prepare_reference_figure()

    provider = get_provider()
    messages = provider.build_messages(reference_data_url, patient_data_url)

    t0 = time.perf_counter()
    content = provider.call(messages)
    try:
        result = parse_vision_result(content)
    except ValueError as exc:
        raise VisionProviderError(
            "invalid", "Vision assessment returned an invalid result."
        ) from exc
    elapsed = time.perf_counter() - t0

    result["review_required"] = (
        result["confidence"] < AI_CONFIDENCE_REVIEW_THRESHOLD
    )

    # Development-side timing only (backend log; never shown in the UI).
    print(
        f"[vision_eval] provider={provider.name} model={provider.model} "
        f"elapsed={elapsed:.2f}s correct={result['correct']} "
        f"confidence={result['confidence']} review_required={result['review_required']}",
        flush=True,
    )
    return result
