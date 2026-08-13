"""
AI-assisted MMSE response evaluation service (research prototype).

Layering:  React frontend  ->  this FastAPI service  ->  AI provider.

BATCH workflow (NOT per-question):
  The frontend collects ALL patient responses first (no AI calls while the
  patient is answering). The examiner then triggers ONE explicit "Assess MMSE
  with AI" action, which sends a single POST /mmse/evaluate containing every
  collected response. The backend evaluates each applicable item and returns
  per-item structured results. Internally this service may make multiple model
  evaluations (each item needs its own question-specific prompt), but the
  frontend always behaves as one assessment operation.

Provider selection (env var `AI_PROVIDER`):
  - "ollama"  (default, local): Ollama via its HTTP API, e.g. Gemma 3.
  - "gemini"  (cloud): Gemini through its OpenAI-compatible API.
  - "none"    : AI disabled; POST /mmse/evaluate returns 503.

Provider config comes from the environment (optionally a backend .env):
  AI_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, GEMINI_BASE_URL,
  GEMINI_API_KEY, GEMINI_MODEL, AI_TIMEOUT.

Security/scope rules:
  - API keys exist ONLY on the backend. They are never shipped to React.
  - The AI result is an assist signal, NOT a diagnosis and NOT clinically
    validated. The examiner always has override/manual-review control.
  - No patient responses are logged or persisted by this service.
  - Only stdlib is used here (urllib + concurrent.futures) to avoid new
    dependencies.

Each MMSE section defines its own evaluation prompt and rules.
"""

from datetime import datetime
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Provider configuration (backend-only secrets)
# ---------------------------------------------------------------------------
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
).rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "30"))

AI_SECTIONS = {
    "orientation_time",
    "orientation_place",
    "registration",
    "attention_serial7",
    "attention_spell_world",
    "delayed_recall",
    "naming",
    "repetition",
    "writing",
}


class MMSEBatchItem(BaseModel):
    """One MMSE item inside a batch request."""

    question: str = ""
    response: str = ""
    expected: str = ""


class MMSEEvaluateRequest(BaseModel):
    """
    All collected AI-scored MMSE responses in one request.

    Each key is "<section>.<item_key>", e.g. "orientation_time.year" or
    "attention_spell_world.3". `expected` carries the examiner-only evaluation
    context (empty where the answer is derived server-side, e.g. time items).
    """

    items: dict[str, MMSEBatchItem] = {}


# ---------------------------------------------------------------------------
# Prompt construction (question-specific rules per section)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an assistant that evaluates single responses from the Mini-Mental "
    "State Examination (MMSE), a cognitive screening instrument. You are NOT "
    "making a diagnosis and you have no clinical authority. Judge only whether "
    "the patient's response matches the expected answer for the one item given. "
    'Respond with ONLY a single JSON object matching exactly this schema: '
    '{"correct": boolean, "score": 0 or 1, "confidence": number between 0 and 1, '
    '"reason": short string}. score must be 1 when correct is true and 0 when '
    "correct is false. Set confidence below 0.7 whenever you are genuinely "
    "uncertain. Do not include any text outside the JSON object."
)


def _orientation_time_expected(item_key: str) -> str:
    """Date-derived expected answers are computed server-side (authoritative)."""
    now = datetime.now()
    if item_key == "year":
        return str(now.year)
    if item_key == "season":
        month = now.month
        if month in (3, 4, 5):
            return "spring"
        if month in (6, 7, 8):
            return "summer"
        if month in (9, 10, 11):
            return "fall (autumn)"
        return "winter"
    if item_key == "date":
        return str(now.day)
    if item_key == "day":
        return now.strftime("%A")
    if item_key == "month":
        return now.strftime("%B")
    return item_key


def _prompt_for(section: str, item_key: str, question: str, response: str, expected: str) -> str:
    resp = response.strip()
    q = question.strip() or f"({item_key})"

    if section == "orientation_time":
        exp = _orientation_time_expected(item_key)
        return (
            f"MMSE — Orientation to Time, item \"{item_key}\". Question: \"{q}\". "
            f"Expected answer: \"{exp}\". The patient responded: \"{resp}\". "
            "Score 1 only if the response is equivalent to the expected answer. "
            'Accept reasonable formats (e.g. "2026" or "two thousand twenty-six" '
            'for the year; a season name such as "fall" or "autumn" for the season).'
        )
    if section == "orientation_place":
        return (
            f"MMSE — Orientation to Place, item \"{item_key}\". Question: \"{q}\". "
            f"Expected answer: \"{expected}\". The patient responded: \"{resp}\". "
            "Score 1 if the response reasonably corresponds to the expected answer "
            "(accept abbreviations and equivalent forms)."
        )
    if section in ("registration", "delayed_recall"):
        return (
            f"MMSE — {'Registration' if section == 'registration' else 'Delayed Recall'} "
            f"object #{item_key}. The examiner presented the object \"{expected}\". "
            f"The patient responded: \"{resp}\". "
            "Score 1 if the patient's response clearly refers to the same object "
            "(accept synonyms and slight wording differences)."
        )
    if section == "attention_serial7":
        return (
            f"MMSE — Serial-7s subtraction step {item_key}. Expected number: "
            f"\"{expected}\". The patient said: \"{resp}\". "
            "Score 1 only if the number is exactly the expected value. Do not "
            "accept approximations or spelling-out of a different number."
        )
    if section == "attention_spell_world":
        return (
            f"MMSE — spelling WORLD backwards, letter position {item_key}. "
            f"Expected letter: \"{expected}\". The patient provided: \"{resp}\" "
            "for this position. Score 1 only if the letter matches exactly "
            "(case-insensitive)."
        )
    if section == "naming":
        return (
            f"MMSE — Naming. The examiner showed/presented: \"{expected}\". "
            f"The patient said: \"{resp}\". "
            "Score 1 if the patient correctly named the object. Accept common "
            'synonyms (e.g. "wristwatch"/"watch", "pen"/"pencil" only if the '
            "same object)."
        )
    if section == "repetition":
        return (
            "MMSE — Repetition. Required phrase: \"No ifs, ands, or buts.\" "
            f"The patient repeated: \"{resp}\". "
            "Score 1 if the response sufficiently matches the required phrase. "
            "Allow reasonable case, punctuation and minor wording differences. "
            "If you are not confident, set a LOW confidence so the examiner "
            "reviews it rather than confidently scoring it incorrect."
        )
    if section == "writing":
        return (
            "MMSE — Writing. The stated scoring criterion is only: the sentence "
            "must contain a noun and a verb. "
            f"The patient wrote: \"{resp}\". "
            "Score 1 only if the sentence contains at least one noun and one "
            "verb. Do NOT judge spelling, grammar, handwriting quality, or "
            "intelligence. If you are genuinely uncertain whether a noun and a "
            "verb are both present, set a LOW confidence."
        )
    return (
        f"MMSE item \"{item_key}\". Question: \"{q}\". Expected answer: "
        f"\"{expected}\". The patient responded: \"{resp}\". "
        "Score 1 if the response matches or reasonably corresponds to the "
        "expected answer."
    )


def build_messages(
    section: str, item_key: str, question: str, response: str, expected: str
) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _prompt_for(
            section, item_key, question, response, expected
        )},
    ]


# ---------------------------------------------------------------------------
# Provider callers (stdlib only)
# ---------------------------------------------------------------------------
class AIProviderError(Exception):
    pass


def _post_json(url: str, payload: dict, headers: dict) -> str:
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=AI_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:500]
        except Exception:
            pass
        raise AIProviderError(f"provider HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AIProviderError(f"cannot reach provider: {exc.reason}") from exc
    except TimeoutError as exc:
        raise AIProviderError("provider timed out") from exc
    return body


def _call_ollama(messages: list) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    body = _post_json(f"{OLLAMA_BASE_URL}/api/chat", payload, {})
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AIProviderError("Ollama returned non-JSON") from exc
    content = data.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise AIProviderError("Ollama returned an empty response")
    return content


def _call_gemini(messages: list) -> str:
    if not GEMINI_API_KEY:
        raise AIProviderError("GEMINI_API_KEY is not set on the backend")
    payload = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    body = _post_json(
        f"{GEMINI_BASE_URL}/chat/completions",
        payload,
        {"Authorization": f"Bearer {GEMINI_API_KEY}"},
    )
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AIProviderError("Gemini returned non-JSON") from exc
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise AIProviderError("Gemini returned an empty response")
    return content


def call_provider(messages: list) -> str:
    if AI_PROVIDER == "ollama":
        return _call_ollama(messages)
    if AI_PROVIDER == "gemini":
        return _call_gemini(messages)
    raise AIProviderError(f"unsupported AI_PROVIDER: {AI_PROVIDER}")


# ---------------------------------------------------------------------------
# Structured-output validation
# ---------------------------------------------------------------------------
def parse_ai_result(content: str) -> dict:
    """Validate the model's JSON output. Raise ValueError on anything invalid."""
    text = content.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    def _load(raw: str) -> dict:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("model output is not a JSON object")
        return data

    try:
        data = _load(text)
    except (json.JSONDecodeError, ValueError):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("model did not return valid JSON")
        try:
            data = _load(text[start : end + 1])
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("model did not return valid JSON") from exc

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
# Public entry point used by api.py (batch evaluation)
# ---------------------------------------------------------------------------
class _ItemFailure:
    """Per-item failure. kind: "provider" (AI unreachable) or "invalid" (bad output)."""

    __slots__ = ("kind", "message")

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        self.message = message


def _evaluate_single(key: str, entry: MMSEBatchItem):
    """Evaluate one item. Returns a validated result dict or an _ItemFailure."""
    section, sep, item_key = key.partition(".")
    if not sep or not item_key:
        return _ItemFailure("invalid", f"Invalid item key: {key!r}")
    if section not in AI_SECTIONS:
        return _ItemFailure(
            "invalid", f"Unsupported MMSE section for AI evaluation: {section}"
        )
    if not entry.response.strip():
        return _ItemFailure("invalid", f"Empty response for {key} cannot be evaluated")

    messages = build_messages(
        section, item_key, entry.question, entry.response, entry.expected
    )
    try:
        content = call_provider(messages)
        return parse_ai_result(content)
    except AIProviderError as exc:
        return _ItemFailure("provider", str(exc))
    except ValueError as exc:
        return _ItemFailure("invalid", str(exc))


def evaluate_mmse_batch(req: MMSEEvaluateRequest) -> dict:
    """
    Evaluate a batch of MMSE responses.

    Returns {"items": {key: result}, "errors": {key: message}}. Raises:
      503 when AI is disabled or the provider is unreachable for every item;
      422 when the request itself is malformed (no items supplied).
    """
    if AI_PROVIDER == "none":
        raise HTTPException(
            status_code=503,
            detail="AI assessment is not configured (AI_PROVIDER=none).",
        )
    if not req.items:
        raise HTTPException(status_code=422, detail="No MMSE items to evaluate.")

    keys = list(req.items.keys())
    results: dict = {}
    errors: dict = {}
    provider_failures = 0

    workers = min(len(keys), 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_evaluate_single, key, req.items[key]): key for key in keys
        }
        for future in as_completed(futures):
            key = futures[future]
            outcome = future.result()
            if isinstance(outcome, dict):
                results[key] = outcome
            else:
                errors[key] = outcome.message
                if outcome.kind == "provider":
                    provider_failures += 1

    if not results and errors and provider_failures == len(errors):
        raise HTTPException(
            status_code=503, detail=f"AI assessment unavailable: {next(iter(errors.values()))}"
        )

    return {"items": results, "errors": errors}