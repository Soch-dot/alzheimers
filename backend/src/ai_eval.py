"""
AI-assisted MMSE response evaluation service (research prototype).

Layering:  React frontend  ->  this FastAPI service  ->  AI provider.

BATCH workflow (NOT per-question):
  The frontend collects ALL patient responses first (no AI calls while the
  patient is answering). The examiner then triggers ONE explicit "Assess MMSE
  with AI" action, which sends a single POST /mmse/evaluate containing every
  collected response. The backend evaluates all applicable items and returns
  per-item structured results. The frontend always behaves as one assessment
  operation.

Provider selection (env var `AI_PROVIDER`):
  - "ollama"  (default, local): Ollama via its HTTP API, e.g. Gemma 3.
  - "gemini"  (cloud): Gemini through its OpenAI-compatible API.
  - "none"    : AI disabled; POST /mmse/evaluate returns 503.

Batching strategy is provider-aware:
  - OLLAMA: the whole batch is sent as ONE `/api/chat` call with a single
    prompt containing every AI-evaluable item and its section-specific rules.
    The model returns a single JSON object with one entry per item. This avoids
    30 sequential local generations on a single GPU/CPU (the previous
    bottleneck). No per-item Ollama requests and no concurrent Ollama requests.
  - GEMINI: keeps the previous per-item parallel evaluation for now, but the
    provider abstraction is structured so a single batched call can be added
    later without touching the caller.

Provider config comes from the environment (optionally a backend .env):
  AI_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, GEMINI_BASE_URL,
  GEMINI_API_KEY, GEMINI_MODEL, AI_TIMEOUT, OLLAMA_BATCH_TIMEOUT,
  OLLAMA_MAX_CONCURRENCY, GEMINI_MAX_CONCURRENCY.

Security/scope rules:
  - API keys exist ONLY on the backend. They are never shipped to React.
  - The AI result is an assist signal, NOT a diagnosis and NOT clinically
    validated. The examiner always has override/manual-review control.
  - Reference/expected answers are backend-only (used inside prompts for the
    model). They are never shown to the patient or shipped to React.
  - No patient responses are logged or persisted by this service.
  - Only stdlib is used here (urllib + concurrent.futures) to avoid new
    dependencies.

Each MMSE section defines its own evaluation prompt and rules.
"""

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel

from src.mmse_rules import (
    evaluate_attention_serial7,
    evaluate_orientation_time,
)

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
# The single Ollama batch call must fit inside the frontend's batch timeout
# (180 s) with margin; it is separate from the per-item AI_TIMEOUT used by the
# Gemini path.
OLLAMA_BATCH_TIMEOUT = float(os.getenv("OLLAMA_BATCH_TIMEOUT", "175"))
OLLAMA_MAX_CONCURRENCY = max(1, int(os.getenv("OLLAMA_MAX_CONCURRENCY", "1")))
GEMINI_MAX_CONCURRENCY = max(1, int(os.getenv("GEMINI_MAX_CONCURRENCY", "8")))

# Sections scored deterministically on the backend (no AI provider). These are
# excluded from AI_SECTIONS so they never reach the model.
DETERMINISTIC_SECTIONS = {"orientation_time", "attention_serial7"}

# Deterministic evaluator per section: (item_key, response) -> result dict|None.
DETERMINISTIC_EVALUATORS = {
    "orientation_time": evaluate_orientation_time,
    "attention_serial7": evaluate_attention_serial7,
}

AI_SECTIONS = {
    "orientation_place",
    "registration",
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
    "Evaluate the MEANING of the response, not its formatting: do NOT mark a "
    "response incorrect solely because of differences in capitalization, "
    "punctuation, spacing, a number written in words instead of digits (e.g. "
    "'2026' and 'two thousand twenty-six'), or a harmless synonym. Mark it "
    "incorrect only when the response does not actually satisfy the question. "
    'Respond with ONLY a single JSON object matching exactly this schema: '
    '{"correct": boolean, "score": 0 or 1, "confidence": number between 0 and 1, '
    '"reason": short string}. score must be 1 when correct is true and 0 when '
    "correct is false. Set confidence below 0.7 whenever you are genuinely "
    "uncertain. Do not include any text outside the JSON object."
)


BATCH_SYSTEM_PROMPT = (
    "You are an assistant that evaluates responses from the Mini-Mental State "
    "Examination (MMSE), a cognitive screening instrument. You are NOT making a "
    "diagnosis and you have no clinical authority. You will be given several "
    "MMSE items; judge each one independently on whether the patient's response "
    "matches its expected answer. Evaluate the MEANING of each response, not its "
    "formatting: do NOT mark a response incorrect solely because of differences "
    "in capitalization, punctuation, spacing, a number written in words instead "
    "of digits (e.g. '2026' and 'twenty twenty-six'), or a harmless synonym. "
    "Mark an item incorrect only when the response does not actually satisfy "
    "the question. "
    'Respond with ONLY a single JSON object matching exactly this schema: '
    '{"items": { "<item_key>": {"correct": boolean, "score": 0 or 1, '
    '"confidence": number between 0 and 1, "reason": short string} } }. '
    "Include an entry for EVERY item key listed below. score must be 1 when "
    "correct is true and 0 when correct is false. Set confidence below 0.7 "
    "whenever you are genuinely uncertain. Keep every reason short (a few "
    "words). Do not include any text outside the JSON object."
)


def _prompt_for(section: str, item_key: str, question: str, response: str, expected: str) -> str:
    resp = response.strip()
    q = question.strip() or f"({item_key})"

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


def build_batch_messages(items: "dict[str, MMSEBatchItem]") -> list:
    """
    Build ONE message pair evaluating all items in a single model call.

    Each item keeps its section-specific rules (from `_prompt_for`), so the
    single call preserves the same question-specific semantics as per-item
    evaluation while generating all results in one forward pass.
    """
    lines = []
    for key, entry in items.items():
        section, sep, item_key = key.partition(".")
        rule = _prompt_for(
            section, item_key, entry.question, entry.response, entry.expected
        )
        lines.append(f"[{key}]\n{rule}")
    user_content = (
        "Evaluate each of the following MMSE items and return one JSON entry "
        "per item key:\n\n" + "\n\n".join(lines)
    )
    return [
        {"role": "system", "content": BATCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Provider callers (stdlib only)
# ---------------------------------------------------------------------------
class AIProviderError(Exception):
    pass


def _post_json(url: str, payload: dict, headers: dict, timeout: float | None = None) -> str:
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    effective_timeout = AI_TIMEOUT if timeout is None else timeout
    try:
        with urllib.request.urlopen(request, timeout=effective_timeout) as resp:
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


def _call_ollama(messages: list, timeout: float | None = None) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    body = _post_json(f"{OLLAMA_BASE_URL}/api/chat", payload, {}, timeout=timeout)
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
def _parse_json_object(text: str) -> dict:
    """Best-effort parse of the model's JSON output. Raise ValueError on failure."""
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
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


def _validate_item(data: dict) -> dict:
    """Validate one item result dict. Raise ValueError on anything invalid."""
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


def parse_ai_result(content: str) -> dict:
    """Validate the model's per-item JSON output. Raise ValueError on anything invalid."""
    return _validate_item(_parse_json_object(content))


def parse_batch_result(content: str, requested_keys: list[str]) -> tuple[dict, dict]:
    """
    Validate a single-call batch output: {"items": {key: result}}.

    Returns (valid_results, errors). Each requested key is validated
    independently: a valid entry is scored, while a missing or malformed entry
    becomes a per-item error. Missing items are NEVER silently scored.
    """
    try:
        data = _parse_json_object(content)
    except ValueError as exc:
        return {}, {k: f"Invalid AI batch output: {exc}" for k in requested_keys}
    items = data.get("items")
    if not isinstance(items, dict):
        # The model sometimes omits the "items" wrapper and returns a flat
        # {key: result} map directly. Accept that shape too.
        if data and all(isinstance(v, dict) for v in data.values()):
            items = data
        else:
            return {}, {k: "Invalid AI batch output: no 'items' object" for k in requested_keys}

    results: dict = {}
    errors: dict = {}
    for key in requested_keys:
        if key not in items:
            errors[key] = f"Model did not return a result for item {key!r}"
            continue
        entry = items[key]
        if not isinstance(entry, dict):
            errors[key] = f"Invalid result for item {key!r}: not a JSON object"
            continue
        try:
            results[key] = _validate_item(entry)
        except ValueError as exc:
            errors[key] = f"Invalid result for item {key!r}: {exc}"
    return results, errors


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


def _max_concurrency() -> int:
    """Provider-aware per-item concurrency for the Gemini path."""
    if AI_PROVIDER == "gemini":
        return GEMINI_MAX_CONCURRENCY
    return OLLAMA_MAX_CONCURRENCY


def _evaluate_ollama_batch(req: MMSEEvaluateRequest) -> tuple[dict, dict, int]:
    """
    Single-call Ollama batch evaluation: ONE /api/chat generation for the whole
    batch. Returns (results, errors, provider_failures). provider_failures counts
    items that failed because the provider itself was unreachable.
    """
    keys = list(req.items.keys())
    results: dict = {}
    errors: dict = {}
    provider_failures = 0

    # Items with invalid keys / empty responses are rejected client-side, but
    # guard server-side too so the model is never asked about them.
    valid_keys: list[str] = []
    for key in keys:
        section, sep, item_key = key.partition(".")
        if not sep or not item_key:
            errors[key] = f"Invalid item key: {key!r}"
            continue
        if section not in AI_SECTIONS:
            errors[key] = f"Unsupported MMSE section for AI evaluation: {section}"
            continue
        if not req.items[key].response.strip():
            errors[key] = f"Empty response for {key} cannot be evaluated"
            continue
        valid_keys.append(key)

    if not valid_keys:
        return results, errors, provider_failures

    messages = build_batch_messages({k: req.items[k] for k in valid_keys})
    try:
        content = _call_ollama(messages, timeout=OLLAMA_BATCH_TIMEOUT)
        parsed, parsed_errors = parse_batch_result(content, valid_keys)
    except AIProviderError as exc:
        for key in valid_keys:
            errors[key] = str(exc)
        return results, errors, len(valid_keys)

    results.update(parsed)
    errors.update(parsed_errors)
    return results, errors, provider_failures


def evaluate_mmse_batch(req: MMSEEvaluateRequest) -> dict:
    """
    Evaluate a batch of MMSE responses.

    Deterministic sections (see DETERMINISTIC_SECTIONS) are scored server-side
    with NO AI provider call. Only the remaining items reach the provider:
    Ollama uses ONE model generation for the whole AI batch; Gemini keeps
    per-item parallel evaluation. Returns
    {"items": {key: result}, "errors": {key: message}}. Raises:
      503 when AI is disabled or the provider is unreachable for every item
          (only relevant when the request actually contains AI-evaluated items);
      422 when the request itself is malformed (no items supplied).
    """
    if not req.items:
        raise HTTPException(status_code=422, detail="No MMSE items to evaluate.")

    t0 = time.perf_counter()
    results: dict = {}
    errors: dict = {}
    provider_failures = 0

    # Split the batch: deterministic items are scored first without any AI
    # provider involvement. Only the remaining (AI-required) items are sent to
    # the model, so the actual model payload never contains deterministic keys.
    ai_items: dict[str, MMSEBatchItem] = {}
    for key, entry in req.items.items():
        section, sep, item_key = key.partition(".")
        evaluator = DETERMINISTIC_EVALUATORS.get(section)
        if evaluator is not None:
            result = evaluator(item_key, entry.response)
            if result is not None:
                results[key] = result
            else:
                errors[key] = (
                    f"Unsupported deterministic item key: {key!r}"
                )
        else:
            ai_items[key] = entry

    if ai_items:
        if AI_PROVIDER == "none":
            raise HTTPException(
                status_code=503,
                detail="AI assessment is not configured (AI_PROVIDER=none).",
            )
        ai_req = MMSEEvaluateRequest(items=ai_items)
        if AI_PROVIDER == "ollama":
            ai_results, ai_errors, ai_failures = _evaluate_ollama_batch(ai_req)
        else:
            ai_results, ai_errors, ai_failures = {}, {}, 0
            workers = min(len(ai_items), _max_concurrency())
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_evaluate_single, key, entry): key
                    for key, entry in ai_items.items()
                }
                for future in as_completed(futures):
                    key = futures[future]
                    outcome = future.result()
                    if isinstance(outcome, dict):
                        ai_results[key] = outcome
                    else:
                        ai_errors[key] = outcome.message
                        if outcome.kind == "provider":
                            ai_failures += 1
        results.update(ai_results)
        errors.update(ai_errors)
        provider_failures = ai_failures

    elapsed = time.perf_counter() - t0
    deterministic_count = sum(
        1 for key in req.items if key.partition(".")[0] in DETERMINISTIC_SECTIONS
    )
    # Development-side timing only (backend log; never shown in the UI).
    print(
        f"[ai_eval] AI_PROVIDER={AI_PROVIDER} items={len(req.items)} "
        f"(deterministic={deterministic_count}, ai={len(ai_items)}) "
        f"provider_calls={1 if AI_PROVIDER == 'ollama' and ai_items else len(ai_items)} "
        f"elapsed={elapsed:.2f}s results={len(results)} errors={len(errors)}",
        flush=True,
    )

    if ai_items and not ai_results and errors and provider_failures == len(errors):
        raise HTTPException(
            status_code=503, detail=f"AI assessment unavailable: {next(iter(errors.values()))}"
        )

    return {"items": results, "errors": errors}