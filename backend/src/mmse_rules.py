"""
Deterministic MMSE evaluation rules (no AI provider involved).

Only stdlib is used. Sections fall into two groups:

1. FULLY deterministic — never reach the AI provider:

   - orientation_time.*          current server date/time is authoritative
   - attention_serial7.1..5      arithmetic sequence 93, 86, 79, 72, 65
   - attention_spell_world.1..5  fixed letter sequence D, L, R, O, W

2. HYBRID — scored deterministically when the answer is safely provable;
   otherwise the item is routed to the AI provider (return the AMBIGUOUS
   sentinel):

   - orientation_place.*   compared to the examiner-configured assessment
                           location (normalized capitalization, whitespace,
                           harmless articles; NO invented synonyms)
   - registration.1..3     configured objects (Apple, Table, Penny)
   - delayed_recall.1..3   the SAME configured objects (recall)
   - naming.*              configured objects incl. watch/wristwatch
   - repetition.*          required phrase "No ifs, ands, or buts."

Every response is normalized so harmless variations are accepted
(e.g. "2026"/"twenty twenty-six"; "an apple"; "watch" -> wristwatch).
Clearly different responses score incorrect — never a per-item error. Only
genuinely ambiguous responses (uncertain, referential, generic, or partial)
are routed to AI, so a normal clean MMSE sends almost nothing to the model.
"""

import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Expected sequences (index 0 -> item 1). One point per item.
# ---------------------------------------------------------------------------
SERIAL_7_EXPECTED = (93, 86, 79, 72, 65)
SPELL_WORLD_EXPECTED = ("D", "L", "R", "O", "W")

# Hybrid evaluators return this sentinel when the item cannot be scored safely
# and should be sent to the AI provider instead.
AMBIGUOUS = object()
MATCH = object()
NO_MATCH = object()

# ---------------------------------------------------------------------------
# English number-word parsing (cardinal and ordinal forms)
# ---------------------------------------------------------------------------
_NUMBERS = {
    # cardinal
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    # ordinal
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
    "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80,
    "ninetieth": 90,
}

_SEASONS = {"spring", "summer", "fall", "autumn", "winter"}

_DAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _normalize(text: str) -> str:
    """Lowercase, trim and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower()).strip()


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens (hyphens and commas become spaces)."""
    return _normalize(text.replace("-", " ").replace(",", " ")).split()


def _words_to_int(words: list[str]) -> int | None:
    """Additive English number parser (cardinal/ordinal forms). None on failure."""
    total = 0
    current = 0
    for word in words:
        if word in _NUMBERS:
            current += _NUMBERS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word == "thousand":
            total += (current or 1) * 1000
            current = 0
        elif word in ("and",):
            continue
        else:
            return None
    return total + current


def _parse_number(text: str) -> int | None:
    """Parse a plain number: digits, ordinal digits, or English words."""
    if not text:
        return None
    t = _normalize(text)
    digits = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", t)
    if digits.isdigit():
        return int(digits)
    return _words_to_int(_tokenize(t))


def _parse_year(text: str) -> int | None:
    """
    Parse a year from digits or English words.

    Handles "2026", "twenty twenty-six" (century-pair form -> 2026),
    "two thousand twenty-six", "nineteen ninety-nine" -> 1999, etc.
    """
    if not text:
        return None
    t = _normalize(text)
    if t.isdigit():
        return int(t)
    words = _tokenize(text)
    if not words:
        return None
    if "thousand" in words:
        return _words_to_int(words)
    # Century-pair form: "twenty twenty six" -> 20 * 100 + 26.
    for split in range(1, len(words)):
        left = _words_to_int(words[:split])
        right = _words_to_int(words[split:])
        if left is not None and right is not None and left < 100 and right < 100:
            return left * 100 + right
    return _words_to_int(words)


def _parse_date(text: str) -> int | None:
    """Parse the day-of-month (1-31) from digits, ordinal digits or words."""
    if not text:
        return None
    t = _normalize(text)
    t = re.sub(r"^(the\s+)?", "", t)
    value = _parse_number(t)
    if value is not None and 1 <= value <= 31:
        return value
    return None


def _season_for(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _parse_season(text: str) -> str | None:
    season = _normalize(text)
    if season == "autumn":
        season = "fall"
    return season if season in _SEASONS else None


def _parse_day(text: str) -> int | None:
    return _DAYS.get(_normalize(text))


def _parse_month(text: str) -> int | None:
    """Parse a month name, tolerating leading "the month of" / "month of"."""
    t = _normalize(text)
    t = re.sub(r"^(the\s+)?(month\s+of\s+)?", "", t)
    return _MONTHS.get(t)


# ---------------------------------------------------------------------------
# Safe text normalization for object/phrase matching (hybrid sections)
# ---------------------------------------------------------------------------
_ARTICLES = ("a ", "an ", "the ")

_GENERIC_WORDS = {
    "fruit", "vegetable", "food", "plant", "animal", "thing", "object",
    "item", "stuff", "something", "anything", "one", "it", "this", "that",
    "these", "those", "kind", "sort", "type", "piece", "part", "shape",
    "colour", "color",
}

_UNCERTAIN_PHRASES = (
    "don't know", "dont know", "do not know", "not sure", "no idea",
    "i don't know", "i do not know", "i don't remember", "i do not remember",
    "can't remember", "cant remember", "cannot remember", "dunno", "unknown",
    "forgot", "never mind",
)

# Bounded, deliberate semantic equivalents (per the naming criterion):
# a wristwatch item accepts "watch" / "wrist watch". Nothing else is invented.
_EQUIVALENTS = {
    "wristwatch": {"watch", "wrist watch"},
}


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = re.sub(r"['\"]", "", text or "")
    s = re.sub(r"[.,;:!?()\[\]{}]+", " ", s)
    return re.sub(r"\s+", " ", s.strip().lower()).strip()


def _strip_articles(text: str) -> str:
    """Remove leading harmless articles ("a", "an", "the") repeatedly."""
    s = (text or "").strip()
    while True:
        stripped = False
        for article in _ARTICLES:
            if s.startswith(article):
                s = s[len(article):].strip()
                stripped = True
        if not stripped:
            break
    return s


def _is_uncertain(text: str) -> bool:
    t = _normalize_text(text)
    return any(phrase in t for phrase in _UNCERTAIN_PHRASES)


def _is_generic_or_referential(text: str) -> bool:
    """True when the response names only a generic category / placeholder."""
    words = _strip_articles(_normalize_text(text)).split()
    if not words:
        return True
    return any(word in _GENERIC_WORDS for word in words)


def _object_verdict(response: str, expected: str):
    """
    Match a response against a configured object/location name.

    Returns MATCH / NO_MATCH / AMBIGUOUS. Only safe variations are accepted:
    capitalization, whitespace, punctuation, harmless articles, the expected
    keywords appearing as whole words, and the bounded watch/wristwatch
    equivalence. Uncertain, generic/referential, or partially-overlapping
    responses are AMBIGUOUS (the caller routes them to AI).
    """
    resp = _normalize_text(response)
    exp = _normalize_text(expected)
    if not exp:
        return AMBIGUOUS  # not configured -> keep old AI behavior
    if not resp:
        return AMBIGUOUS  # empty (defensive; frontend never sends it)

    if resp == exp:
        return MATCH
    resp_n = _strip_articles(resp)
    exp_n = _strip_articles(exp)
    if resp_n == exp_n:
        return MATCH
    if exp_n in _EQUIVALENTS and resp_n in _EQUIVALENTS[exp_n]:
        return MATCH

    exp_words = exp_n.split()
    resp_words = resp_n.split()
    if exp_words and all(word in resp_words for word in exp_words):
        return MATCH

    if _is_uncertain(resp) or _is_generic_or_referential(resp_n):
        return AMBIGUOUS
    if set(exp_words) & set(resp_words):
        return AMBIGUOUS  # partial overlap -> genuinely ambiguous
    return NO_MATCH


def _repetition_verdict(response: str, expected: str):
    """Match the repetition phrase. Exact normalized match wins; partial /
    uncertain responses are ambiguous; unrelated responses are incorrect."""
    resp = _normalize_text(response)
    exp = _normalize_text(expected)
    if not resp or not exp:
        return AMBIGUOUS
    if resp == exp:
        return MATCH
    if _is_uncertain(resp):
        return AMBIGUOUS
    if set(resp.split()) & set(exp.split()):
        return AMBIGUOUS  # partial repetition -> let AI decide
    return NO_MATCH


def _item_position(item_key: str, size: int) -> int | None:
    """1-indexed item key -> 0-based position (key "0" accepted as item 1)."""
    key = (item_key or "").strip()
    try:
        index = int(key)
    except (TypeError, ValueError):
        return None
    if index == 0:
        return 0
    if 1 <= index <= size:
        return index - 1
    return None


def _result(correct: bool, reason: str) -> dict:
    return {"correct": correct, "score": 1 if correct else 0, "confidence": 1.0, "reason": reason}


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------
def evaluate_orientation_time(
    item_key: str, response: str, expected: str = "", now: datetime | None = None
) -> dict | None:
    """
    Deterministically score one Orientation to Time item against the current
    server-side date/time. Returns a validated result dict (same schema as the
    AI path, but with confidence 1.0) or None for an unknown item key.

    Responses that cannot be matched to an expected value score `correct: False`
    with a reason — they never become per-item errors. The 5 known keys always
    return a result; only unknown keys return None.
    """
    key = (item_key or "").strip().lower()
    resp = (response or "").strip()
    clock = now or datetime.now()

    if key == "year":
        expected = clock.year
        value = _parse_year(resp)
        correct = value == expected
        reason = f"Matches the current year ({expected})." if correct else f"Expected {expected}."
    elif key == "season":
        expected = _season_for(clock.month)
        value = _parse_season(resp)
        correct = value == expected
        reason = f"Matches the current season ({expected})." if correct else f"Expected {expected}."
    elif key == "date":
        expected = clock.day
        value = _parse_date(resp)
        correct = value == expected
        reason = f"Matches the current date ({expected})." if correct else f"Expected {expected}."
    elif key == "day":
        expected = clock.weekday()
        value = _parse_day(resp)
        correct = value == expected
        label = clock.strftime("%A")
        reason = f"Matches the current day ({label})." if correct else f"Expected {label}."
    elif key == "month":
        expected = clock.month
        value = _parse_month(resp)
        correct = value == expected
        label = clock.strftime("%B")
        reason = f"Matches the current month ({label})." if correct else f"Expected {label}."
    else:
        return None

    return {
        "correct": correct,
        "score": 1 if correct else 0,
        "confidence": 1.0,
        "reason": reason,
    }


def evaluate_attention_serial7(item_key: str, response: str, expected: str = "") -> dict | None:
    """
    Deterministically score one Serial-7s item (attention_serial7.1..5).

    The expected sequence is 93, 86, 79, 72, 65 — pure arithmetic, so no AI
    provider is involved. The patient's answer is parsed as a number (digits,
    surrounding whitespace, or English number words such as "ninety-three")
    and compared exactly against the expected value.

    The frontend sends 1-indexed keys (`attention_serial7.1` .. `attention_serial7.5`
    mapping to SERIAL_7_EXPECTED positions). Key "0" is accepted defensively as
    the first item. Returns a validated result dict (confidence always 1.0) or
    None for an unknown item key. Unparseable responses score incorrect (never
    a per-item error, never sent to AI).
    """
    key = (item_key or "").strip()
    resp = (response or "").strip()

    try:
        index = int(key)
    except (TypeError, ValueError):
        return None
    if index == 0:
        position = 0
    elif 1 <= index <= len(SERIAL_7_EXPECTED):
        position = index - 1
    else:
        return None

    expected = SERIAL_7_EXPECTED[position]
    value = _parse_number(resp)
    correct = value is not None and value == expected
    reason = (
        f"Matches expected serial-7 result ({expected})."
        if correct
        else f"Expected {expected}."
    )
    return {
        "correct": correct,
        "score": 1 if correct else 0,
        "confidence": 1.0,
        "reason": reason,
    }


def evaluate_attention_spell_world(item_key: str, response: str, expected: str = "") -> dict | None:
    """
    Deterministically score one WORLD-backwards letter (attention_spell_world.1..5).

    The expected sequence is D, L, R, O, W. Each frontend item carries a single
    letter; harmless case differences are accepted. This is fully deterministic
    — never ambiguous, never sent to AI. Unknown keys return None.
    """
    position = _item_position(item_key, len(SPELL_WORLD_EXPECTED))
    if position is None:
        return None
    exp_letter = (expected or "").strip().upper() or SPELL_WORLD_EXPECTED[position]
    resp_letters = re.sub(r"[^A-Za-z]", "", (response or "")).upper()
    # The frontend sends one letter per item; if a longer string is supplied,
    # compare the letter at this item's position (e.g. "DLROW" -> item 4 = "O").
    resp_letter = resp_letters[position] if len(resp_letters) > 1 else resp_letters
    correct = bool(resp_letter) and resp_letter == exp_letter
    reason = (
        f"Matches expected letter ({exp_letter})."
        if correct
        else f"Expected {exp_letter}."
    )
    return _result(correct, reason)


def evaluate_orientation_place(item_key: str, response: str, expected: str = "") -> dict:
    """
    Deterministically score one Orientation-to-Place item when the assessment
    location is configured (expected non-empty). Accepts only safe variations:
    capitalization, whitespace, punctuation, harmless articles. Clearly
    different answers are incorrect. Unconfigured, uncertain, generic or
    partially-overlapping responses return the AMBIGUOUS sentinel (route to AI).
    """
    verdict = _object_verdict(response, expected)
    if verdict is AMBIGUOUS:
        return AMBIGUOUS
    exp = _normalize_text(expected)
    correct = verdict is MATCH
    reason = (
        f"Matches configured assessment location ({exp})."
        if correct
        else f"Expected {exp}."
    )
    return _result(correct, reason)


def evaluate_registration(item_key: str, response: str, expected: str = "") -> dict:
    """Deterministically score one Registration object (configured objects)."""
    if _item_position(item_key, 3) is None:
        return None
    verdict = _object_verdict(response, expected)
    if verdict is AMBIGUOUS:
        return AMBIGUOUS
    exp = _normalize_text(expected)
    correct = verdict is MATCH
    reason = (
        f"Matches expected object ({exp})."
        if correct
        else f"Expected {exp}."
    )
    return _result(correct, reason)


def evaluate_delayed_recall(item_key: str, response: str, expected: str = "") -> dict:
    """Deterministically score one Delayed Recall object (SAME configured
    objects as Registration — no duplicate object configuration here)."""
    if _item_position(item_key, 3) is None:
        return None
    verdict = _object_verdict(response, expected)
    if verdict is AMBIGUOUS:
        return AMBIGUOUS
    exp = _normalize_text(expected)
    correct = verdict is MATCH
    reason = (
        f"Matches expected object ({exp})."
        if correct
        else f"Expected {exp}."
    )
    return _result(correct, reason)


def evaluate_naming(item_key: str, response: str, expected: str = "") -> dict:
    """Deterministically score one Naming item (wristwatch/pencil) using the
    configured object plus the bounded watch/wristwatch equivalence."""
    key = (item_key or "").strip().lower()
    if key not in ("wristwatch", "pencil"):
        return None
    verdict = _object_verdict(response, expected)
    if verdict is AMBIGUOUS:
        return AMBIGUOUS
    exp = _normalize_text(expected)
    correct = verdict is MATCH
    reason = (
        f"Matches expected object ({exp})."
        if correct
        else f"Expected {exp}."
    )
    return _result(correct, reason)


def evaluate_repetition(item_key: str, response: str, expected: str = "") -> dict:
    """Deterministically score the Repetition phrase (normalized: case,
    punctuation, repeated whitespace). Exact normalized match is correct;
    unrelated responses are incorrect; partial/uncertain are ambiguous."""
    verdict = _repetition_verdict(response, expected)
    if verdict is AMBIGUOUS:
        return AMBIGUOUS
    correct = verdict is MATCH
    reason = (
        "Matches the required phrase."
        if correct
        else "Expected the required phrase."
    )
    return _result(correct, reason)
