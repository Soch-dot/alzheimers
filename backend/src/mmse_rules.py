"""
Deterministic MMSE evaluation rules (no AI provider involved).

Only stdlib is used. The current server-side date/time is authoritative for
the five Orientation to Time items:

    orientation_time.year   -> current year (digits or English words)
    orientation_time.season -> current season (case-insensitive, "autumn"/"fall")
    orientation_time.date   -> current day of month (1-31, digits/ordinal/words)
    orientation_time.day    -> current weekday (case-insensitive, full/abbrev)
    orientation_time.month  -> current month (case-insensitive, full/abbrev)

Every response is normalized so equivalent representations are accepted
(e.g. "2026"/"twenty twenty-six"/"two thousand twenty-six"; "13th"/"thirteen").
Responses that cannot be matched score incorrect — never a per-item error — so
these items are fully deterministic and never depend on the AI provider.
"""

import re
from datetime import datetime

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
    digits = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text)
    if digits.isdigit():
        return int(digits)
    return _words_to_int(_tokenize(text))


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
# Public evaluator
# ---------------------------------------------------------------------------
def evaluate_orientation_time(item_key: str, response: str, now: datetime | None = None) -> dict | None:
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
