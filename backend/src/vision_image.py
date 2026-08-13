"""
Image handling for the MMSE Question 11 (figure copying) vision evaluation.

Responsibilities (all in-memory; nothing is persisted):
  - validate MIME type and file size of the uploaded patient drawing
  - decode the image (Pillow)
  - normalize orientation (EXIF)
  - preserve aspect ratio
  - downscale very large images to a reasonable maximum while keeping
    enough detail for geometric evaluation
  - blank / near-blank pre-check (deterministic sanity gate BEFORE any vision
    provider call; rejects obviously unusable submissions)
  - encode images as base64 data URLs for the vision providers
  - load the trusted server-side reference figure (never the client)

Privacy: uploaded images are processed in memory and discarded. They are never
written to disk, never logged, and never returned to the client.

Only stdlib + Pillow (already a pinned backend dependency) are used.
"""

import base64
import io
import os

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("VISION_MAX_UPLOAD_BYTES", "10_000_000"))  # 10 MB
MAX_IMAGE_DIMENSION = int(os.getenv("VISION_MAX_IMAGE_DIMENSION", "2048"))
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
# Blank / near-blank pre-check thresholds (conservative sanity gate).
#
# Rationale: a legitimate patient drawing — even a faint pencil sketch — covers
# well above 0.5% of the page with strokes that are at least 20 gray levels
# darker than the paper. Pure or near-blank pages have essentially no such
# pixels. The thresholds are deliberately low so a weak-but-plausible drawing is
# NEVER rejected; only obviously unusable submissions are caught before they are
# sent to a vision provider. This gate does NOT attempt figure recognition.
VISION_BLANK_INK_DELTA = int(os.getenv("VISION_BLANK_INK_DELTA", "20"))  # gray levels below paper
VISION_BLANK_MIN_INK_FRACTION = float(os.getenv("VISION_BLANK_MIN_INK_FRACTION", "0.005"))  # 0.5%
# The trusted reference figure. Default is the exact asset used by the
# frontend (frontend/public/mmse-copying-figure.png). Never trust the client
# to supply a reference; it is always loaded server-side from this path.
# A backend-local override is allowed via env for deployment layouts.
COPYING_REFERENCE_PATH = os.getenv("COPYING_REFERENCE_PATH", "").strip()
_DEFAULT_REFERENCE_CANDIDATES = (
    # repo root relative to backend/src
    "frontend/public/mmse-copying-figure.png",
    # backend-local asset location (only used if the file exists)
    "assets/mmse-copying-figure.png",
)


class VisionImageError(Exception):
    """Image validation/processing failure. Maps to a controlled client error."""


def _repo_root() -> str:
    # backend/src/vision_image.py -> backend/src -> backend -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reference_path() -> str:
    if COPYING_REFERENCE_PATH:
        return COPYING_REFERENCE_PATH
    root = _repo_root()
    for candidate in _DEFAULT_REFERENCE_CANDIDATES:
        path = os.path.join(root, candidate)
        if os.path.isfile(path):
            return path
    raise VisionImageError(
        "Reference figure not found on the server. "
        f"Expected at {os.path.join(root, _DEFAULT_REFERENCE_CANDIDATES[0])}."
    )


def load_reference_figure_bytes() -> bytes:
    """Read the trusted server-side reference figure as raw bytes (in memory)."""
    try:
        with open(_reference_path(), "rb") as fh:
            return fh.read()
    except VisionImageError:
        raise
    except OSError as exc:
        raise VisionImageError(f"Could not read the reference figure: {exc}") from exc


def _normalize_image(data: bytes, max_dimension: int = MAX_IMAGE_DIMENSION) -> Image.Image:
    """Decode, transpose (EXIF), convert to RGB, and downscale to a max side."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise VisionImageError("Could not decode the image as a valid picture.") from exc

    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    return img


def has_drawing_content(img: Image.Image) -> bool:
    """
    Deterministic blank / near-blank sanity gate.

    Returns True when the image plausibly contains a drawing. Conservative by
    design: it measures the FRACTION of pixels that are meaningfully darker
    than the estimated paper level (a high percentile of the grayscale
    histogram). A real drawing — even a faint pencil sketch — leaves far more
    than VISION_BLANK_MIN_INK_FRACTION (0.5%) of the page with strokes at
    least VISION_BLANK_INK_DELTA (20) gray levels darker than the paper.
    Pure white pages, near-blank pages, and extremely low-contrast images
    fall below that and are rejected. No figure recognition is attempted.
    """
    gray = img.convert("L")
    hist = gray.histogram()  # 256 bins
    total = sum(hist)
    if total == 0:
        return False

    # Paper level: the grayscale value at the 90th percentile (robust to a
    # small drawing being present — the paper dominates the histogram).
    target = total * 0.9
    acc = 0
    paper = 255
    for level, count in enumerate(hist):
        acc += count
        if acc >= target:
            paper = level
            break

    ink_cutoff = max(0, int(paper - VISION_BLANK_INK_DELTA))
    ink_pixels = sum(hist[: ink_cutoff + 1])
    return (ink_pixels / total) >= VISION_BLANK_MIN_INK_FRACTION


def prepare_patient_image(data: bytes, content_type: str | None = None) -> str:
    """
    Validate, normalize, and encode a patient drawing for a vision provider.

    Returns a base64 data URL. Raises VisionImageError for any rejection:
      - missing body
      - unsupported MIME type
      - file too large
      - undecodable image
      - blank / near-blank image (no drawing content)
    """
    if not data:
        raise VisionImageError("No image was provided.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise VisionImageError(
            f"Image exceeds the maximum allowed size "
            f"({MAX_UPLOAD_BYTES // 1024 // 1024} MB)."
        )

    declared = (content_type or "").strip().lower()
    if declared:
        declared = declared.split(";")[0].strip()
        if declared and declared not in ALLOWED_MIME_TYPES:
            raise VisionImageError(
                "Unsupported image type. Use JPEG, PNG, or WebP."
            )

    img = _normalize_image(data)

    # Blank / near-blank pre-check BEFORE any vision provider call. This is a
    # cheap deterministic gate; it never invokes Ollama/Gemini/OpenAI.
    if not has_drawing_content(img):
        raise VisionImageError(
            "No drawing detected. Please submit a clear photo of the patient's drawing."
        )

    # Re-encode to JPEG in memory so the provider always receives a consistent
    # format; keeps the payload small.
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def prepare_reference_figure() -> str:
    """Load the trusted reference figure and encode it as a base64 data URL."""
    raw = load_reference_figure_bytes()
    img = _normalize_image(raw)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
