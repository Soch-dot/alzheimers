"""
Alzheimer's Disease Prediction API - Clean 5-Feature Model (CDR Removed)
Run with: uvicorn src.api:app --reload
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import hashlib
import joblib
import numpy as np
import pandas as pd
import base64
import json
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# AI-assisted MMSE evaluation service (provider-agnostic; does not touch /predict).
from src.ai_eval import MMSEEvaluateRequest, evaluate_mmse_batch
# MMSE Question 11 vision-assisted figure copying evaluation.
from src.vision_eval import (
    VisionProviderError,
    VisionImageError,
    evaluate_copying_image,
)

# ------------------------------------
# Input Schema (5 FIELDS - CDR REMOVED: data leakage)
# ------------------------------------
class PatientInput(BaseModel):
    age: int
    sex: int  # 1 = Male, 0 = Female
    education_years: int
    mmse: float
    ses: float

# ------------------------------------
# Experimental binary screening candidate (frozen Phase 11) - versioned contract
# ------------------------------------
SCREENING_MODEL_VERSION = "binary_lr_latest_visit_v1"
SCREENING_THRESHOLD = 0.40
SCREENING_FEATURES = ["age", "sex", "education_years", "mmse", "ses"]
SCREENING_ARTIFACT_MD5 = "8FC95A3838FFF665CF47FA55C4322096"
SCREENING_TARGET = "dementia_related_outcome"  # 1 = Converted OR Demented

# Display-only sigmoid calibration (Phase 15). The screening DECISION stays on the
# RAW screening_probability (>= 0.40); calibrated_screening_probability is DISPLAY
# ONLY and never determines screening_result.
CALIBRATOR_VERSION = "sigmoid_calibrator_v1"
CALIBRATOR_ARTIFACT_MD5 = "1BF75C442194E83D30847FD0F5B8C044"
CALIBRATED_DISPLAY_BOUNDARY = 0.4082  # calibrated value at raw 0.40 (informational only)

label_map = {0: "Nondemented", 1: "Converted", 2: "Demented"}

# ------------------------------------
# Load model (versioned production artifact)
# ------------------------------------
def _md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_model() -> Optional[Any]:
    """Load the versioned experimental binary screening model.

    The production artifact is backend/models/production/binary_lr_latest_visit_v1.pkl.
    Its checksum is verified against the locked value; if it changes unexpectedly
    the load fails loudly (the legacy best_model.pkl is NOT used as the active
    predictor - it remains only as a rollback/research artifact).
    Returns None if the model file doesn't exist.
    """
    possible_paths = [
        Path(__file__).resolve().parents[1] / "models" / "production" / "binary_lr_latest_visit_v1.pkl",
        Path.cwd() / "models" / "production" / "binary_lr_latest_visit_v1.pkl",
        Path("/app/models/production/binary_lr_latest_visit_v1.pkl"),
    ]

    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        possible_paths.insert(0, Path(env_model_path))

    for model_path in possible_paths:
        if model_path.exists():
            actual = _md5_of_file(model_path)
            if actual != SCREENING_ARTIFACT_MD5:
                print(f"ERROR: {model_path} checksum {actual} does not match locked {SCREENING_ARTIFACT_MD5}")
                raise RuntimeError(
                    f"Model artifact checksum mismatch for {model_path.name}: "
                    f"got {actual}, expected {SCREENING_ARTIFACT_MD5}. "
                    "Do not use a stale or modified artifact."
                )
            print(f"Loading versioned model from: {model_path} (md5 {actual})")
            return joblib.load(model_path)

    print("Warning: Versioned screening model file not found. Prediction will fail.")
    return None


def load_calibrator() -> Optional[Any]:
    """Load the display-only sigmoid calibrator (sigmoid_calibrator_v1).

    Fails safely (returns None) if missing or checksum-mismatched. The calibrator
    only produces the DISPLAY value; it never affects the raw screening decision.
    """
    possible_paths = [
        Path(__file__).resolve().parents[1] / "models" / "production" / "sigmoid_calibrator_v1.pkl",
        Path.cwd() / "models" / "production" / "sigmoid_calibrator_v1.pkl",
        Path("/app/models/production/sigmoid_calibrator_v1.pkl"),
    ]
    env_cal_path = os.getenv("CALIBRATOR_PATH")
    if env_cal_path:
        # When explicitly configured, the override is authoritative — do not
        # silently fall back to a different artifact.
        possible_paths = [Path(env_cal_path)]

    for cal_path in possible_paths:
        if cal_path.exists():
            actual = _md5_of_file(cal_path)
            if actual != CALIBRATOR_ARTIFACT_MD5:
                print(f"ERROR: {cal_path} checksum {actual} does not match locked {CALIBRATOR_ARTIFACT_MD5}")
                return None
            print(f"Loading display-only calibrator from: {cal_path} (md5 {actual})")
            return joblib.load(cal_path)

    print("Warning: Display-only calibrator file not found. Calibrated display value unavailable.")
    return None


model = load_model()
calibrator = load_calibrator()
app = FastAPI(title="Alzheimer Risk API", version="1.0")

# CORS configuration
frontend_url = os.getenv("FRONTEND_URL", "")
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if frontend_url:
    origins.append(frontend_url)
    origins.append(frontend_url.replace("https://", "http://"))

if not frontend_url and os.getenv("ENVIRONMENT") != "production":
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/predict")
def options_predict() -> Response:
    return Response(status_code=200)

# ------------------------------------
# AI-assisted MMSE batch evaluation (separate service; /predict contract unchanged)
# ------------------------------------
@app.post("/mmse/evaluate")
def mmse_evaluate(req: MMSEEvaluateRequest) -> Dict:
    return evaluate_mmse_batch(req)


# ------------------------------------
# MMSE Question 11 — vision-assisted figure copying (separate service;
# /predict and /mmse/evaluate contracts unchanged)
# ------------------------------------
@app.post("/mmse/copying/evaluate")
async def mmse_copying_evaluate(request: Request) -> Dict:
    """
    Evaluate a patient's copy of the MMSE figure against the trusted
    server-side reference. The patient drawing is supplied as the raw request
    body (Content-Type: image/jpeg|png|webp) or as JSON:
        {"image": "data:image/jpeg;base64,..."}
    The reference figure is always loaded server-side from the trusted asset.
    Patient images are processed in memory and never persisted.
    """
    content_type = request.headers.get("content-type", "")
    raw_body = await request.body()
    if not raw_body:
        raise HTTPException(status_code=400, detail="No image was provided.")

    image_bytes = raw_body
    mime = content_type
    if content_type.lower().startswith("application/json"):
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
        image_field = payload.get("image", "")
        if not isinstance(image_field, str) or not image_field.strip():
            raise HTTPException(status_code=400, detail="No image was provided.")
        if "," in image_field and image_field.strip().startswith("data:"):
            header, _, b64 = image_field.strip().partition(",")
            mime = header.removeprefix("data:").split(";")[0]
        else:
            b64 = image_field
            mime = "image/jpeg"
        try:
            image_bytes = base64.b64decode(b64)
        except (ValueError, base64.binascii.Error) as exc:
            raise HTTPException(status_code=400, detail="Invalid image data.") from exc

    try:
        return evaluate_copying_image(image_bytes, mime)
    except VisionImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VisionProviderError as exc:
        if exc.kind == "timeout":
            raise HTTPException(
                status_code=504, detail="Vision assessment timed out."
            ) from exc
        if exc.kind == "invalid":
            raise HTTPException(
                status_code=502, detail="Vision assessment returned an invalid result."
            ) from exc
        raise HTTPException(
            status_code=503, detail="Vision assessment unavailable."
        ) from exc

@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Alzheimer Risk Prediction API is running."}

# ------------------------------------
# Prediction — 5 INPUT FEATURES: age, sex, education_years, mmse, ses
# Versioned binary screening contract (experimental binary screening candidate).
# ------------------------------------
@app.post("/predict")
def predict(patient: PatientInput) -> Dict:
    try:
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Versioned screening model not found. Please ensure "
                       "binary_lr_latest_visit_v1.pkl exists in models/production.",
            )
        
        # Build DataFrame with EXACT feature order: age, sex, education_years, mmse, ses
        feature_values = {
            "age": patient.age,
            "sex": patient.sex,
            "education_years": patient.education_years,
            "mmse": patient.mmse,
            "ses": patient.ses,
        }
        df = pd.DataFrame([feature_values])
        
        print(f"\n=== PREDICTION REQUEST ===")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame values:\n{df}")
        print(f"========================\n")
        
        screening_probability = float(model.predict_proba(df)[0][1])
        predicted_class = "dementia_related" if screening_probability >= SCREENING_THRESHOLD else "nondemented"
        screening_result = "positive" if screening_probability >= SCREENING_THRESHOLD else "negative"
        
        # DISPLAY-ONLY sigmoid calibration (Phase 15). Never changes the decision.
        calibrated_screening_probability = None
        calibrated_available = False
        if calibrator is not None:
            try:
                calibrated_screening_probability = float(
                    calibrator.predict(np.array([[screening_probability]]))[0]
                )
                calibrated_available = True
            except Exception as exc:
                print(f"WARNING: calibrator prediction failed: {exc}")
        
        return {
            "model_version": SCREENING_MODEL_VERSION,
            "calibrator_version": CALIBRATOR_VERSION if calibrated_available else None,
            "screening_target": SCREENING_TARGET,
            "screening_probability": round(screening_probability, 6),
            "calibrated_screening_probability": (
                round(calibrated_screening_probability, 6) if calibrated_available else None
            ),
            "screening_threshold": SCREENING_THRESHOLD,
            "calibrated_threshold_equivalent": CALIBRATED_DISPLAY_BOUNDARY,
            "screening_result": screening_result,
            "predicted_class": predicted_class,
            "features": feature_values,
            "interpretation": {
                "label": "Model-estimated screening probability",
                "not_a_diagnosis": True,
            },
            "calibration": {
                "display_calibrated": calibrated_available,
                "method": "sigmoid",
                "decision_uses_raw_probability": True,
                "note": ("Display-only calibration. Screening decision is based on the "
                         "RAW screening_probability threshold 0.40, not the calibrated value."),
            },
            "limitations": {
                "clinical_validation": False,
                "prospective_conversion_prediction": False,
                "calibrated_not_clinically_validated": True,
            },
        }
        
    except Exception as exc:
        print(f"ERROR in /predict: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc