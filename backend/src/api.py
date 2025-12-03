"""
Alzheimer's Disease Prediction API - Clean 6-Feature Model
Run with: uvicorn src.api:app --reload
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------
# Input Schema (ONLY 6 FIELDS - EXACT ORDER)
# ------------------------------------
class PatientInput(BaseModel):
    age: int
    sex: int  # 1 = Male, 0 = Female
    education_years: int
    mmse: float
    cdr: float
    ses: float

label_map = {0: "Nondemented", 1: "Converted", 2: "Demented"}

# ------------------------------------
# Load model
# ------------------------------------
def load_model() -> Optional[Any]:
    """Load the trained model. Returns None if model file doesn't exist."""
    possible_paths = [
        Path(__file__).resolve().parents[1] / "models" / "best_model.pkl",  # Local dev
        Path.cwd() / "models" / "best_model.pkl",  # Production
        Path("/app/models/best_model.pkl"),  # Docker/Render
    ]
    
    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        possible_paths.insert(0, Path(env_model_path))
    
    for model_path in possible_paths:
        if model_path.exists():
            print(f"Loading model from: {model_path}")
            return joblib.load(model_path)
    
    print("Warning: Model file not found. Prediction will fail.")
    return None

model = load_model()
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

@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Alzheimer Risk Prediction API is running."}

# ------------------------------------
# Prediction — ONLY 6 INPUT FEATURES, EXACT ORDER, NO ENGINEERING
# ------------------------------------
@app.post("/predict")
def predict(patient: PatientInput) -> Dict:
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not found. Please ensure best_model.pkl exists.")
        
        # Build DataFrame with EXACT feature order: age, sex, education_years, mmse, cdr, ses
        df = pd.DataFrame([{
            "age": patient.age,
            "sex": patient.sex,
            "education_years": patient.education_years,
            "mmse": patient.mmse,
            "cdr": patient.cdr,
            "ses": patient.ses,
        }])
        
        # DEBUG: Print columns being sent to model
        print(f"\n=== PREDICTION REQUEST ===")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame values:\n{df}")
        print(f"========================\n")
        
        # Predict using model (includes StandardScaler in pipeline)
        pred_class = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]
        
        # Format probabilities
        probs_dict = {
            label_map[i]: float(probabilities[i])
            for i in range(len(label_map))
        }
        
        # Calculate detection status
        detection_percentage = (probs_dict.get("Converted", 0.0) + probs_dict.get("Demented", 0.0)) * 100
        alzheimers_detected = pred_class in [1, 2]
        
        return {
            "alzheimers_detected": alzheimers_detected,
            "detection_percentage": round(detection_percentage, 2),
            "predicted_class": label_map.get(pred_class, "Unknown"),
            "class_index": pred_class,
            "probabilities": probs_dict,
            "rule_applied": False,
            "rule_usage_percentage": 0.0,
        }
        
    except Exception as exc:
        print(f"ERROR in /predict: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
