"""
FastAPI app to expose the Alzheimer's model via /predict endpoint.
Run with: uvicorn src.api:app --reload
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class PatientInput(BaseModel):
    visit: int
    mr_delay: int
    sex: int
    hand: int
    age: int
    education_years: int
    ses: float
    mmse: float
    cdr: float
    etiv: int
    nwbv: float
    asf: float


label_map = {0: "Nondemented", 1: "Converted", 2: "Demented"}


def load_model() -> Optional[Any]:
    """Load the trained model. Returns None if model file doesn't exist."""
    model_path = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def dummy_predict(patient: PatientInput) -> Tuple[int, np.ndarray]:
    """
    Dummy prediction function using simple rule-based logic.
    Returns (predicted_class_index, probabilities_array).
    """
    mmse = patient.mmse
    cdr = patient.cdr
    age = patient.age
    
    # Initialize base probabilities
    prob_nondemented = 0.0
    prob_converted = 0.0
    prob_demented = 0.0
    
    # MMSE-based risk assessment
    if mmse < 20:
        # High risk - Demented
        prob_demented = 0.7
        prob_converted = 0.2
        prob_nondemented = 0.1
    elif mmse <= 24:
        # Medium risk - Converted
        prob_converted = 0.5
        prob_demented = 0.2
        prob_nondemented = 0.3
    else:
        # Low risk - Nondemented
        prob_nondemented = 0.8
        prob_converted = 0.15
        prob_demented = 0.05
    
    # CDR adjustment (higher CDR = higher dementia risk)
    if cdr > 1.0:
        # Increase Demented probability
        prob_demented += 0.15
        prob_converted += 0.05
        prob_nondemented = max(0.0, prob_nondemented - 0.2)
    elif cdr > 0.5:
        # Moderate increase in Converted probability
        prob_converted += 0.1
        prob_demented += 0.05
        prob_nondemented = max(0.0, prob_nondemented - 0.15)
    
    # Age adjustment (older = slightly higher risk)
    if age > 75:
        prob_demented += 0.05
        prob_converted += 0.03
        prob_nondemented = max(0.0, prob_nondemented - 0.08)
    
    # Normalize probabilities to sum to 1.0
    total = prob_nondemented + prob_converted + prob_demented
    if total > 0:
        prob_nondemented /= total
        prob_converted /= total
        prob_demented /= total
    
    probabilities = np.array([prob_nondemented, prob_converted, prob_demented])
    predicted_class = int(np.argmax(probabilities))
    
    return predicted_class, probabilities


model = load_model()
app = FastAPI(title="Alzheimer Risk API", version="1.0")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/predict")
def options_predict() -> Response:
    """
    Explicitly handle CORS preflight for /predict to avoid 405.
    """
    return Response(status_code=200)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Alzheimer Risk Prediction API is running."}


@app.post("/predict")
def predict(patient: PatientInput) -> Dict:
    try:
        # Use real model if available, otherwise use dummy prediction
        if model is not None:
            # Extract CLINICAL FEATURES ONLY (matching training)
            clinical_data = {
                "age": patient.age,
                "sex": patient.sex,
                "education_years": patient.education_years,
                "mmse": patient.mmse,
                "cdr": patient.cdr,
                "ses": patient.ses,
            }
            
            # Feature engineering (same as training)
            import pandas as pd
            df_temp = pd.DataFrame([clinical_data])
            
            # Age groups
            age_group = pd.cut([patient.age], bins=[0, 70, 75, 80, 85, 100], labels=[0, 1, 2, 3, 4])[0]
            if pd.isna(age_group):
                age_group = 2
            clinical_data["age_group"] = int(age_group)
            
            # MMSE categories
            mmse_cat = pd.cut([patient.mmse], bins=[0, 17, 23, 30], labels=[0, 1, 2])[0]
            if pd.isna(mmse_cat):
                mmse_cat = 1
            clinical_data["mmse_category"] = int(mmse_cat)
            
            # Create DataFrame with correct feature order
            df = pd.DataFrame([clinical_data])
            
            pred_class = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
        else:
            # Use dummy prediction
            pred_class, probabilities = dummy_predict(patient)
        
        probs = {
            label_map[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }
        
        # Calculate Alzheimer's detection status
        # Converted (1) or Demented (2) = detected
        detection_percentage = (probs.get("Converted", 0.0) + probs.get("Demented", 0.0)) * 100
        alzheimers_detected = pred_class in [1, 2]  # Converted or Demented
        
        return {
            "alzheimers_detected": alzheimers_detected,
            "detection_percentage": round(detection_percentage, 2),
            "predicted_class": label_map.get(pred_class, "Unknown"),
            "class_index": int(pred_class),
            "probabilities": probs,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

