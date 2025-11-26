"""
FastAPI app to expose the Alzheimer's model via /predict endpoint.
Run with: uvicorn src.api:app --reload
"""

from pathlib import Path
from typing import Dict

import joblib
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


def load_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


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
def predict(patient: PatientInput) -> Dict[str, Dict[str, float]]:
    try:
        df = pd.DataFrame([patient.dict()])
        pred_class = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        probs = {
            label_map[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }
        return {
            "predicted_class": label_map.get(pred_class, "Unknown"),
            "class_index": int(pred_class),
            "probabilities": probs,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

