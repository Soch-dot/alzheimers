## Alzheimer's Risk Prediction Using Clinical Features

This project estimates Alzheimer's disease risk from a small set of routine clinical measurements. It has a FastAPI backend serving a trained machine-learning model and a React frontend for interacting with it.

### Project motivation

Alzheimer's is often diagnosed late, after cognitive decline is already noticeable. This project asks whether a simple model, trained on basic clinical features, can flag patients who should get a closer look from a neurologist.

This is a screening tool, not a diagnostic one. It flags risk — it doesn't replace clinical judgment.

---

## Dataset

- **Source**: PremChaurasiya07 / *Alzheimer-prediction* on GitHub — clinical tabular data, 373 rows, OASIS-style.
- **Task**: Multi-class classification of patient status:
  - 0 – Nondemented
  - 1 – Converted
  - 2 – Demented
- The raw CSV is used for training only and isn't bundled in this repo. To retrain, download it and place it under the backend data path described below.

### Features used (5 clinical variables)

1. **age** – patient age (years)
2. **sex** – 1 = male, 0 = female
3. **education_years** – years of formal education
4. **mmse** – Mini-Mental State Examination score
5. **ses** – socioeconomic status score

The target is the grouped diagnosis (`group`), mapped to the three classes above.

### A note on data leakage (CDR removal)

An earlier version of this model included CDR (Clinical Dementia Rating) as an input. That was a mistake — CDR is assigned by clinicians *after* they've already assessed dementia severity, so using it to predict dementia is circular. The model was effectively being told the answer.

Removing CDR was the right call: the earlier ~89% was inflated by CDR leakage, not real predictive skill. The later row-level evaluation reported approximately 76% accuracy, but that estimate is also not a valid generalization estimate — it used a row-level train/test split, so repeated visits from the same subjects appeared in both training and test sets. The honest, leakage-free subject-level baseline for this model is approximately 59.8% accuracy, with balanced accuracy approximately 48.0% and macro F1 approximately 47.5% (see below).

---

## Model

- **Algorithm**: Random Forest classifier.
- **Input**: the five features above, no additional feature engineering.
- **Output**: predicted class, class index, class probabilities, and a derived "detection percentage" (probability of Converted or Demented combined).

### Current performance (leakage-free subject-level baseline)

| Metric | Value |
|---|---|
| Accuracy | ~59.8% |
| Balanced accuracy | ~48.0% |
| Macro F1 | ~47.5% |

These are the only currently valid performance numbers. Earlier row-level evaluation reported approximately 76% accuracy, but that estimate was invalid because repeated visits from the same subjects appeared in both training and test sets. The apparent 96.74% accuracy was produced by evaluating a Random Forest trained with a row-level split that leaked subjects across training and test; it is not a valid generalization estimate.

This is a subject-level research baseline, not a clinically validated result: 150 subjects total, 38 outer-test subjects, only 14 Converted subjects overall and 4 in the outer test, so Converted-class metrics are statistically unstable. It is not prospective future-conversion prediction.

The trained model is saved as `best_model.pkl` under `backend/models/`. The FastAPI backend loads it at startup.

---

## Project structure

​```
alzheimers_ml_project/
├── README.md
├── backend/
│   ├── models/
│   │   └── best_model.pkl   # Trained Random Forest model
│   ├── data/
│   │   └── raw/
│   │       └── Dataset.csv  # Training data (not committed)
│   ├── Procfile
│   ├── requirements.txt
│   ├── runtime.txt
│   └── src/
│       ├── api.py
│       └── train_clean_clinical_model.py
└── frontend/
    ├── index.html
    ├── package.json
    ├── package-lock.json
    ├── postcss.config.js
    ├── public/
    │   └── vite.svg
    ├── src/
    │   ├── api.ts
    │   ├── App.tsx
    │   ├── components/
    │   ├── main.ts
    │   ├── main.tsx
    │   └── style.css
    ├── tailwind.config.js
    ├── tsconfig.json
    └── vite.config.ts
​```

## Backend: FastAPI service

- **Framework**: FastAPI
- **Model storage**: `joblib`-serialized Random Forest pipeline (`backend/models/best_model.pkl`)
- **Server**: Uvicorn

### Prediction request schema

`/predict` accepts a JSON body with exactly these five fields:

```json
{
  "age": 75,
  "sex": 1,
  "education_years": 16,
  "mmse": 27.0,
  "ses": 2.0
}
```

### Example response

```json
{
  "alzheimers_detected": true,
  "detection_percentage": 82.35,
  "predicted_class": "Demented",
  "class_index": 2,
  "probabilities": {
    "Nondemented": 0.12,
    "Converted": 0.23,
    "Demented": 0.65
  },
  "rule_applied": false,
  "rule_usage_percentage": 0.0
}
```

`detection_percentage` combines Converted and Demented probabilities into one 0–100 number.

### Running the backend locally

1. Create and activate a virtual environment from `backend/`:

```bash
   python -m venv venv
   source venv/bin/activate      # macOS / Linux
   # .\venv\Scripts\Activate.ps1  # Windows PowerShell
```

2. Install dependencies:

```bash
   pip install -r requirements.txt
```

3. Confirm `backend/models/best_model.pkl` exists. If not, retrain it (see below).

4. Start the server:

```bash
   uvicorn src.api:app --host 127.0.0.1 --port 8000
```

5. Send a POST request to `http://127.0.0.1:8000/predict` with the JSON body above.

---

## Training the 5-feature model

1. Download the dataset from `PremChaurasiya07/Alzheimer-prediction`.
2. Place it at `backend/data/raw/Dataset.csv`.
3. Run, from inside `backend/`:

```bash
   python -m src.train_clean_clinical_model
```

4. This trains the classifier, prints metrics, and writes `backend/models/best_model.pkl`. The API picks up the new file on next start.

---

## Frontend: React interface

- **Framework**: React
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Build tool**: Vite + TypeScript

A centered panel collects the five clinical inputs, a button triggers analysis, and a results panel shows the predicted class alongside a probability breakdown chart.

### API configuration (VITE_API_URL)

The frontend reads the backend URL from `VITE_API_URL`, defaulting to `http://127.0.0.1:8000` for local dev. To override, create `frontend/.env`:

```env
VITE_API_URL=https://your-backend-domain.com
```

### Running the frontend locally

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production build, output in dist/
```

---

## End-to-end flow

1. User fills in the five clinical inputs.
2. Frontend POSTs to `/predict`.
3. Backend loads `best_model.pkl`, builds a DataFrame from the five features, and generates class probabilities.
4. API returns predicted class, class index, and probability distribution.
5. Frontend renders the result and a probability chart.

---

## Roadmap

- **Version 1 (current)**: clinical screening tool on 5 structured features.
- **Version 2**: SHAP explainability, MoCA scoring.
- **Version 3**: MRI/imaging, multimodal input.
- **Version 4**: production-grade healthcare AI platform.

---

## Notes and limitations

This is a research/educational project, not a medical device. Predictions shouldn't inform clinical decisions without proper validation. The dataset is small (373 rows) and class-imbalanced, so results are a proof of concept rather than a clinically validated tool.

The project does show a full pipeline from raw clinical data to a working model with an API and frontend — and catching the CDR leakage issue and fixing it properly is part of that story, not something to hide.