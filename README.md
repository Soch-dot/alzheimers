## Alzheimer’s Risk Prediction Using Clinical Features

This project implements an end‑to‑end application that estimates Alzheimer’s disease risk from a small set of routine clinical measurements.  
It contains a FastAPI backend that serves a trained machine‑learning model and a React frontend that provides an interactive user interface.

### Project Motivation

Alzheimer’s disease is often diagnosed late, after noticeable cognitive decline.  
The goal of this project is to explore whether a simple model, trained only on basic clinical features, can help flag patients who may have a higher probability of dementia and should receive more detailed evaluation.

The project is designed to look and feel like a compact research project that could be used in a college course or capstone.
---

## Dataset

- **Source**: Public Alzheimer’s disease dataset hosted on GitHub: PremChaurasiya07 / *Alzheimer-prediction* (clinical tabular data).
- **Task**: Multi‑class classification of patient status:
  - 0 – Nondemented  
  - 1 – Converted  
  - 2 – Demented
- **Usage in this project**:
  - The raw CSV from that repository is used for training only.
  - The data is not bundled here; to retrain, download the dataset from the original repository and place it under the backend data path described below.

### Features Used (6 Clinical Variables)

The final model intentionally uses only six straightforward clinical features:

1. **age** – Patient age (years)  
2. **sex** – Binary encoding (1 = male, 0 = female)  
3. **education_years** – Years of formal education  
4. **mmse** – Mini‑Mental State Examination score  
5. **cdr** – Clinical Dementia Rating  
6. **ses** – Socioeconomic status score

The target label is the grouped diagnosis (`group`) mapped to the three classes above.

---

## Model

- **Algorithm**: Gradient‑boosted decision trees (XGBoost classifier) wrapped in a pipeline.
- **Input**: The six features listed above, with no additional feature engineering beyond basic preprocessing.
- **Output**:
  - Predicted class (Nondemented / Converted / Demented)
  - Class index (0, 1, or 2)
  - Class probabilities for all three categories
  - A derived “detection percentage” summarizing the probability of any dementia‑related class (Converted or Demented).

During training, the model is saved as `best_model.pkl` under the backend `models/` directory.  
The FastAPI backend loads this file at startup and uses it for inference.

---

## Project Structure (Cleaned)

At a high level:

```
alzheimers_ml_project/
├── README.md                # This file
├── backend/
│   ├── models/
│   │   └── best_model.pkl   # Trained XGBoost model used by the API
│   ├── Procfile             # Optional: process definition for deployment platforms
│   ├── requirements.txt     # Backend Python dependencies
│   ├── runtime.txt          # Optional: Python runtime pin for deployment
│   └── src/
│       ├── api.py           # FastAPI application (inference only)
│       └── train_clean_clinical_model.py  # Training script for the 6‑feature model
└── frontend/
    ├── index.html
    ├── package.json         # React + Vite + Tailwind + Framer Motion
    ├── package-lock.json
    ├── postcss.config.js
    ├── public/
    │   └── vite.svg
    ├── src/
    │   ├── api.ts           # HTTP client using VITE_API_URL
    │   ├── App.tsx
    │   ├── components/      # Reusable UI components
    │   ├── main.ts
    │   ├── main.tsx
    │   └── style.css
    ├── tailwind.config.js
    ├── tsconfig.json
    └── vite.config.ts
```

Only the files needed for the final model and UI are kept.  
Exploratory notebooks, intermediate datasets, experimental scripts, and AI‑generated summaries have been removed or excluded from the core structure.

---

## Backend: FastAPI Service

The backend exposes a simple prediction API around the trained classifier.

### Technology

- **Framework**: FastAPI  
- **Model storage**: `joblib`‑serialized XGBoost pipeline (`backend/models/best_model.pkl`)  
- **Server**: Uvicorn (for local development and deployment)

### Directory Layout (Backend)

```text
backend/
├── models/
│   └── best_model.pkl
├── Procfile
├── requirements.txt
├── runtime.txt
└── src/
    ├── api.py
    └── train_clean_clinical_model.py
```

### Prediction Request Schema

The `/predict` endpoint accepts a JSON body with exactly these six fields:

```json
{
  "age": 75,
  "sex": 1,
  "education_years": 16,
  "mmse": 27.0,
  "cdr": 0.5,
  "ses": 2.0
}
```

### Example Response

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

The `detection_percentage` aggregates the probabilities of the Converted and Demented classes to give a single interpretable number (0–100%).

### Running the Backend Locally

1. **Create and activate a virtual environment** (from the `backend/` folder):

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS / Linux
   # or on Windows (PowerShell)
   # .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the trained model file is present**:

   - Confirm that `backend/models/best_model.pkl` exists.  
   - If you need to retrain the model, see the training section below.

4. **Start the FastAPI server with Uvicorn**:

   ```bash
   uvicorn src.api:app --host 127.0.0.1 --port 8000
   ```

5. **Test the API**:

   - Open a tool such as `curl`, Postman, or a browser plugin and send a POST request to:
     - `http://127.0.0.1:8000/predict`
   - Use the JSON body shown in the example above.

If the model file cannot be found, the API will return a 500 error explaining that `best_model.pkl` is missing.

---

## Training the Clean 6‑Feature Model

The training script is kept intentionally simple and is separated from the API.  
It uses only the six clinical features listed earlier.

### Steps to Retrain

1. **Obtain the dataset**:
   - Visit the Alzheimer’s dataset repository `PremChaurasiya07/Alzheimer-prediction` on GitHub.
   - Download the clinical data CSV (commonly named `Dataset.csv` or similar).

2. **Place the dataset for the backend**:
   - Create the following directory under `backend/` if it does not exist:

     ```text
     backend/data/raw/
     ```

   - Save the downloaded CSV there as:

     ```text
     backend/data/raw/Dataset.csv
     ```

3. **Run the training script** (from inside `backend/`):

   ```bash
   python -m src.train_clean_clinical_model
   ```

4. **Model output**:
   - The script trains the classifier, prints evaluation metrics, and writes:

     ```text
     backend/models/best_model.pkl
     ```

   - The API will automatically use this updated file the next time it starts.

---

## Frontend: React Interface

The frontend is a single‑page application built with modern tooling:

- **Framework**: React  
- **Styling**: Tailwind CSS  
- **Animations**: Framer Motion  
- **Build tool**: Vite + TypeScript

The UI is designed as a clean, clinical dashboard:

- A centered panel that collects the six clinical inputs.
- Clear labels and tooltips for each field.
- A prominent “Analyze Risk” button.
- A results area that shows the predicted class and a visual breakdown of probabilities (e.g., via a chart or card layout).

### API Configuration (VITE_API_URL)

The frontend reads the backend URL from an environment variable:

- `VITE_API_URL` – base URL for the FastAPI service (e.g., `http://127.0.0.1:8000`).

If `VITE_API_URL` is not set, it defaults to `http://127.0.0.1:8000` for development.  
This logic is implemented in `frontend/src/api.ts`.

To override it, create a `.env` file in the `frontend/` directory:

```env
VITE_API_URL=https://your-backend-domain.com
```

---

## Running the Frontend Locally

From the `frontend/` directory:

1. **Install Node dependencies**:

   ```bash
   npm install
   ```

2. **Set the API URL (optional in local development)**:

   - If you are running the backend on the default `http://127.0.0.1:8000`, no extra configuration is needed.
   - For a custom backend URL, create `.env` as described above.

3. **Start the development server**:

   ```bash
   npm run dev
   ```

   Vite will start the dev server, usually at `http://localhost:5173`.

4. **Build for production**:

   ```bash
   npm run build
   ```

   The production build will be emitted into the `dist/` folder (not committed by default).

---

## End‑to‑End Flow

1. A user opens the React frontend and fills in the six clinical inputs.  
2. The frontend sends a POST request to the FastAPI `/predict` endpoint using `VITE_API_URL` as the base.  
3. The backend loads `best_model.pkl`, constructs a DataFrame with the six features, and generates class probabilities.  
4. The API returns the predicted class, class index, and probability distribution.  
5. The frontend displays the result and a visual breakdown of the probabilities to the user.

---

## Notes and Limitations

- This project is for educational and exploratory purposes only and **is not** a medical device.  
- Predictions should **not** be used for clinical decision‑making without proper validation and oversight.  
- The dataset, while useful for experimentation, may not represent all populations or real‑world clinical variability.

Despite these limitations, the project demonstrates how a small, well‑defined tabular dataset can be turned into a complete machine‑learning application with a clear API and modern user interface.

