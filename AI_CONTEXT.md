# AI Context

> Living project memory for AI coding agents.
> Read this first, then consult the actual source code. If this file conflicts with code, **the code wins** — correct this file and note the discrepancy in the handoff section.

---

## 1. Project Identity

- **Project name:** Alzheimer's Early Screening Tool (aka "Alzheimer's Risk Prediction Using Clinical Features")
- **Purpose:** Estimate Alzheimer's disease risk from a small set of routine clinical measurements, intended as an early-screening / decision-support aid for clinicians.
- **Current research objective:** Give examiners a complete step-by-step MMSE assessment (11 sections, 0–30) that feeds a single computed `mmse` value into the existing ML prediction pipeline.
- **What the system IS:** A research/educational prototype. Screening and decision-support, not diagnosis.
- **What the system is NOT:**
  - NOT a diagnostic device or clinically validated tool.
  - NOT a replacement for clinical judgment.
  - Predictions must never be described as a definitive Alzheimer's diagnosis.
- **Competition/research context:** Standard MMSE (Folstein) is the cognitive screen used here. The current ML model is deliberately simple (5 clinical features) and is a proof-of-concept. No claims of clinical superiority are made.

---

## 2. Current Project Status

### Working (verified)
- Patient clinical input form (age, sex, education_years, mmse, ses).
- ML prediction via FastAPI `POST /predict` (Random Forest pipeline).
- Probability breakdown (pie chart + bars) and confidence/`detection_percentage` display.
- Examiner-assisted **11-section MMSE questionnaire** (replaces the raw MMSE number input) with automatic 0–30 total that flows into the existing API.
- Training-time static SHAP artifacts (plots + `shap_data.pkl`). **There is no live per-patient SHAP endpoint.**

### In Progress
- None currently. (See handoff section.)

### Planned (intentionally postponed)
- Live per-patient SHAP explanations.
- MoCA scoring (mentioned in README roadmap).
- Vision-assisted evaluation of the MMSE copying figure.
- MRI/imaging and multimodal input (long-term README roadmap).

### Rejected / Out of Scope (do NOT implement unless explicitly reconsidered)
- Adding CDR (Clinical Dementia Rating) back as a feature — removed because it leaks the diagnosis (see Research Context).
- Sending individual MMSE answers to the backend or adding 11 new MMSE API fields — the API contract stays 5 fields.
- Automatic AI/NLP grading of the MMSE writing sentence.
- Automatic scoring of the copied figure.

---

## 3. Architecture

### Frontend
- **Framework:** React 19, TypeScript (strict), Vite 7, Tailwind CSS 4, Framer Motion, Axios, Chart.js + react-chartjs-2.
- **Entry point:** `frontend/index.html` → `frontend/src/main.tsx` → `<App/>`.
- **Main components:** `App.tsx` (orchestration + shared state), `Layout`, `FormPanel`, `InputField`, `SelectField`, `AnalyzeButton`, `ResultCard`, `PredictionPieChart`, `EmptyState`, `ErrorMessage`, `LoadingSpinner` (unused), `MMSEAssessment` + MMSE subcomponents.
- **State management:** Local React state only. No router, no Redux/Zustand. Prediction form state and MMSE phase live in `App.tsx`; MMSE section state lives inside `MMSEAssessment`.
- **API layer:** `frontend/src/api.ts` — typed axios client, base URL from `VITE_API_URL` (default `http://127.0.0.1:8000`).
- **Styling architecture:** Tailwind utility classes inline in JSX. No CSS modules, no central design-token file. Dark glassmorphism theme (see Design System).

### Backend
- **Framework:** FastAPI 0.121 + Pydantic v2 + Uvicorn.
- **Entry point:** `backend/src/api.py` (`uvicorn src.api:app`).
- **Routes:** `GET /` (health message), `POST /predict` (prediction), `OPTIONS /predict` (CORS preflight).
- **Validation:** Pydantic `PatientInput` model. Note: no `min`/`max` range constraints are enforced server-side.
- **Model loading:** `joblib.load` at import time via `load_model()`; checks `MODEL_PATH` env, then several default paths. If missing, server still starts but `/predict` returns 500.

### ML
- **Model:** sklearn `Pipeline(StandardScaler, RandomForestClassifier)`.
  - `RandomForestClassifier(n_estimators=200, random_state=42, class_weight={0:1, 1:4, 2:2})`.
- **Preprocessing:** `StandardScaler` (note: scaler before a Random Forest is unnecessary — acknowledged in the metrics JSON; do not add/remove without explicit instruction).
- **Features (exact order):** `age`, `sex`, `education_years`, `mmse`, `ses`.
- **Classes:** `0 = Nondemented`, `1 = Converted`, `2 = Demented`.
- **Training pipeline:** `backend/src/train_clean_clinical_model.py` — loads `Dataset.csv`, renames/maps columns, median-fills NaNs, 80/20 stratified split, trains, saves `best_model.pkl`, and generates static SHAP artifacts.
- **Artifact:** `backend/models/best_model.pkl` (joblib Pipeline).

### Actual data flow
```
User/Examiner
  → React form OR MMSE questionnaire (App.tsx)
  → formData { age, sex, education_years, mmse, ses }
  → predictAlzheimers()  (frontend/src/api.ts)
  → axios POST /predict   (FastAPI backend/src/api.py)
  → DataFrame built with exact 5-feature order
  → pipeline.predict / predict_proba
  → JSON response { alzheimers_detected, detection_percentage, predicted_class,
                    class_index, probabilities, rule_applied, rule_usage_percentage }
  → ResultCard + PredictionPieChart render the result
```

---

## 4. Repository Map

```
alzheimers_ml_project/
├── AI_CONTEXT.md                     # THIS FILE
├── README.md                         # Human-facing project doc (contains the API schema and roadmap)
├── requirements.txt                  # Root, unpinned, incomplete; prefer backend/requirements.txt
├── backend/
│   ├── src/api.py                    # FastAPI prediction API. DO NOT modify for frontend-only features.
│   ├── src/train_clean_clinical_model.py  # Trains model + generates SHAP. Hardcoded absolute data path.
│   ├── src/test_hello.py             # shap/matplotlib import smoke test
│   ├── models/best_model.pkl         # Trained pipeline (joblib). Do not modify.
│   ├── models/clean_clinical_metrics_after_cdr_removal.json  # Post-CDR metrics record
│   ├── models/shap/                  # Static SHAP plots + shap_data.pkl (training-time only)
│   ├── data/raw/Dataset.csv          # OASIS-style training data (committed despite README saying otherwise)
│   ├── data/raw/alzheimers_disease_data.csv  # Unused 4750-row synthetic-style dataset
│   ├── Procfile                      # Heroku: uvicorn src.api:app
│   ├── runtime.txt                   # python-3.10.12
│   └── requirements.txt              # Pinned backend deps (incl. shap, xgboost, matplotlib)
└── frontend/
    ├── index.html                    # mounts #root, loads /src/main.tsx
    ├── package.json                  # deps; build = "tsc && vite build"
    ├── vite.config.ts, tsconfig.json, tailwind.config.js, postcss.config.js
    ├── src/main.tsx                  # REAL entry point
    ├── src/main.ts, src/counter.ts   # DEAD Vite vanilla-template leftovers — do not rely on them
    ├── src/App.tsx                   # Main orchestration + form state + MMSE phase
    ├── src/api.ts                    # Axios client + PredictionResponse/PatientInput types
    ├── src/mmse/state.ts             # MMSE state types, initial state, scoring, completion checks
    ├── src/mmse/config.ts            # MMSE config: objects, expected answers, reference figure path
    └── src/components/
        ├── index.ts                  # Barrel exports for all UI components
        ├── Layout.tsx, FormPanel.tsx, InputField.tsx, SelectField.tsx, AnalyzeButton.tsx,
        ├── ResultCard.tsx, PredictionPieChart.tsx, EmptyState.tsx, ErrorMessage.tsx, LoadingSpinner.tsx (unused)
        └── mmse/                     # MMSE questionnaire UI
            ├── MMSEAssessment.tsx    # Stepper container (intro → 11 sections → summary)
            ├── MMSEIntroduction.tsx, MMSESummary.tsx, DrawingCanvas.tsx, primitives.tsx, sections.tsx, index.ts
```

### Important files — modification policy
| File | Purpose | Modification policy |
|---|---|---|
| `frontend/src/App.tsx` | App orchestration, form + MMSE phase state | Allowed; avoid unrelated refactoring |
| `frontend/src/api.ts` | API types + axios client | Allowed; keep `/predict` payload contract unchanged |
| `frontend/src/mmse/*` | MMSE logic/config/state | Allowed; keep total 0–30 and 11-section structure |
| `frontend/src/components/mmse/*` | MMSE UI | Allowed; preserve design system |
| `frontend/src/components/index.ts` | Barrel exports | Allowed (add exports) |
| `backend/src/api.py` | Prediction API | **Do not modify for frontend-only features** |
| `backend/src/train_clean_clinical_model.py` | Training + SHAP | **Do not modify without explicit instruction** |
| `backend/models/*` | Model + SHAP artifacts | **Do not modify / retrain without explicit instruction** |

---

## 5. API Contract

### `POST /predict`
- **Method:** `POST`
- **Endpoint:** `http://127.0.0.1:8000/predict` (base URL configurable via `VITE_API_URL`)
- **Request body (exactly 5 fields):**
  ```json
  {
    "age": 70,
    "sex": 1,
    "education_years": 12,
    "mmse": 28.0,
    "ses": 2.0
  }
  ```
  - `age`: int; `sex`: int (1 = Male, 0 = Female); `education_years`: int; `mmse`: float (0–30); `ses`: float.
  - No server-side range validation — out-of-range values are currently accepted.
- **Response:**
  ```json
  {
    "alzheimers_detected": true,
    "detection_percentage": 82.35,
    "predicted_class": "Demented",
    "class_index": 2,
    "probabilities": { "Nondemented": 0.12, "Converted": 0.23, "Demented": 0.65 },
    "rule_applied": false,
    "rule_usage_percentage": 0.0
  }
  ```
  - `detection_percentage` = (`Converted` + `Demented` probability) × 100, rounded to 2 dp.
  - `alzheimers_detected` = `pred_class in [1, 2]`.
  - `rule_applied` / `rule_usage_percentage` are hardcoded legacy fields (always `false` / `0.0`).
- **Important assumptions:**
  - `probabilities` and `class_index` are keyed by **position** (`0,1,2`), not by `model.classes_`. This holds only while classes are `[0,1,2]`.
  - The DataFrame is built by column **key**, so request field order does not matter.
- **Frontend caller:** `predictAlzheimers()` in `frontend/src/api.ts`.

---

## 6. ML Contract (CRITICAL — DO NOT CHANGE)

- **Input features (order matters for the scaler):** `age`, `sex`, `education_years`, `mmse`, `ses`.
- **Expected types:** age/sex/education_years ints; mmse/ses floats.
- **Preprocessing:** `StandardScaler` applied before the Random Forest inside the Pipeline.
  - Fitted scaler means: `[77.17, 0.43, 14.53, 27.29, 2.47]`.
- **Model type:** `RandomForestClassifier` (200 trees, `random_state=42`, `class_weight={0:1,1:4,2:2}`).
- **Output classes:** `0 = Nondemented`, `1 = Converted`, `2 = Demented`.
- **Probability interpretation:** probabilities are per-class `predict_proba` outputs, then `detection_percentage` merges Converted+Demented. There is no confidence threshold logic; the ML prediction must never be presented as a diagnosis.
- **Leakage consideration:** CDR was removed because it is assigned *after* clinicians assess dementia severity (circular). Never reintroduce CDR or other post-diagnosis features.
- **Artifact compatibility:** `best_model.pkl` is sklearn-version-sensitive; retraining with different sklearn versions can break `api.py`'s position-indexed probability logic.

---

## 7. MMSE Implementation

- **Why it exists:** Replaces the single raw "MMSE Score" number input with a proper examiner-assisted questionnaire, while keeping the API contract unchanged (only the computed total is sent as `mmse`).
- **Structure:** 11 sections, max total **30**:
  1. Orientation to Time (5) — year/season/date/day/month, 1 pt each.
  2. Orientation to Place (5) — state/county/town/building/floor; correct answers depend on location, configured in `config.ts` (`PLACE_ITEMS`).
  3. Registration (3) — examiner presents 3 objects (`REGISTRATION_OBJECTS`, configurable); 1 pt per correctly repeated object.
  4. Attention & Calculation (5) — examiner selects Serial-7s or spell WORLD backwards; expected answers shown examiner-only.
  5. Delayed Recall (3) — recalls the same Registration objects.
  6. Naming (2) — wristwatch, pencil.
  7. Repetition (1) — "No ifs, ands, or buts."
  8. Three-Step Command (3) — right hand / fold / floor, each scored independently.
  9. Reading (1) — "CLOSE YOUR EYES"; must not be read aloud.
  10. Writing (1) — sentence with a noun and a verb; examiner marks, **no AI grading**.
  11. Copying (1) — reference figure + drawing canvas + examiner scoring. **No automated figure scoring.**
- **Scoring:** `computeTotal()` in `frontend/src/mmse/state.ts`; always an integer 0–30.
- **Examiner-assisted workflow:** The UI is for the examiner, not the patient. Every item is a Correct/Incorrect toggle; instructions are shown in "Examiner instructions — for the examiner, not the patient" boxes.
- **Frontend state:** `src/mmse/state.ts` (`MMSEState` + `createInitialMMSEState`). Section state lives inside `MMSEAssessment` (local state); `App.tsx` only receives the final total.
- **Flow to the API:** MMSE Summary → "Continue to Analysis" → `MMSEAssessment.onComplete(total)` → `App.handleMmseComplete` sets `formData.mmse = total` → existing `predictAlzheimers()` → `POST /predict`.
- **Navigation:** "MMSE Assessment · X of 11" progress bar, Back/Next; Next is disabled until the current section is complete; answers persist when navigating back.
- **Reference figure:** The exact figure asset is **not bundled**. `COPYING_REFERENCE_IMAGE` in `config.ts` is empty and a placeholder is shown until the asset is supplied. Do NOT substitute a generic pentagon.

---

## 8. Design System

- **Theme:** Dark glassmorphism. Black/translucent surfaces with backdrop blur.
- **Colors:** `bg-black/40` cards, `border-white/10`, `text-white` primary, `text-gray-400` secondary; blue gradient CTAs (`from-blue-600 via-blue-500 to-blue-600`); results use emerald (Nondemented), amber (Converted), rose (Demented); background `gray-950 → black → gray-900` gradient.
- **Typography:** System sans stack (`-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'SF Pro Display', Inter, Arial`); large tight-tracked headings; uppercase letter-spaced micro-labels (`text-xs font-semibold text-gray-400 uppercase tracking-[0.08em]`).
- **Spacing:** Generous (`p-6 md:p-8` cards, `gap-7` form grid, `mb-6`–`mb-8` section spacing).
- **Cards:** `rounded-[2rem]`, `bg-black/40`, `backdrop-blur-2xl`, `shadow-[0_8px_32px_rgba(0,0,0,0.4)]`, subtle inner gradient overlays.
- **Buttons:** Blue gradient CTAs with hover lift/scale via Framer Motion; secondary buttons `bg-white/5 border-white/10`.
- **Animations:** Framer Motion entrance animations (fade + slide + scale), `ease: [0.16, 1, 0.3, 1]`, animated probability bars, spinner.
- **Responsive:** Two-column desktop layout (form/MMSE left, results right), stacking on mobile; no horizontal scroll; MMSE drawing canvas supports mouse + touch (Pointer Events, `touch-none`).
- **Reusable components:** `Layout`, `FormPanel`, `InputField`, `SelectField`, `AnalyzeButton`, `ResultCard`, `PredictionPieChart`, `ErrorMessage`, `EmptyState`, plus MMSE primitives (`GlassCard`, `ExaminerInstructions`, `ExaminerToggle`, `SectionShell`).

> **Do not redesign the application unless explicitly requested.**

---

## 9. Current Features (checklist)

- [x] Patient clinical input (age, sex, education_years, mmse, ses)
- [x] ML prediction (`POST /predict`)
- [x] Probability breakdown (pie chart + bars)
- [x] Confidence / `detection_percentage` display
- [x] Detection status (Alzheimer's detected / not)
- [x] MMSE questionnaire (11 sections, examiner-assisted, 0–30)
- [x] MMSE score → existing prediction flow (single `mmse` field)
- [x] Drawing canvas for copying task (mouse + touch)
- [x] Training-time static SHAP artifacts (plots + pickle)
- [ ] Live per-patient SHAP explanations
- [ ] Vision-assisted figure evaluation
- [ ] MoCA scoring

---

## 10. Research Context

- **Dataset:** OASIS-style clinical tabular data (`backend/data/raw/Dataset.csv`), 373 rows, multi-class `Nondemented / Converted / Demented` (class imbalance: only ~37 Converted samples).
- **CDR leakage discovery/fix:** An earlier model included CDR and reached ~89% accuracy. CDR is assigned after dementia severity is known, so it leaked the answer. Removing CDR and retraining dropped accuracy to ~76% — that lower number is the honest one.
- **Current model metrics** (5 features, no CDR, from `clean_clinical_metrics_after_cdr_removal.json`):
  - Accuracy 0.76; macro precision 0.86, recall 0.62, F1 0.65.
  - Converted-class recall is weak (~0.25) due to class imbalance — a known limitation, not a clinical claim.
- **Clinical consultation:** The tool is explicitly framed as a screening prototype; no diagnostic thresholds are claimed. Do not invent thresholds (e.g., "MMSE below X means Alzheimer's").
- **Literature comparison:** README cites Standard MMSE; do not exaggerate research claims or convert planned work into completed work.

---

## 11. Known Issues

| Issue | Severity | Impact | Safe to ignore? | Planned resolution |
|---|---|---|---|---|
| Copying-section reference figure asset missing (`COPYING_REFERENCE_IMAGE` empty) | Medium | Placeholder shown instead of the figure | No (feature incomplete) | Supply asset + set path in `config.ts` |
| `/predict` probabilities keyed by position, not `model.classes_` | Medium | Silent mislabeling if retrained with different class order | Yes until retrained | Use `model.classes_` in `api.py` when next touched |
| No range validation on `PatientInput` (e.g., mmse=999 accepted) | Medium | Garbage-in → misleading output | Mostly | Add Pydantic `Field` constraints |
| StandardScaler before RandomForest is unnecessary | Low | Harmless, slightly confusing | Yes | Clean up only if retraining is authorized |
| Hardcoded absolute dataset path in training script | Low | Retraining fails on other machines | Yes | Make path relative/configurable |
| Dead boilerplate: `src/main.ts`, `counter.ts`, `LoadingSpinner.tsx` (unused) | Low | Confusion | Yes | Remove only with explicit permission |
| Hardcoded legacy `rule_applied` / `rule_usage_percentage` fields | Low | Misleading API surface | Yes | Remove on next API refactor |
| README claims `Dataset.csv` isn't committed, but it is | Low | Doc mismatch | Yes | Fix README |
| Converted-class recall weak (class imbalance) | Medium | Screening may miss conversions | Yes (research limitation) | Larger dataset / resampling (future) |

---

## 12. Known Risks

- **API/ML position-indexing:** Response labels depend on class order `[0,1,2]`; retraining must preserve this or `api.py` must be updated.
- **Model artifact compatibility:** `best_model.pkl` is tied to the installed scikit-learn version; version drift can break loading or prediction.
- **Stale SHAP artifacts:** SHAP is training-time/static; do not assume a live per-patient SHAP endpoint exists.
- **CORS default:** Without `FRONTEND_URL`/production env, CORS is `["*"]` (wide open) — acceptable locally, review before production.
- **Deployment constraints:** Heroku Procfile + `runtime.txt` (Python 3.10); model pkl must ship with the deploy. Dual venvs exist (root `.venv`, `backend/venv`).
- **Dependency warnings:** Vite reports large chunk size (Chart.js) and stale browserslist data — cosmetic, not blocking.
- **Documentation drift:** `README.md` contains stale claims; treat code as authoritative.

---

## 13. Development Rules

1. Do not modify the backend for frontend-only changes.
2. Do not retrain or modify the ML model without explicit instruction.
3. Do not change the `/predict` request/response contract without explicit instruction.
4. Never commit secrets, API keys, `.env` files, credentials, or patient-identifiable data.
5. Do not invent medical rules or clinical thresholds.
6. Do not introduce unnecessary dependencies — reuse the existing stack first.
7. Preserve the existing design system; do not redesign unless explicitly requested.
8. Keep MMSE total an integer in `[0, 30]` and keep the API payload as the single `mmse` number.
9. Use Git: commit after major milestones with descriptive messages (e.g., `feat(mmse): ...`, `fix(...)`, `docs(...)`); push to `origin`.
10. Update `AI_CONTEXT.md` after meaningful architectural/functional changes; keep it accurate, not bloated.
11. Verify before commit: build/type-check/tests, review `git status`/`git diff`, confirm only intended files change.
12. Do not force-push or rewrite history unless explicitly authorized.

---

## 14. Git History / Major Milestones

| Date | Commit | Change | Reason |
|---|---|---|---|
| 2025-11-26 | `b16dc1c` | first commit | Initial project scaffold |
| 2025-11-26 | `9b75216` | Dummy prediction + model loading error handling | API resilience |
| 2025-11-29 | `61f4ce4` | Class weights for Converted detection | Class imbalance |
| 2025-12-01 | `41260aa` | Feature engineering + rule tracking | Test edge cases |
| 2025-12-03 | `c1d88a5` | UI redesign (glassmorphism) | Apple-inspired look |
| 2026-07-11 | `460f777` | Removed CDR from frontend | Data-leakage fix (5 features) |
| 2026-07-11 | `4055df3` | README: 5 features, RF, metrics, CDR note | Doc accuracy |
| 2026-07-11 | `f4fab7b` | SHAP multiclass fix + per-class plots | Correct SHAP artifacts |
| 2026-08-13 | `590ff38` | `feat(mmse): add step-by-step MMSE assessment with examiner scoring` | MMSE questionnaire implementation |
| 2026-08-13 | `c97ba47` | `docs: add AI_CONTEXT.md for agent context management` | AI_CONTEXT.md milestone |

---

## 15. Current Roadmap

### Immediate
- MMSE questionnaire is implemented and building. Remaining: supply the copying-figure asset and set `COPYING_REFERENCE_IMAGE`; optionally configure `PLACE_ITEMS` answers per location.

### Next
- Live per-patient SHAP explanations (backend endpoint + frontend consumption).
- MoCA scoring (README V2 roadmap).

### Later
- Vision-assisted evaluation of the copied figure (document separately when implemented).
- MRI/imaging, multimodal input (README V3/V4 roadmap).
- Larger/balanced dataset to improve Converted-class recall.

### Explicitly deferred
- AI/NLP grading of the writing sentence.
- Automated figure scoring.
- Any change to the ML contract.

---

## 16. Agent Handoff Notes

### Last Completed Work
- Implemented the full examiner-assisted 11-section MMSE questionnaire (frontend only) replacing the raw MMSE number input; final score `mmse: totalMMSE` (0–30) flows through the existing `POST /predict` path. Backend/API/model untouched. `npm run build` (tsc + vite) passes.

### Current State
- Working tree: MMSE implementation (uncommitted) plus this `AI_CONTEXT.md`. Git branch `main`, tracking `origin/main`.

### Next Recommended Task
- Supply the exact MMSE copying figure as an image asset, place it under `frontend/public/`, and set `COPYING_REFERENCE_IMAGE` in `frontend/src/mmse/config.ts`.
- Optionally set per-location expected answers in `PLACE_ITEMS` (`frontend/src/mmse/config.ts`).
- Then consider live per-patient SHAP as the next feature.

### Files Recently Changed
- `frontend/src/App.tsx` (MMSE phase + score integration)
- `frontend/src/components/index.ts` (new exports)
- `frontend/src/components/mmse/*` (new MMSE UI)
- `frontend/src/mmse/state.ts`, `frontend/src/mmse/config.ts` (new MMSE logic/config)

### Tests Verified
- `npm run build` (`tsc && vite build`) — passes, no TypeScript errors.
- Dev server boots cleanly (`vite`).
- Confirmed MMSE strings present in the production bundle.
- No backend tests exist; backend left untouched.

### Commit
- `590ff38` — `feat(mmse): add step-by-step MMSE assessment with examiner scoring` (11 files, +1481/−59)
- `c97ba47` — `docs: add AI_CONTEXT.md for agent context management`

### Push Status
- Pushed to `origin/main` successfully. `main` is up to date with `origin/main`; working tree clean.

### Important Warnings
- No live SHAP endpoint exists — do not assume per-patient SHAP.
- Copying-figure asset is missing; do not substitute a generic pentagon.
- `probabilities` in `/predict` are position-indexed (`[0,1,2]`), not keyed by `model.classes_`.
- Do not retrain the model or modify `backend/src/api.py` without explicit instruction.
- `README.md` has stale claims (e.g., Dataset.csv "not committed") — treat code as authoritative.
