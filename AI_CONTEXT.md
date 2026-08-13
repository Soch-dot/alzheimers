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
- **AI-assisted 11-section MMSE questionnaire** (replaces the raw MMSE number input): patient responses are recorded, scored automatically by an AI evaluation service (`POST /mmse/evaluate`, provider-agnostic: Ollama or Gemini), with examiner override/review. Final `mmse` (0–30) flows into the existing `/predict`.
- Browser speech capture (Web Speech API, no new dependency) with graceful degradation to typing.
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
- Sending individual MMSE answers to the backend or adding 11 new MMSE API fields — the API contract stays 5 fields. (Per-item responses DO go to the separate `/mmse/evaluate` AI service, never to `/predict`.)
- Automatic scoring of the copied figure (Question 11) — planned separately as a vision workflow (photograph → vision model → examiner review).
- Automated vision scoring of the Reading instruction (Question 9) — currently examiner-observed; vision layer not implemented.

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
- **Routes:** `GET /` (health message), `POST /predict` (prediction), `POST /mmse/evaluate` (AI-assisted MMSE item scoring), `OPTIONS /predict` (CORS preflight).
- **AI service:** `backend/src/ai_eval.py` — provider-agnostic (`AI_PROVIDER=ollama|gemini|none`), structured JSON output validation, per-section prompts. Uses only Python stdlib `urllib` (no new deps). No patient data logged/persisted. Backend-only secrets via env / `.env` (python-dotenv).
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
│   ├── src/api.py                    # FastAPI prediction API + /mmse/evaluate. /predict contract frozen.
│   ├── src/ai_eval.py                # AI-assisted MMSE evaluation service (provider-agnostic)
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
| `backend/src/api.py` | Prediction API | `/predict` contract frozen; adding new endpoints (e.g. `/mmse/evaluate`) is allowed |
| `backend/src/ai_eval.py` | AI-assisted MMSE evaluation service | Allowed (new) |
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

### `POST /mmse/evaluate` (AI-assisted MMSE item scoring — separate service)
- **Purpose:** Scores one MMSE item's patient response via an AI provider. Never sends item answers to `/predict`; only the final `mmse` total does.
- **Request body:**
  ```json
  { "section": "naming", "item_key": "wristwatch", "question": "What is this?", "response": "watch", "expected": "wristwatch" }
  ```
  - `section` is one of: `orientation_time, orientation_place, registration, attention_serial7, attention_spell_world, delayed_recall, naming, repetition, writing`.
  - `expected` is the hidden evaluation context (examiner-only). For `orientation_time` the backend derives today's date-based expected answer server-side and ignores the client value.
- **Response (validated structured output):**
  ```json
  { "correct": true, "score": 1, "confidence": 0.98, "reason": "..." }
  ```
  - `correct` bool; `score` must equal 1/0 per correct; `confidence` in [0,1]; `reason` non-empty. Validation failure → `422` (no score auto-assigned). Provider/network/config failure → `503`.
- **Provider config (backend env only):** `AI_PROVIDER=ollama|gemini|none`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL` (default `gemma3`), `GEMINI_BASE_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `AI_TIMEOUT`.
- **Frontend caller:** `evaluateMmseItem()` in `frontend/src/api.ts`.
- **Disclaimers:** AI result is an assist signal, never a diagnosis. Confidence is a model signal, not clinical certainty. `AI_PROVIDER=none` or an unreachable provider surfaces "AI assessment unavailable" with Retry/Manual-review in the UI.

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

- **Why it exists:** Replaces the single raw "MMSE Score" number input with an AI-assisted examiner-supervised questionnaire, while keeping the API contract unchanged (only the computed total is sent as `mmse`).
- **Workflow:** Question → patient responds → response captured (typed or speech) → AI evaluates (`POST /mmse/evaluate`) → score recorded automatically → MMSE total updates. The examiner supervises; manual scoring is an override/exception, not the default.
- **Structure:** 11 sections, max total **30**:
  1. Orientation to Time (5) — year/season/date/day/month; AI-scored against server-derived date values.
  2. Orientation to Place (5) — state/county/town/building/floor; AI-scored against per-location `PLACE_ITEMS` expected answers; **unconfigured items are manual-only** ("score manually").
  3. Registration (3) — 3 objects (`REGISTRATION_OBJECTS`, configurable); AI-scored; objects reused for Delayed Recall.
  4. Attention & Calculation (5) — Serial-7s (5 fields) or spell WORLD backwards (full response split into per-letter scores); AI-scored; expected sequence examiner-only.
  5. Delayed Recall (3) — recalls the same Registration objects; AI-scored.
  6. Naming (2) — wristwatch, pencil; AI-scored (accepts synonyms like wristwatch/watch).
  7. Repetition (1) — "No ifs, ands, or buts."; AI-scored (allows case/punctuation differences; low confidence → review).
  8. Three-Step Command (3) — right hand / fold / floor; **manual examiner observations** (physical task, no AI).
  9. Reading (1) — "CLOSE YOUR EYES"; **manual examiner observation** + optional note. Automated vision scoring NOT implemented (documented limitation).
  10. Writing (1) — sentence with noun and verb; **AI-scored on that criterion only** (never spelling/grammar/handwriting/intelligence).
  11. Copying (1) — reference figure + drawing canvas + **manual examiner scoring**. Vision workflow planned separately; **not implemented in the AI-scoring task**.
- **AI confidence:** model/service signal only, never clinical certainty. Confidence < `AI_CONFIDENCE_REVIEW_THRESHOLD` (0.7, config.ts) → item flagged "⚠ Review required"; it does not count as complete until the examiner accepts the AI result or overrides.
- **Failure handling:** AI error → "AI assessment unavailable." with Retry + Manual review. No silent auto-score on failure. Invalid AI JSON → backend 422, no score assigned.
- **Frontend state:** `src/mmse/state.ts` (`MMSEState` + `createInitialMMSEState`). `ItemState { response, status, aiScore, reviewRequired, reviewed, manual, error }` keeps response text separate from the score. `effectiveCorrect()` = manual verdict wins over AI; `isItemFinalized()` gates section completion (AI finalized unless low-confidence-unreviewed).
- **Speech capture:** `useSpeechRecognition()` in `primitives.tsx` uses the native Web Speech API (`SpeechRecognition`/`webkitSpeechRecognition`), no new dependency. Unsupported browsers degrade to typing with a notice.
- **Flow to the API:** MMSE Summary → "Continue to Analysis" → `MMSEAssessment.onComplete(total)` → `App.handleMmseComplete` sets `formData.mmse = total` → existing `predictAlzheimers()` → `POST /predict`.
- **Navigation:** "MMSE Assessment · X of 11" progress bar, Back/Next; Next is disabled until the current section is complete; answers persist when navigating back.
- **Right panel during MMSE phase:** `EmptyState` accepts optional `title`/`description`/`showAnalyze` props; during the MMSE phase `App.tsx` renders a contextual variant ("MMSE Assessment / Complete the assessment to generate your screening result.") with the Analyze button hidden so the panel doesn't look disconnected from the assessment flow.
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
- **Reusable components:** `Layout`, `FormPanel`, `InputField`, `SelectField`, `AnalyzeButton`, `ResultCard`, `PredictionPieChart`, `ErrorMessage`, `EmptyState` (supports contextual `title`/`description`/`showAnalyze` variants), plus MMSE primitives (`GlassCard`, `ExaminerInstructions`, `PatientResponse`, `ExaminerScoring`, `AIAssessmentPanel`, `AIScoredResponse`, `useSpeechRecognition`, `SectionShell`, `inputClass`).

> **Do not redesign the application unless explicitly requested.**

---

## 9. Current Features (checklist)

- [x] Patient clinical input (age, sex, education_years, mmse, ses)
- [x] ML prediction (`POST /predict`)
- [x] Probability breakdown (pie chart + bars)
- [x] Confidence / `detection_percentage` display
- [x] Detection status (Alzheimer's detected / not)
- [x] MMSE questionnaire (11 sections, AI-assisted scoring, 0–30)
- [x] AI-assisted per-item evaluation (`POST /mmse/evaluate`, provider-agnostic: Ollama/Gemini)
- [x] Browser speech capture (Web Speech API) with typing fallback
- [x] Examiner review/override of AI scores (incl. low-confidence review flow)
- [x] MMSE score → existing prediction flow (single `mmse` field)
- [x] Drawing canvas for copying task (mouse + touch)
- [x] Training-time static SHAP artifacts (plots + pickle)
- [ ] Live per-patient SHAP explanations
- [ ] Vision-assisted figure evaluation (Question 11)
- [ ] Vision-assisted reading evaluation (Question 9)
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
| AI provider not configured (`AI_PROVIDER=none`) or Ollama not running | Medium | AI-scored items degrade to manual review | Yes (by design) | Configure provider env; manual fallback is built in |
| Real-model AI scoring untested on this machine (no Ollama runtime) | Medium | Only failure/validation paths exercised live | Yes until deployed | Test with a real provider (Ollama/Gemini) before release |
| AI is an assist signal only — never a diagnosis or clinical certainty | Medium | Misinterpretation risk | No (must keep disclaimers) | Keep confidence phrased as model signal; examiner review required |

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
| 2026-08-13 | `ccc45d3` | `fix(mmse): show contextual right panel during MMSE assessment phase` | EmptyState UX fix |
| 2026-08-13 | `73ef09c` | `feat(mmse): record patient responses separately from examiner scoring` | Response-recording UX |

---

## 15. Current Roadmap

### Immediate
- Supply the exact MMSE copying figure as an image asset, set `COPYING_REFERENCE_IMAGE` (`frontend/src/mmse/config.ts`); optionally configure `PLACE_ITEMS` answers per location.
- Configure a real AI provider (Ollama local, or Gemini with a backend `GEMINI_API_KEY`) and validate live AI-scored responses before release. `AI_PROVIDER`/model/base URLs live in backend env (never in React).

### Next
- Vision-assisted evaluation of the copied figure (Question 11) as a documented workflow (photograph → vision model → examiner review). Do NOT fold into the current AI-text evaluation.
- Live per-patient SHAP explanations (backend endpoint + frontend consumption).
- MoCA scoring (README V2 roadmap).

### Later
- Vision-assisted reading evaluation (Question 9).
- MRI/imaging, multimodal input (README V3/V4 roadmap).
- Larger/balanced dataset to improve Converted-class recall.

### Explicitly deferred
- Any change to the ML contract (`/predict`, model, SHAP).
- Automated figure scoring — pending the separate vision workflow.
- Automated vision scoring of the Reading instruction.

---

## 16. Agent Handoff Notes

### Last Completed Work
- **AI-assisted MMSE scoring:** Replaced examiner-manual scoring with a provider-agnostic AI evaluation flow (Patient response → `POST /mmse/evaluate` → validated structured score → 0–30 total). New `backend/src/ai_eval.py` service (Ollama `/api/chat` or Gemini `/chat/completions` via stdlib `urllib`, env config `AI_PROVIDER`/`OLLAMA_*`/`GEMINI_*`/`AI_TIMEOUT`, per-section prompts, strict `parse_ai_result` validation → 422 on invalid output, 503 on provider failure). `api.py` gained only `POST /mmse/evaluate`; `/predict` contract, model, and SHAP untouched. Frontend: `ItemState { response, status, aiScore, reviewRequired, reviewed, manual, error }`, `effectiveCorrect()` (manual overrides AI), `isItemFinalized()` (low-confidence items need examiner Accept/Override), `AIAssessmentPanel` (assessing/assessed/error states), `AIScoredResponse`, `useSpeechRecognition` (Web Speech API, typing fallback), sections 1–7 + 10 AI-scored; sections 8 (three-step command) and 9 (reading) examiner-observed; section 11 (copying) manual — vision deferred. Confidence is a model signal; threshold `AI_CONFIDENCE_REVIEW_THRESHOLD = 0.7`. `npm run build` (tsc + vite) passes.
- Earlier milestones: `590ff38` examiner-scored MMSE questionnaire; `ccc45d3` contextual MMSE right panel (EmptyState props); `73ef09c` patient-response recording separated from scoring.

### Current State
- Working tree: AI-assisted scoring work is **uncommitted** (see Files Recently Changed). Git branch `main`, tracking `origin/main`. Backend AI service implemented and validation-tested (mocks + live `/mmse/evaluate` with a mock Gemini provider → 200). Real Ollama/Gemini model output not yet exercised on this machine (no Ollama runtime).

### Next Recommended Task
- Configure a real AI provider and smoke-test live AI responses (correct/incorrect paths) before release.
- Supply the exact MMSE copying figure asset and set `COPYING_REFERENCE_IMAGE`.
- Optionally set per-location expected answers in `PLACE_ITEMS` (`frontend/src/mmse/config.ts`).
- Then consider live per-patient SHAP as the next feature.

### Files Recently Changed
- `backend/src/ai_eval.py` (new — AI evaluation service, provider-agnostic)
- `backend/src/api.py` (added `POST /mmse/evaluate` only; `/predict` untouched)
- `frontend/src/api.ts` (`evaluateMmseItem`, `MmseEvaluateRequest/Result`, `extractApiError`)
- `frontend/src/mmse/state.ts` (`ItemState`/`AIScore` model, `effectiveCorrect`, `isItemFinalized`)
- `frontend/src/mmse/config.ts` (`AI_CONFIDENCE_REVIEW_THRESHOLD`)
- `frontend/src/components/mmse/primitives.tsx` (`AIAssessmentPanel`, `AIScoredResponse`, `useSpeechRecognition`, `PatientResponse` with rightSlot)
- `frontend/src/components/mmse/sections.tsx` (AI-scored flow per section; manual command/reading/copying)
- Earlier: `frontend/src/App.tsx`, `frontend/src/components/index.ts`, `frontend/src/components/EmptyState.tsx`

### Tests Verified
- `npm run build` (`tsc && vite build`) — passes, no TypeScript errors.
- Dev server boots cleanly (`vite`), serves HTTP 200.
- Backend: 13 validation/failure checks pass (valid/incorrect, fenced JSON, invalid JSON, score mismatch, confidence range, correct-type, orientation_time server-side expected, `AI_PROVIDER=none`→503, unreachable→503, unsupported section→422, mock valid→200 dict, mock invalid JSON→422). Live: `/predict` 200 (unchanged contract); `/mmse/evaluate` via mock Gemini provider → 200 `{correct, score, confidence, reason}`.
- Frontend state logic verified by Node harness: totals 0/3/4/30, maxes per section sum 30, spell-World path 30, override wins, low-confidence gating (not finalized until reviewed), empty/incomplete sections gated. Real provider AI correctness untested (no Ollama runtime).

### Commit
- `590ff38` — `feat(mmse): add step-by-step MMSE assessment with examiner scoring`
- `c97ba47` — `docs: add AI_CONTEXT.md for agent context management`
- `24e662e` — `docs: record MMSE milestone commit hashes in AI_CONTEXT.md`
- `ccc45d3` — `fix(mmse): show contextual right panel during MMSE assessment phase`
- `73ef09c` — `feat(mmse): record patient responses separately from examiner scoring`
- Pending: `feat(mmse): add ai-assisted response scoring` (current uncommitted work)

### Push Status
- Pushed to `origin/main` successfully through `73ef09c`. Current AI-assisted work is uncommitted (see above); push after committing.

### Important Warnings
- No live SHAP endpoint exists — do not assume per-patient SHAP.
- Copying-figure asset is missing; do not substitute a generic pentagon.
- `probabilities` in `/predict` are position-indexed (`[0,1,2]`), not keyed by `model.classes_`.
- Do not retrain the model or modify the `/predict` contract without explicit instruction.
- AI results are an assist signal, never a diagnosis; confidence is a model signal, not clinical certainty. Keep these disclaimers in the UI.
- AI secrets (`GEMINI_API_KEY`, etc.) must stay backend-only (env/`.env`); never ship them in React or commit them.
- Real AI-model correctness is untested on this machine (no Ollama runtime) — only failure/validation paths were exercised.
- `README.md` has stale claims (e.g., Dataset.csv "not committed") — treat code as authoritative.
