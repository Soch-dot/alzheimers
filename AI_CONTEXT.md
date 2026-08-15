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
- **Assessment Details step (pre-MMSE):** the app now collects Age / Sex / Education (years) / SES on a dedicated first screen before the MMSE begins. Values live in App-level `formData` so they survive Assessment Details → MMSE → AI assessment → Summary → Analysis (no re-entry). "Restart Assessment" clears them back to defaults and returns to Assessment Details. The post-MMSE screen shows them as a read-only summary (no duplicate editable inputs).
- ML prediction via FastAPI `POST /predict` (Random Forest pipeline).
- Probability breakdown (pie chart + bars) and confidence/`detection_percentage` display.
- **AI-assisted 11-section MMSE questionnaire** (replaces the raw MMSE number input) with a **two-phase batch workflow**:
  1. **Collect responses** — the examiner/patient complete the whole questionnaire; the app only records responses (typed or speech). **No AI calls, no per-question assessment, no "Assessing…" states.**
  2. **Assess MMSE with AI** — one explicit button sends a **single batch** `POST /mmse/evaluate` with all collected responses; the backend returns per-item structured results. The examiner reviews (low-confidence → "Review required") and can override manually. Final `mmse` (0–30) flows into the existing `/predict`.
- Provider-agnostic AI service (Ollama or Gemini); failure/timeout always resolves to a clear error with Retry + Manual review — never an indefinite spinner. **Raw OS/network/backend error text is never shown in the normal UI** — it lives behind a collapsed "View technical details" disclosure or console.
- **Location-aware Orientation to Place:** the examiner configures the assessment location (state / county / town / building / floor) in a clean in-app form at the top of the section; those values are the reference answers for the batch and are never shown to the patient. No source-code paths or developer config exposed.
- **Semantic response evaluation:** the AI judges the *meaning* of a response, not its formatting (case/punctuation/spacing/number-as-words/synonyms are not penalized). Results display as "✓ Correct response" / "✕ Incorrect response" / "⚠ Review required" / "Response required".
- **Observation-based sections** (Three-Step Command, Reading) are clearly labelled as such, with an "AI vision assistance will be added in a later milestone" note — the examiner is not presented as the intended permanent scorer.
- **Q11 photo-based vision assessment (UI):** the Copying section now runs a photo flow — reference figure → patient copies on paper → examiner takes/uploads a photo → preview → explicit "Analyze Drawing" → `POST /mmse/copying/evaluate` → normalized result with confidence + `review_required` → examiner accepts/overrides. The on-screen drawing canvas was removed. Photo preview is an in-memory data URL only (never persisted to disk/localStorage/backend). No auto-call on photo selection.
- Browser speech capture (Web Speech API, no new dependency) with graceful degradation to typing; transcripts only populate the response field and never trigger AI.
- Training-time static SHAP artifacts (plots + `shap_data.pkl`). **There is no live per-patient SHAP endpoint.**
- **Assessment Mode + details persistence + input safeguards (UX milestone):** the app now starts on a Mode step (`AssessmentModePicker`) — Patient / Examiner / Examiner+Patient. A React context (`AssessmentModeContext`, `frontend/src/mmse/mode.ts`) gates role-specific UI: expected answers/hints, examiner instructions, AI score details + manual override, provider/model/technical info, assessment-location config, and Q11 camera/upload are all hidden in Patient mode; scoring is identical in every mode. The Assessment Details values (mode, age, sex, education_years, ses) persist to localStorage (`alzheimers_assessment_details_v1`) and auto-restore on reload (skipping the Mode step when valid); Restart/Re-take clears storage and returns to Mode. Numeric details are held as sanitized strings (digits only, leading zeros stripped, empty allowed — no "021"/0 coercion) and converted to numbers only in the `/predict` payload. Continue enables immediately via `detailsValid`.
- **Q6 Naming object library:** Naming (2 points, exactly 2) now randomly draws TWO distinct objects from a 12-object library (`frontend/src/mmse/namingLibrary.ts`, local SVGs under `frontend/public/objects/`). The two batch slot keys stay `naming.wristwatch`/`naming.pencil` (the backend only accepts those keys); the selected object's canonical answer is sent as `expected`, so deterministic `_object_verdict` scoring still applies (29 deterministic + 1 AI hybrid preserved). The examiner can change each selected object; expected answers are hidden in Patient mode.
- **Q11 client-side image processing:** `frontend/src/mmse/imageUtils.ts` validates (MIME jpeg/png/webp, ≤10 MB, min 800×600) and optimizes (downscale >2048 long side, never upscale, JPEG q0.82) the Q11 photo in the browser BEFORE upload; only the optimized data URL is sent to `/mmse/copying/evaluate`. Low-resolution photos are rejected with the exact message "Image resolution is too low for reliable assessment. Please take a clearer photo." Original/optimized dimensions + sizes are shown as `photoInfo` (never persisted). Frontend Q11 timeout is a named constant `VISION_CLIENT_TIMEOUT = 120000` ms (must stay above/equal the backend `VISION_TIMEOUT` 120s).
- **Mic control + Enter navigation + Patient-mode explainability (UX-fix milestone):**
  - **Mic control (`[🎙 Mic]`):** the microphone icon (inline SVG) and the word "Mic" live inside the SAME clickable button (`MicButton` in `primitives.tsx`, reused by `AIScoredResponse` and the Spell-World block) — never icon-only, never a second label elsewhere. It toggles to `[🎙 Stop]` while listening; speech behavior is unchanged (transcript populates the field, never auto-triggers AI; typing still available; unsupported-browser fallback message preserved). Verified at desktop AND 375px mobile width (icon + text remain visible, not clipped).
  - **Enter key advances sections:** a single shared `goToNext` path (`MMSEAssessment.tsx`) is used by BOTH the Next button and the Enter key. `SectionNavigationContext` (`primitives.tsx`) + `PatientResponseInput` make Enter advance on any single-line patient response input when `canAdvance` (section complete AND not AI-assessing). Enter does NOT advance while the section is incomplete, during batch assessment, or from a textarea (Writing keeps newline entry; verified). `goToNext` clamps at the Summary step so the last section cannot skip the final action.
  - **Patient-mode Next explainability:** root cause of the "all 5 place responses but Next disabled" issue — `sectionResponseCounts().orientationPlace` requires BOTH a configured location field AND a response, and pure Patient mode never sets the location (examiner form hidden), so `done` stays 0 and completion logic blocks Next; `buildBatchItems` also skips unconfigured place items so they can never be AI-scored. **Decision (option A):** Patient mode intentionally stays examiner-dependent — the block is kept and now EXPLAINED (in-section amber "Examiner required" notice on Orientation to Place and Copying, plus the nav message "An examiner is required for this section" instead of a generic "Complete all items to continue"). No scoring is fabricated or bypassed; Examiner + Patient remains the full flow (verified to the analysis form).
- **Full assessment-session persistence (refresh-safe):** the WHOLE in-progress assessment survives a browser refresh (`frontend/src/mmse/session.ts`, key `alzheimers_assessment_session_v1`, `SESSION_VERSION = 1`). One authoritative startup restore in `App.tsx` rebuilds the exact phase + MMSE step/phase/state; a 400ms-debounced save effect persists changes (skipped when the session is empty). Restoring an `'assessing'`/`'error'` MMSE phase maps to `'collect'`; Q11 photos are NEVER persisted (`previewData`/`previewName` stripped on save, `photoInfo` cleared on restore, restored as "Choose/Take Photo" while keeping the AI score so the running total stays accurate). A full "Restart Assessment" (buttons on Assessment Details and the Analysis form) calls `clearSession()` → Assessment Mode; MMSE "Re-take" clears only the MMSE part (mode + details preserved). The legacy `alzheimers_assessment_details_v1` key is migrated into the session once and removed; corrupt/incompatible sessions are rejected and cleared safely (never crash).

### In Progress
- None currently. (See handoff section.)

### Planned (intentionally postponed)
- Live per-patient SHAP explanations.
- MoCA scoring (mentioned in README roadmap).
- MRI/imaging and multimodal input (long-term README roadmap).

### Rejected / Out of Scope (do NOT implement unless explicitly reconsidered)
- Adding CDR (Clinical Dementia Rating) back as a feature — removed because it leaks the diagnosis (see Research Context).
- Sending individual MMSE answers to the backend or adding 11 new MMSE API fields — the API contract stays 5 fields. (Per-item responses DO go to the separate `/mmse/evaluate` AI service, never to `/predict`.)
- Automated vision scoring of the Reading instruction (Question 9) — currently examiner-observed; vision layer not implemented.

---

## 3. Architecture

### Frontend
- **Framework:** React 19, TypeScript (strict), Vite 7, Tailwind CSS 4, Framer Motion, Axios, Chart.js + react-chartjs-2.
- **Entry point:** `frontend/index.html` → `frontend/src/main.tsx` → `<App/>`.
- **Main components:** `App.tsx` (orchestration + shared state), `Layout`, `FormPanel`, `InputField`, `SelectField`, `AnalyzeButton`, `ResultCard`, `PredictionPieChart`, `EmptyState`, `ErrorMessage`, `LoadingSpinner` (unused), `AssessmentModePicker` (mode step), `AssessmentDetails` (pre-MMSE demographics), `MMSEAssessment` + MMSE subcomponents.
- **State management:** Local React state only. No router, no Redux/Zustand. App phases: `'mode' | 'details' | 'mmse' | 'form'` (in `App.tsx`); MMSE section state lives inside `MMSEAssessment`. Role gating is a React context (`AssessmentModeContext` from `src/mmse/mode.ts`), consumed by MMSE primitives/sections/summary — no prop drilling. The ENTIRE in-progress assessment session persists to a single versioned localStorage key `alzheimers_assessment_session_v1` (`frontend/src/mmse/session.ts`): `{ version, mode, details, appPhase, mmse: { step, phase, state } | null, mmseScore, result }`. Restore is one authoritative effect in `App.tsx`; saves are 400ms-debounced and skipped when empty. Q11 images are never stored; corrupt/incompatible sessions are cleared. The legacy `alzheimers_assessment_details_v1` key migrates into the session once and is removed.
- **API layer:** `frontend/src/api.ts` — typed axios client, base URL from `VITE_API_URL` (default `http://127.0.0.1:8000`).
- **Styling architecture:** Tailwind utility classes inline in JSX. No CSS modules, no central design-token file. Dark glassmorphism theme (see Design System).

### Backend
- **Framework:** FastAPI 0.121 + Pydantic v2 + Uvicorn.
- **Entry point:** `backend/src/api.py` (`uvicorn src.api:app`).
- **Routes:** `GET /` (health message), `POST /predict` (prediction), `POST /mmse/evaluate` (AI-assisted MMSE **batch** scoring), `POST /mmse/copying/evaluate` (Q11 vision figure-copying evaluation), `OPTIONS /predict` (CORS preflight).
- **AI service:** `backend/src/ai_eval.py` — provider-agnostic (`AI_PROVIDER=ollama|gemini|none`), **batch** request with per-item structured JSON results, strict per-item validation, per-section prompts. **Provider-aware batching:** Ollama evaluates the WHOLE batch with a SINGLE `/api/chat` call (one prompt containing every item + its section-specific rules; one structured JSON response with one entry per item — no per-item Ollama calls, no concurrent Ollama calls). Gemini keeps per-item parallel evaluation via stdlib `concurrent.futures` (`GEMINI_MAX_CONCURRENCY=8`). Uses only Python stdlib (no new deps). No patient data logged/persisted. Backend-only secrets via env / `.env` (python-dotenv).
- **Vision service (Q11):** `backend/src/vision_eval.py` + `backend/src/vision_image.py` — MMSE Question 11 figure-copying evaluation via `POST /mmse/copying/evaluate`. Provider abstraction built around an OpenAI-compatible multimodal `chat/completions` contract (`VISION_PROVIDER=ollama|gemini|openai`; exactly ONE provider per assessment, no voting/fallback). Shared layer owns the MMSE copying criterion prompt, normalized schema, strict validation, confidence/review, and error normalization; provider adapters only handle base URL/model/auth/payload/response extraction. `VISION_TIMEOUT` (default 120s) is separate from the text-MMSE timeouts. Images are processed in-memory with Pillow (no new deps, no multipart upload — raw binary or JSON base64 body). The trusted reference figure is loaded server-side from `frontend/public/mmse-copying-figure.png` (never from the client).
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
  → Assessment Details (frontend/src/components/AssessmentDetails.tsx, App.tsx 'details' phase)
  → MMSE questionnaire (App.tsx 'mmse' phase)
  → Analysis (App.tsx 'form' phase): read-only summary + MMSE score + Analyze
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
│   ├── src/api.py                    # FastAPI prediction API + /mmse/evaluate (batch) + /mmse/copying/evaluate (Q11 vision). /predict contract frozen.
│   ├── src/ai_eval.py                # AI-assisted MMSE BATCH evaluation service (provider-agnostic)
│   ├── src/vision_eval.py            # Q11 vision evaluation service (provider abstraction: ollama/gemini/openai; normalized result)
│   ├── src/vision_image.py           # Q11 image processing (validate/normalize/encode in-memory; trusted reference loader)
│   ├── src/test_hello.py             # shap/matplotlib import smoke test
│   ├── tests/test_vision_eval.py     # stdlib unittest: Q11 vision service (synthetic images only)
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
    ├── src/App.tsx                   # Main orchestration + form state + MMSE phase ('mode' | 'details' | 'mmse' | 'form')
    ├── src/api.ts                    # Axios client + PredictionResponse/PatientInput types + /mmse/evaluate batch client + VISION_CLIENT_TIMEOUT
    ├── src/mmse/state.ts             # MMSE state types, initial state, scoring, completion checks, MmsePhase
    ├── src/mmse/batch.ts             # Batch payload builder, per-item result applier, response-completeness helpers
    ├── src/mmse/config.ts            # MMSE config: objects, expected answers (non-location), LOCATION_FIELDS
    ├── src/mmse/mode.ts              # AssessmentMode types + AssessmentModeContext / useAssessmentMode
    ├── src/mmse/details.ts           # String-based DetailsDraft, DETAILS_FIELDS metadata, validation, localStorage persistence
    ├── src/mmse/namingLibrary.ts     # Q6 12-object naming library + selection helpers
    ├── src/mmse/imageUtils.ts        # Q11 client-side image validation + optimization (photoInfo)
    └── src/components/
        ├── index.ts                  # Barrel exports for all UI components
        ├── AssessmentModePicker.tsx  # Mode step (Patient / Examiner / Examiner+Patient)
        ├── AssessmentDetails.tsx     # Pre-MMSE demographics step (Age/Sex/Education/SES); Continue gated on validity
        ├── Layout.tsx, FormPanel.tsx, InputField.tsx, SelectField.tsx, AnalyzeButton.tsx,
        ├── ResultCard.tsx, PredictionPieChart.tsx, EmptyState.tsx, ErrorMessage.tsx, LoadingSpinner.tsx (unused)
        └── mmse/                     # MMSE questionnaire UI (two-phase: collect → batch assess)
            ├── MMSEAssessment.tsx    # Stepper container (intro → 11 sections → summary) + batch orchestrator
            ├── MMSEIntroduction.tsx, MMSESummary.tsx, Q11PhotoAssessment.tsx, primitives.tsx, sections.tsx, index.ts
            # sections.tsx: 11 sections; OrientationPlace hosts the examiner-only assessment-location form
```

### Important files — modification policy
| File | Purpose | Modification policy |
|---|---|---|
| `frontend/src/App.tsx` | App orchestration, form + MMSE phase state | Allowed; avoid unrelated refactoring |
| `frontend/src/api.ts` | API types + axios client | Allowed; keep `/predict` payload contract unchanged |
| `frontend/src/mmse/*` | MMSE logic/config/state/batch | Allowed; keep total 0–30 and 11-section structure |
| `frontend/src/components/mmse/*` | MMSE UI | Allowed; preserve design system |
| `frontend/src/components/index.ts` | Barrel exports | Allowed (add exports) |
| `backend/src/api.py` | Prediction API | `/predict` contract frozen; adding new endpoints (e.g. `/mmse/evaluate`, `/mmse/copying/evaluate`) is allowed |
| `backend/src/ai_eval.py` | AI-assisted MMSE evaluation service | Allowed (new) |
| `backend/src/vision_eval.py` | Q11 vision evaluation service | Allowed (new) |
| `backend/src/vision_image.py` | Q11 image processing + trusted reference loader | Allowed (new) |
| `backend/tests/*` | Backend tests | Allowed (add) |
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

### `POST /mmse/evaluate` (AI-assisted MMSE batch scoring — separate service)
- **Purpose:** Scores ALL collected MMSE responses in ONE request via an AI provider. Triggered only by the explicit "Assess MMSE with AI" action. Never sends item answers to `/predict`; only the final `mmse` total does.
- **Request body (batch):**
  ```json
  {
    "items": {
      "orientation_time.year": { "question": "What year is it?", "response": "2026", "expected": "" },
      "attention_spell_world.3": { "question": "Letter 3 of WORLD backwards", "response": "R", "expected": "R" },
      "naming.wristwatch": { "question": "What is this?", "response": "watch", "expected": "wristwatch" }
    }
  }
  ```
  - Keys are `"<section>.<item_key>"`. `section` is one of: `orientation_time, orientation_place, registration, attention_serial7, attention_spell_world, delayed_recall, naming, repetition, writing`. The backend derives `section`/`item_key` from the key; the frontend never sends empty/unanswered responses.
  - `expected` is the hidden evaluation context (examiner-only). For `orientation_time` the backend derives today's date-based expected answer server-side and ignores the client value. For `orientation_place` the frontend sends the examiner-configured assessment-location values as `expected`.
  - **Fully deterministic (never AI):** `orientation_time.*` (year / season / date / day / month) and `attention_serial7.*` (Serial-7s arithmetic: 93 / 86 / 79 / 72 / 65) and `attention_spell_world.*` (WORLD-backwards letters D / L / R / O / W, case-insensitive) are scored 100% deterministically in `backend/src/mmse_rules.py` — the date items against the server's current date/time, the serial-7 items by exact numeric comparison, the spell-world letters by per-item letter comparison (the frontend splits the full spelling response into single-letter items). These items NEVER reach the AI provider: `FULLY_DETERMINISTIC_SECTIONS` is disjoint from `AI_SECTIONS` in `ai_eval.py`, and the batch split happens before any provider call, so the model payload contains only AI-required items. Results always carry `confidence: 1.0` (never flagged for review) and an unparseable response scores `correct: false` — it never becomes a per-item error, so the Summary is never stuck.
  - **Hybrid (deterministic when safely provable, else AI):** `orientation_place.*` (state/county/town/building/floor vs the frontend-supplied examiner `expected` location), `registration.*`, `delayed_recall.*` (3 objects), `naming.*` (wristwatch/pencil), and `repetition.phrase` are scored deterministically when the answer is clearly right or clearly wrong, and routed to AI only when genuinely ambiguous. The rule engine (`_object_verdict` / `_repetition_verdict` in `mmse_rules.py`) returns a result dict (MATCH → correct / NO_MATCH → incorrect, `confidence` 1.0) or the `AMBIGUOUS` sentinel. Deterministic matching accepts semantic equivalents (articles stripped: "an apple" == "apple", "a wristwatch" == "wristwatch"), bounded equivalences (`wristwatch` ↔ `watch`), case/whitespace/punctuation normalization, and whole-word keyword containment. AMBIGUOUS (→ single item sent to AI): uncertain phrases ("I don't know", "not sure"...), generic/referential answers ("that fruit", "the thing used to tell time"), partial word overlap (e.g. "LA County" vs "Los Angeles County"), or an unconfigured `orientation_place` expected value. `writing.sentence` remains AI-only. See the section table below.
  - **Semantic evaluation:** the system prompt instructs the model to judge the *meaning* of the response — never marking incorrect solely for capitalization, punctuation, spacing, numbers written as words (e.g. "2026" vs "two thousand twenty-six"), or harmless synonyms.
- **Response (per-item structured results):**
  ```json
  {
    "items": {
      "orientation_time.year": { "correct": true, "score": 1, "confidence": 0.98, "reason": "..." }
    },
    "errors": {
      "naming.wristwatch": "model did not return valid JSON"
    }
  }
  ```
    - `correct` bool; `score` must equal 1/0 per correct; `confidence` in [0,1]; `reason` non-empty. Every result is validated; malformed output lands in `errors` with **no score assigned** (no silent scoring). Deterministic `orientation_time` results have `confidence` 1.0 and always appear in `items`.
    - `503` when AI is disabled (`AI_PROVIDER=none`) or the provider is unreachable for every item — only relevant when the request actually contains AI-required items (a deterministic-only batch succeeds even with `AI_PROVIDER=none`); `422` for a malformed request (no items supplied).
   - **Batching is provider-aware:** with Ollama the backend sends exactly ONE model call for the whole batch (single prompt + single JSON response); missing/malformed per-item entries become per-item `errors`, never silent scores. With Gemini it evaluates items in parallel internally (`concurrent.futures`, ≤8 workers). Either way it is still ONE frontend HTTP request.
- **Provider config (backend env only):** `AI_PROVIDER=ollama|gemini|none`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL` (default `gemma3`), `GEMINI_BASE_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `AI_TIMEOUT` (per provider call, default 30s), `OLLAMA_BATCH_TIMEOUT` (single Ollama batch call, default 175s, must stay under the frontend's 180s).
- **Frontend caller:** `evaluateMmseBatch()` in `frontend/src/api.ts` (axios timeout 180s; `isTimeoutError()` detects timeouts → "AI assessment timed out.").
- **Disclaimers:** AI result is an assist signal, never a diagnosis. Confidence is a model signal, not clinical certainty. `AI_PROVIDER=none` or an unreachable provider surfaces "AI assessment unavailable." with Retry/Manual-review in the UI.

### `POST /mmse/copying/evaluate` (Q11 vision-assisted figure copying — separate service)
- **Purpose:** Evaluates ONLY MMSE Question 11 (figure copying). The backend loads the trusted reference figure server-side from `frontend/public/mmse-copying-figure.png` (or `COPYING_REFERENCE_PATH`); the client NEVER supplies the reference. Patient drawing is processed in memory (Pillow) and discarded — never persisted, logged, or returned.
- **Request:** raw image bytes with `Content-Type: image/jpeg|image/png|image/webp` **or** JSON `{"image": "data:image/jpeg;base64,..."}` (or plain base64). No multipart (python-multipart is not installed).
- **Image handling:** validates MIME + size (`VISION_MAX_UPLOAD_BYTES`, default 10 MB), decodes via Pillow, normalizes EXIF orientation, preserves aspect ratio, downscales > `VISION_MAX_IMAGE_DIMENSION` (default 2048). **Blank / near-blank pre-check** (`has_drawing_content` in `vision_image.py`): a deterministic sanity gate BEFORE any provider call. It measures the fraction of pixels meaningfully darker than the estimated paper level (90th percentile grayscale). Conservative thresholds: `VISION_BLANK_INK_DELTA` (default 20 gray levels below paper) and `VISION_BLANK_MIN_INK_FRACTION` (default 0.5% of pixels). Blank/near-blank/ultra-low-contrast → HTTP 400 "No drawing detected. Please submit a clear photo of the patient's drawing." — the vision provider is NEVER invoked for a rejected image. A faint-but-visible pencil drawing passes (deliberately conservative to avoid over-rejection). Rejects: empty body, unsupported MIME, oversized, undecodable → HTTP 400 with a friendly message.
- **Response (normalized structured result):**
  ```json
  {
    "correct": true,
    "score": 1,
    "confidence": 0.91,
    "reason": "Both overlapping figures are present and the required intersection is preserved.",
    "review_required": false
  }
  ```
  - `score` = 0/1 and must agree with `correct`; `confidence` in [0,1]; `reason` non-empty. Strict validation: malformed provider output → NO score, HTTP 502 "Vision assessment returned an invalid result." Low confidence (`< AI_CONFIDENCE_REVIEW_THRESHOLD`, 0.7) → `review_required: true`.
  - Timeout (`VISION_TIMEOUT`, default 120s) → HTTP 504 "Vision assessment timed out." Provider unreachable/not configured → HTTP 503 "Vision assessment unavailable." These are separate from the text-MMSE 175/180s budgets.
- **Provider config (backend env only):** `VISION_PROVIDER=ollama|gemini|openai`, `OLLAMA_VISION_MODEL` (default `gemma3`), `GEMINI_API_KEY`/`GEMINI_VISION_MODEL`/`GEMINI_BASE_URL`, `OPENAI_API_KEY`/`OPENAI_VISION_MODEL`/`OPENAI_BASE_URL`, `VISION_TIMEOUT`, `VISION_MAX_UPLOAD_BYTES`, `VISION_MAX_IMAGE_DIMENSION`, `VISION_BLANK_INK_DELTA`, `VISION_BLANK_MIN_INK_FRACTION`, `COPYING_REFERENCE_PATH`.
- **MMSE copying criterion:** geometric structure, presence of both figures, and required overlap/intersection ONLY. Explicitly NOT scored: artistic quality, handwriting, penmanship, aesthetics, paper cleanliness, color, line thickness, "looks professional". No invented clinical criteria.
- **Frontend:** NO UI changes in the backend milestone; the camera/upload interface is a later UI/UX prompt. No frontend code consumes this endpoint yet.

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
- **Two-phase workflow (batch — NOT per-question):**
  1. **Collect responses.** The patient/examiner move through all 11 sections; the app only records responses (typed or speech). **No AI calls, no per-item assessment, no confidence display.** Sections show "n / max responses" and a "✓ Response recorded" marker.
  2. **Assess MMSE with AI.** On the Summary the examiner taps the single "Assess MMSE with AI" button → **one** batch `POST /mmse/evaluate` with every applicable, answered response → per-item results are applied → results shown per item ("✓ Correct response / ✕ Incorrect response / AI confidence / ⚠ Review required") → MMSE total updates → "Continue to Analysis" → existing `/predict`.
  - The batch is built by `buildBatchItems()` (`src/mmse/batch.ts`); results applied by `applyBatchResultsToDraft()`. Editing a response after assessment clears its AI score and the item is re-assessed on the next batch (only missing/unscored items are re-sent).
- **Result states per response:** a correct/incorrect verdict; a low-confidence item → "⚠ Review required" (not finalized until Accept/Override); an empty response → "Response required" (empty responses are never sent to the batch); a per-item failure → "AI assessment unavailable for this item." with technical detail behind a disclosure.
- **Structure:** 11 sections, max total **30**:
  1. Orientation to Time (5) — year/season/date/day/month; **deterministic** against server-derived date values (never AI).
  2. Orientation to Place (5) — state/county/town/building/floor; **location-aware**: the examiner configures the assessment location in a clean in-app form at the top of the section (examiner-only), and those values are used as the reference answers. **Hybrid** — deterministic when the answer clearly matches/mismatches the configured location; ambiguous answers routed to AI. Patient-facing UI never shows them. No source-code paths / developer config exposed.
  3. Registration (3) — 3 objects (`REGISTRATION_OBJECTS`, configurable); **hybrid** (deterministic for clear object answers, e.g. "an apple" ≡ "Apple"; ambiguous → AI); objects reused for Delayed Recall.
  4. Attention & Calculation (5) — Serial-7s (5 fields) or spell WORLD backwards (full response split into per-letter scores); **deterministic** (serial-7 exact numeric; spell-world per-letter D/L/R/O/W, case-insensitive); expected sequence examiner-only.
  5. Delayed Recall (3) — recalls the same Registration objects; **hybrid** (same deterministic rules as Registration).
  6. Naming (2) — the examiner assigns each slot ONE object from a **12-object library** (`wristwatch, pencil, key, cup, ball, book, scissors, comb, fork, chair, apple, umbrella`; local SVGs at `frontend/public/objects/<id>.svg`); two distinct objects are drawn randomly per assessment (`createInitialNamingState` / `pickNamingObjects`), and the examiner can change either slot. The batch slot keys REMAIN `naming.wristwatch` / `naming.pencil` (the backend `evaluate_naming` accepts only these keys); each slot sends the selected object's canonical answer as `expected`, so `_object_verdict` deterministic scoring still applies (wristwatch≡watch, articles stripped; ambiguous descriptions → AI). Expected answers hidden in Patient mode. Q6 total stays exactly 2 points.
  7. Repetition (1) — "No ifs, ands, or buts."; **hybrid** (exact phrase match deterministic; partial/uncertain → AI).
  8. Three-Step Command (3) — right hand / fold / floor; **observation-based** (badge + "AI vision assistance will be added in a later milestone"). Examiner records observations.
  9. Reading (1) — "CLOSE YOUR EYES"; **observation-based** + optional note. Automated vision scoring NOT implemented (documented limitation).
  10. Writing (1) — sentence with noun and verb; **AI-scored on that criterion only** (never spelling/grammar/handwriting/intelligence).
  11. Copying (1) — reference figure + **photo-based vision assessment** (patient copies on paper → examiner photo → `POST /mmse/copying/evaluate` → normalized result → examiner accepts/overrides). Vision workflow complete (backend + UI); placeholder remains only if the figure asset is missing (not the case — asset is bundled).
- **AI confidence:** model/service signal only, never clinical certainty. Confidence < `AI_CONFIDENCE_REVIEW_THRESHOLD` (0.7, config.ts) → item flagged "⚠ Review required"; it does not count as complete until the examiner accepts the AI result or overrides.
- **Failure handling (never an indefinite spinner):** the batch resolves to success, error, or timeout.
  - Provider down/config off → Summary: "AI assessment unavailable" + "The selected AI provider is currently unavailable." + [Retry]. The raw OS/network error is **never shown in the normal UI** — it sits behind a "View technical details" disclosure.
  - Timeout (axios 90s) → "AI assessment timed out." + [Retry].
  - Partial failure → "Some responses could not be assessed." + [Review items] (+ re-assess for the still-missing items). Per-item failures show "AI assessment unavailable for this item." with technical detail behind a disclosure. Invalid AI output is never scored silently (goes to `errors`, item marked `error`).
- **Frontend state:** `src/mmse/state.ts` (`MMSEState` + `createInitialMMSEState`). `ItemState { response, status, aiScore, reviewRequired, reviewed, manual, error }` keeps response text separate from the score. `effectiveCorrect()` = manual verdict wins over AI; `isItemFinalized()` gates section completion (AI finalized unless low-confidence-unreviewed). `MmsePhase = collect | assessing | assessed | error` drives the two-phase UI. `MMSEState.location` holds the examiner-configured assessment location (reference answers for Orientation to Place). `NamingState { watch, pencil, watchObject, pencilObject }` tracks the two random library objects. Q11 uses a dedicated `CopyingState { status: empty|photo|analyzing|assessed|error, previewData, previewName, aiScore, reviewRequired, reviewed, manual, errorKind, errorDetail, photoInfo }`; `copyingEffective()`/`isCopyingFinalized()` mirror the ItemState helpers. `previewData` is an in-memory data URL — never persisted.
- **Assessment modes:** `AssessmentModeContext` (`src/mmse/mode.ts`) exposes `{ mode, setMode }` where `mode: 'patient' | 'examiner' | 'both'`. `isExaminerView(mode)` is true for `'examiner'` and `'both'`. Examiner-only UI is gated in `primitives.tsx` (`ExaminerInstructions`, `AIResultPanel` details/manual override/retry, `AIScoredResponse` expected answers, `SectionShell` scores/instructions), `sections.tsx` (naming object selector + expected, orientation-place location config, Q11 camera/upload), and `MMSESummary.tsx` (score breakdown, technical details, review messaging). Patient mode shows the question + response UI without any expected answers or AI internals. Scoring and the batch payload are identical in all modes.
- **Speech capture:** `useSpeechRecognition()` in `primitives.tsx` uses the native Web Speech API (`SpeechRecognition`/`webkitSpeechRecognition`), no new dependency. A transcript only populates the response field — it never triggers AI. Unsupported browsers degrade to typing with a notice.
- **Flow to the API:** Assessment Details (pre-MMSE, `App.tsx` 'details' phase) → MMSE → MMSE Summary → "Continue to Analysis" → `MMSEAssessment.onComplete(total)` → `App.handleMmseComplete` sets `formData.mmse = total` → Analysis screen (read-only demographics + MMSE score + Analyze) → existing `predictAlzheimers()` → `POST /predict`.
- **Assessment Details step:** first screen of the flow (`phase === 'details'`). Collects Age (50–100), Sex (Male=1/Female=0), Education years (0–25), SES (1–5) using the same `InputField`/`SelectField` components and field semantics as the original post-MMSE form. The form starts with neutral/invalid defaults (`age 0, ses 0`) so the `[ Continue to MMSE ]` button stays disabled until every field is within range; the MMSE cannot be started with missing/invalid details. Values are stored in App-level `formData` and preserved across forward/backward navigation through the whole flow. On the Analysis screen they appear as a read-only summary (`Age / Sex / Education / SES`), not editable inputs. "Restart Assessment" (`handleRestart`) resets them to the same neutral defaults and returns to the 'details' phase. **Input handling (UX milestone):** numeric fields are `type="text"` + `inputMode="numeric"` and stored as sanitized strings (digits only, leading zeros stripped, empty allowed) in a `DetailsDraft` (`src/mmse/details.ts`); they convert to numbers ONLY when building the `/predict` payload (`detailsToPatientInput`). Field metadata (`required`, range hints) lives in `DETAILS_FIELDS`. Continue enables immediately per `detailsValid`. The app now has a **Mode step first** (`phase === 'mode'`, `AssessmentModePicker`); approved details persist to localStorage (`alzheimers_assessment_details_v1`) and auto-restore (skipping Mode when valid); Restart clears storage.
- **Navigation:** "MMSE Assessment · X of 11" progress bar, Back/Next; during Phase 1 Next is enabled once the section's responses are complete (`isSectionResponseComplete`); after assessment it requires finalized scores (`isSectionComplete`); navigation is locked while the batch runs. Answers persist when navigating back.
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
- **Reusable components:** `Layout`, `FormPanel`, `InputField`, `SelectField`, `AnalyzeButton`, `ResultCard`, `PredictionPieChart`, `ErrorMessage`, `EmptyState` (supports contextual `title`/`description`/`showAnalyze` variants), plus MMSE primitives (`GlassCard`, `ExaminerInstructions`, `PatientResponse`, `ExaminerScoring`, `AIResultPanel`, `AIScoredResponse`, `useSpeechRecognition`, `SectionShell`, `inputClass`).
- **MMSE section badges:** `SectionShell` shows a small pill next to each section title — blue "AI-assessed" for the 8 AI sections, amber "Observation-based" for Three-Step Command / Reading / Copying.
- **Assessment-location form:** a clean examiner-only form (state / county / town / building / floor) at the top of the Orientation to Place section; styled with the standard `inputClass` inputs. Expected answers are never rendered in the patient-facing question/response UI.

> **Do not redesign the application unless explicitly requested.**

---

## 9. Current Features (checklist)

- [x] Patient clinical input (age, sex, education_years, mmse, ses)
- [x] Assessment Details step (Age/Sex/Education/SES) collected BEFORE the MMSE; values preserved across the flow; read-only summary on Analysis; restart clears them
- [x] **Assessment Mode step (Patient / Examiner / Examiner+Patient)** with role-gated UI (expected answers, instructions, AI details/override, location config, Q11 camera hidden in Patient mode; scoring identical)
- [x] **Details persistence:** approved mode + demographics restored from localStorage; Restart clears; numeric fields sanitized strings (no leading zeros / no "0" coercion), converted only for `/predict`
- [x] **Q6 Naming object library:** 12 local SVG objects, 2 drawn randomly per assessment, examiner-selectable, deterministic scoring preserved via fixed slot keys + `expected`
- [x] ML prediction (`POST /predict`)
- [x] Probability breakdown (pie chart + bars)
- [x] Confidence / `detection_percentage` display
- [x] Detection status (Alzheimer's detected / not)
- [x] MMSE questionnaire (11 sections, 0–30) with **two-phase batch workflow** (collect → one "Assess MMSE with AI" action)
- [x] AI-assisted batch evaluation (`POST /mmse/evaluate`, provider-agnostic: Ollama/Gemini, per-item results)
- [x] No AI calls while answering — typing/speech/Enter never triggers evaluation
- [x] Location-aware Orientation to Place (examiner-configured assessment location; expected answers never shown to the patient)
- [x] Semantic response evaluation (meaning, not formatting) with "✓ Correct response / ✕ Incorrect response / ⚠ Review required / Response required" display
- [x] Observation-based sections clearly labelled (Command/Reading; vision noted as a later milestone)
- [x] Friendly error states with Retry + "View technical details" disclosure (raw OS/network errors never in the normal UI)
- [x] Browser speech capture (Web Speech API) with typing fallback
- [x] Examiner review/override of AI scores (incl. low-confidence review flow)
- [x] Timeout/failure handling with Retry + Manual review (never stuck loading)
- [x] MMSE score → existing prediction flow (single `mmse` field)
- [x] **Q11 photo-based vision assessment UI** (take/upload photo → preview → "Analyze Drawing" → normalized result → accept/override; on-screen canvas removed)
- [x] **Q11 client-side image validation/compression** (MIME/size/min-resolution gates, downscale >2048 never upscale, JPEG q0.82, photoInfo display, `VISION_CLIENT_TIMEOUT`)
- [x] Training-time static SHAP artifacts (plots + pickle)
- [ ] Live per-patient SHAP explanations
- [x] Vision-assisted figure evaluation (Question 11)
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
| Partial batch failure (invalid AI output / provider error for some items) | Medium | Those items stay unscored until Retry/manual | Yes (by design) | Per-item errors surfaced in UI; no silent score |
| AI is an assist signal only — never a diagnosis or clinical certainty | Medium | Misinterpretation risk | No (must keep disclaimers) | Keep confidence phrased as model signal; examiner review required |
| Q11 vision UI not wired (backend milestone only) | **Fixed** | Camera/upload/preview/reviewer UI implemented | — | `Q11PhotoAssessment.tsx` + `POST /mmse/copying/evaluate` |
| Blank Q11 submissions were reaching the vision model | Fixed | Blank canvas could be scored as correct | — | Pre-check gate added (`has_drawing_content`), verified blank → 400 before provider |

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
| 2026-08-13 | `1235955` | `feat(mmse): add ai-assisted response scoring` | AI-assisted per-item scoring (pre-batch) |
| 2026-08-13 | `8cadbc5` | `refactor(mmse): batch AI assessment with a single evaluate request` | Two-phase batch workflow |
| 2026-08-13 | `babcb03` | `fix(mmse): location-aware place, friendly errors, observation labels` | MMSE UX refinements |
| 2026-08-13 | `2abab40` | `fix(mmse): sequential Ollama batch eval with 180s frontend timeout` | Provider-aware concurrency + batch timeout |
| 2026-08-13 | `9f29276` | `fix(mmse): add exact copying figure reference asset` | Section 11 reference figure wired |
| 2026-08-13 | `6943501` | `perf(mmse): reduce local ai assessment latency` | Ollama evaluates the whole batch in ONE model call |
| 2026-08-13 | `4fdc6fe` | `feat(mmse): add q11 vision evaluation service` | Backend vision endpoint + provider abstraction |
| 2026-08-13 | `177c62f` | `fix(mmse): reject blank q11 submissions before vision` | Blank/near-blank pre-check gate |
| 2026-08-13 | `ee64a7c` | `feat(mmse): add q11 photo assessment ui` | Q11 camera/upload → analyze → review UI |
| 2026-08-13 | `cb2b7f0` | `feat(ui): collect assessment details before mmse` | Pre-MMSE Assessment Details step |
| 2026-08-13 | `9345169` | `docs: record assessment details before mmse commit hash` | Assessment Details docs |
| 2026-08-14 | `0c97278` | `fix(mmse): correct deterministic orientation scoring` | Orientation to Time is deterministic, never AI |
| 2026-08-14 | `f445042` | `docs: record deterministic orientation scoring commit hash` | Deterministic scoring docs |
| 2026-08-14 | `595d95c` | `fix(mmse): score serial seven deterministically` | Serial-7s arithmetic is deterministic, never AI |
| 2026-08-14 | `e9f326b` | `docs: record serial seven scoring commit hash` | Serial-7 deterministic scoring docs |
| 2026-08-14 | `d717ad3` | `perf(mmse): expand deterministic scoring coverage` | Hybrid MMSE scoring: 29/30 items deterministic, only writing + ambiguous → AI |
| 2026-08-14 | `2b0dc75` | `docs: record deterministic coverage commit hash` | Expanded deterministic coverage docs |
| 2026-08-14 | `e6fd581` | `fix(ui): start assessment details empty` | Neutral/invalid defaults so Continue is disabled until valid |
| 2026-08-14 | `907483a` | `feat(ux): add assessment modes persistence and input safeguards` | Assessment modes, details persistence, Q6 library, Q11 image processing |
| 2026-08-15 | `e4cfe16` | `fix(ux): complete assessment mode and input safeguards` | Details persistence guard + `VISION_CLIENT_TIMEOUT` 120000 |
| 2026-08-15 | `fe4d253` | `fix(ux): improve mic controls and keyboard navigation` | Mic icon+text control, Enter-key section advance, Patient-mode Next explainability |

---

## 15. Current Roadmap

### Immediate
- Configure a real AI provider (Ollama local, or Gemini with a backend `GEMINI_API_KEY`) and validate live AI-scored responses before release. `AI_PROVIDER`/model/base URLs live in backend env (never in React). (Ollama + Gemma 3 4B now configured and verified on this machine.)
- ~~Supply the exact MMSE copying figure as an image asset, set `COPYING_REFERENCE_IMAGE` (`frontend/src/mmse/config.ts`).~~ **Done** — `frontend/public/mmse-copying-figure.png`, `COPYING_REFERENCE_IMAGE = '/mmse-copying-figure.png'`.

### Next
- ~~Vision-assisted evaluation of the copied figure (Question 11) as a documented workflow (photograph → vision model → examiner review).~~ **Done** — backend (`POST /mmse/copying/evaluate`) + frontend UI (`Q11PhotoAssessment.tsx`) both complete. Do NOT fold into the text-AI evaluation.
- Live per-patient SHAP explanations (backend endpoint + frontend consumption).
- MoCA scoring (README V2 roadmap).

### Later
- Vision-assisted reading evaluation (Question 9).
- MRI/imaging, multimodal input (README V3/V4 roadmap).
- Larger/balanced dataset to improve Converted-class recall.

### Explicitly deferred
- Any change to the ML contract (`/predict`, model, SHAP).
- Automated vision scoring of the Reading instruction.

---

## 16. Agent Handoff Notes

### Last Completed Work
- **Full assessment-session persistence (UX-fix milestone, `fix(ux): persist and restore active assessment session`):** Frontend-only. A browser refresh now restores the exact in-progress location (Assessment Mode / Details / MMSE step / Analysis form) with all collected responses and AI results. Verified in headless Chrome (CDP, no new deps) — **41/41 checks PASS** (test matrix A–G + fresh/corrupt/legacy/re-take scenarios).
  - **New `frontend/src/mmse/session.ts`:** single source of truth. `SESSION_KEY = 'alzheimers_assessment_session_v1'`, `SESSION_VERSION = 1`, legacy `alzheimers_assessment_details_v1` migrated once and removed. Schema: `{ version, mode, details, appPhase ('mode'|'details'|'mmse'|'form'), mmse: { step (0 intro .. 12 summary), phase (MmsePhase), state (MMSEState) } | null, mmseScore, result }`. Structural validators (`isRecord`, `isScoreMark`, `isAIScore`, `isItemState`, `isItemStateArray`, `isStringMap`, `isMMSEState`, `isDetailsDraftShape`, `isPredictionResponse`, `isValidSession`) reject corrupt/incompatible payloads — `loadSession` clears invalid keys so the app starts clean at Assessment Mode, never crashes. `appPhase` `'mmse'`/`'form'` requires valid details; `'form'` requires a numeric `mmseScore`.
  - **Q11 refresh rule:** `previewData` (data URL) and `previewName` are stripped on EVERY save; `photoInfo` cleared on restore. A saved non-`'empty'` Q11 status restores as `'empty'` ("Choose/Take Photo") but keeps `aiScore`/`manual`/`reviewRequired`/`reviewed` so the running total stays accurate (verified: assessed Summary total 12 preserved before and after refresh). If the examiner re-uploads a photo, `readFile` wipes the AI score → re-analysis required.
  - **`App.tsx`:** ONE authoritative startup restore effect (`restored` flag gates saves); 400ms-debounced save effect skipped when the session is empty (`isEmptySession`). `handleMmseComplete` captures step 12 + `mmseScore`; `handleMmseStateChange` (useCallback) mirrors MMSE step/phase/state into the session; `handleRetake` clears MMSE + saves synchronously (instant refresh lands on a fresh intro); `handleRestart` calls `clearSession()` and returns to Assessment Mode. New "Restart Assessment" button on the Assessment Details screen (next to "← Back") and the Analysis form (next to "Re-take Assessment"). `MMSEAssessment` receives `initialStep`/`initialPhase`/`initialState`/`onSessionStateChange` and debounces its own 400ms state callback; restored `'assessing'`/`'error'` phases map to `'collect'` (in-flight/failed batch results are gone).
  - **Regression:** old details persistence not broken (superseded via migration); exactly one restore path; no redesign (only two functional text buttons added); `/predict`, MMSE scoring/rules, Q11 provider, ML, SHAP untouched.
  - **Verified:** `npm run build` passes; backend suite 85/85 passes (no backend changes). **41/41 CDP checks:** fresh mode → refresh → mode (no session key); corrupt stale-version + unparseable sessions → Assessment Mode, key cleared, no crash; legacy migration → details (age 70) + old key removed; Back preserves session; details refresh keeps values; TEST A (Section 2 responses + location restored), TEST B (Naming Q6 objects preserved), TEST C (Three-Step Command marks remain, Next enabled), TEST D (assessed Summary total 12 preserved), TEST E (Analysis form + details preserved), TEST F (Q11 → no restored image, Choose/Take Photo shown, new photo selectable), TEST G (Restart → Assessment Mode, session key removed), plus re-take (MMSE cleared, details+mode preserved, refresh → fresh intro).
- **Mic control (icon + text), Enter navigation, Patient-mode Next explainability (UX-fix milestone, `fix(ux): improve mic controls and keyboard navigation`):** Frontend-only. Verified in headless Chrome (CDP, no new deps) — **32/32 checks PASS**.
  - **Mic control (`primitives.tsx`):** new `MicButton` — the microphone icon (inline SVG, no new dependency) and the word "Mic" are INSIDE the same clickable control (`[🎙 Mic]`), toggling to `[🎙 Stop]` while listening. Reused by `AIScoredResponse` and the Spell-World block (both previously text-only "Mic"). `aria-label`/`title` added. Speech behavior unchanged (transcript populates the field only; never auto-triggers AI; typing available; unsupported-browser fallback message preserved). Single-line input right padding widened `pr-16 → pr-20` so the wider control never overlaps typed text. Verified at desktop AND 375px mobile width (icon + "Mic" text both visible, control not clipped), plus Mic→Stop→Mic toggle.
  - **Enter key advances sections (`MMSEAssessment.tsx` + `primitives.tsx`):** one shared `goToNext` (clamps at `SUMMARY_STEP`) used by BOTH the Next button and the Enter key via `SectionNavigationContext` + `PatientResponseInput`. Enter advances only when `canAdvance` (section complete AND not AI-assessing). Does NOT advance while the section is incomplete, during batch assessment, or from textarea inputs (Writing keeps newline entry; verified value preserved, no advance). Verified: Orientation-to-Time 5 responses + Enter → section 2; Registration 3 + Enter → section 4 (Attention); Attention complete + Enter → section 5; Enter with 4/5 responses does NOT advance; last section caps at Summary (no skip of the final action).
  - **Patient-mode Next bug — root cause:** `sectionResponseCounts().orientationPlace` (in `batch.ts`) counts an item ONLY when its location field is configured AND the response is non-empty; pure Patient mode never sets the location (the examiner form is hidden), so `done` stays 0/5 even with all 5 responses filled → `isSectionResponseComplete` false → Next disabled. Additionally `buildBatchItems` skips unconfigured place items (never AI-scored) and `isSectionComplete('orientationPlace')` needs every item finalized, so the block persists after assessment. **Decision (option A):** Patient mode intentionally stays examiner-dependent — kept the block, NO scoring fabricated or bypassed. Added clarity: an amber in-section **"Examiner required"** notice on Orientation to Place and Copying (explains the examiner must set the location / photograph the drawing, and to switch to Examiner or Examiner+Patient), and the nav message now reads **"An examiner is required for this section"** instead of a generic "Complete all items to continue". Enter also does NOT bypass the block (verified).
  - **Regression:** Examiner + Patient full flow verified end-to-end to the analysis form (location config visible, naming dual UI, Q11 analyze round-trip, batch assessment ~8–10s on the warm model, `/ 30` total). MMSE scoring, deterministic rules, AI batch architecture, Q6 2-point scoring, Q11 provider/compression, `/predict`, ML, SHAP, Ollama/Gemma untouched. `npm run build` passes; backend suite 85/85 passes (no backend changes).
- **Assessment modes + details persistence + input safeguards + Q6 library + Q11 image processing (UX milestone):** Frontend-only (plus the previously-committed backend Q11 `VISION_TIMEOUT`). Delivered as ONE commit `feat(ux): add assessment modes persistence and input safeguards`.
  - **New `frontend/src/mmse/mode.ts`:** `AssessmentMode = 'patient' | 'examiner' | 'both'`, `ASSESSMENT_MODES` (with `description` for the picker), `DEFAULT_ASSESSMENT_MODE = 'examiner'`, `isAssessmentMode`/`isExaminerView`, `AssessmentModeContext`, `useAssessmentMode()`.
  - **New `frontend/src/mmse/details.ts`:** string-based `DetailsDraft` (all fields `string | number`); `EMPTY_DETAILS` (empty strings); `DetailFieldConfig` + `DETAILS_FIELDS` (age 50–100, sex 1/0, education_years 0–25, ses 1–5 — every field required); `sanitizeNumericInput` (digits only, leading zeros stripped); `parseField`/`detailsValid`; `detailsToPatientInput` (converts to numbers only for `/predict`); `StoredDetails` + `loadStoredDetails`/`saveStoredDetails`/`clearStoredDetails` for localStorage key `alzheimers_assessment_details_v1` (mode + age + sex + education_years + ses ONLY — never MMSE answers/images/payloads). Restore only applies when `detailsValid`.
  - **New `frontend/src/components/AssessmentModePicker.tsx`:** Patient / Examiner / Examiner+Patient cards (title + description + radio) + Continue; exported from the barrel.
  - **`App.tsx`:** `phase = 'mode' | 'details' | 'mmse' | 'form'` (default 'mode'); mode + string-based details state restored from localStorage (skips Mode when a valid draft is stored); save-on-change effect; `handleRestart` clears localStorage + returns to 'mode'; `/predict` payload built via `detailsToPatientInput` (still exactly `{age, sex, education_years, mmse, ses}`). `AssessmentModeContext.Provider` wraps the layout so MMSE components consume the role without prop drilling.
  - **`AssessmentDetails.tsx`:** renders from `DETAILS_FIELDS` metadata (labels, options, placeholder, required), `Back` button, Continue enabled immediately per `detailsValid`. **`InputField.tsx`:** numeric fields `type="text"` + `inputMode="numeric"` for clean string handling.
  - **Mode gating (`primitives.tsx`, `sections.tsx`, `MMSESummary.tsx`):** patient mode hides expected answers/hints, `ExaminerInstructions`, AI score details/confidence/provider/technical info, manual override + retry, assessment-location config, and Q11 camera/upload; scoring and the batch payload are identical in every mode. Summary: breakdown rows + technical error details examiner-only; total always shown.
  - **Q6 naming library:** `frontend/src/mmse/namingLibrary.ts` — `NAMING_OBJECTS` (wristwatch, pencil, key, cup, ball, book, scissors, comb, fork, chair, apple, umbrella) each with id/name/image/expected; `DEFAULT_NAMING_SELECTION`; `getNamingObject`; `pickNamingObjects` (2 distinct random). `state.ts`: `NamingState { watch, pencil, watchObject, pencilObject }`; `createInitialNamingState` picks two random objects. `batch.ts`: slots keep keys `naming.wristwatch`/`naming.pencil` but send `expected = getNamingObject(slot).expected` (backend `_object_verdict` matches against `expected`, so deterministic scoring holds). `sections.tsx`: NamingSection shows the object image + "What is this?", examiner-only object selector + expected answer. Removed unused `NAMING_ITEMS` from `config.ts`. Created **12 local SVG assets** at `frontend/public/objects/<id>.svg` (no external URLs).
  - **Q11 image processing:** `frontend/src/mmse/imageUtils.ts` — `processPhoto(file)` validates MIME (`image/jpeg|png|webp` or extension), size (≤10 MB), min resolution 800×600 (low-res rejected with EXACT message "Image resolution is too low for reliable assessment. Please take a clearer photo."), downscales long side >2048 (aspect preserved, NEVER upscale), re-encodes JPEG q0.82, returns `{dataUrl, info}` (`PhotoInfo` = original/optimized width/height/bytes + `wasOptimized`). `state.ts` `CopyingState.photoInfo`. `Q11PhotoAssessment.tsx` uses `processPhoto`; shows original → optimized dimension/size line; only the optimized data URL is sent. `api.ts`: `VISION_CLIENT_TIMEOUT = 120000` named constant (user-specified, matching backend `VISION_TIMEOUT` 120s).
  - **Verified:** `npm run build` passes; backend suite 85/85 passes (no backend changes beyond the pre-existing `vision_eval.py` timeout). No `/predict`, ML, SHAP, `mmse_rules.py`, `ai_eval.py`, or provider changes.
- **Assessment Details pre-MMSE step (feat(ui): collect assessment details before mmse):** Frontend-only. Added a first screen that collects Age / Sex / Education (years) / SES BEFORE the MMSE, so the demographics no longer appear as editable inputs on the post-MMSE screen.
  - **New `frontend/src/components/AssessmentDetails.tsx`:** uses the existing `FormPanel`/`InputField`/`SelectField` components. Fields: Age (`min 50 / max 100`), Sex (SelectField Male=1 / Female=0), Education (years, `min 0 / max 25`), SES (`min 1 / max 5`). Subtitle "Enter the information required for the risk assessment." `[ Continue to MMSE ]` button is disabled until every field is within range (`valid`), then calls `onContinue` → `App.tsx` switches to the `'mmse'` phase.
  - **`App.tsx` changes:** `phase` union widened from `'mmse' | 'form'` to `'details' | 'mmse' | 'form'`, defaulting to `'details'`. `formData` stays in App state so values survive the whole flow (Assessment Details → MMSE → AI assessment → Summary → Analysis) and are never lost on navigation. New `handleRestart` resets `formData` to defaults (age 70, sex 1, education_years 12, ses 2, mmse 0) and returns to the `'details'` phase (replaces the old `handleRestartMmse` which only reset `mmse`). The post-MMSE Analysis screen now shows a **read-only "Assessment Details" summary card** (Age / Sex / Education / SES) + the MMSE Score box + `AnalyzeButton`; the duplicate editable `InputField`/`SelectField` grid was removed. Right panel shows a contextual `EmptyState` for the `'details'` phase ("Assessment Details / Enter the patient's assessment details to begin the screening.", Analyze hidden).
  - **`/predict` contract unchanged:** `handleAnalyze` still sends exactly `{ age, sex, education_years, mmse, ses }`; the Analyze button is additionally gated by `detailsValid` (always true once the details step passed, defensive only). No backend, ML, SHAP, MMSE scoring, or Q11 vision changes.
  - **Verified:** `npm run build` (`tsc && vite build`) passes. Live `POST /predict` on `127.0.0.1:8000` with `{age:70, sex:1, education_years:12, mmse:28, ses:2}` → HTTP 200 `{alzheimers_detected: true, predicted_class: Demented, detection_percentage: 79.5}`.
- **Q11 photo-based vision assessment UI (feat(mmse): add q11 photo assessment ui):** Replaced the on-screen drawing canvas for MMSE Question 11 with a photo flow. Frontend-only; the backend `POST /mmse/copying/evaluate` contract is consumed unchanged.
  - **New `frontend/src/components/mmse/Q11PhotoAssessment.tsx`:** reference figure shown in `sections.tsx` → examiner takes (`capture="environment"`) or uploads a photo (native file input, no camera library) → preview with `[Retake]`/`[Choose Another]`/`[Analyze Drawing]` → explicit "Analyze Drawing" calls `evaluateCopyingImage()` (JSON `{image: dataUrl}`, 120s axios timeout) → normalized result with confidence + `review_required`. States: `empty → photo → analyzing → assessed | error`. Loading shows "Analyzing drawing…" + spinner + "This may take up to a minute on the local vision model." (no fake progress, no technical logs, duplicate-click guarded). Assessed view shows "✓ Correct response"/"✕ Incorrect response" + "AI confidence: X%" + "⚠ Review required" (when `review_required`) with `[Accept AI result]`/`[Override]` (manual 0/1 becomes final). Error mapping: blank → "No drawing detected. Please submit a clear photo of the patient's drawing." (Retake/Choose Another); timeout → "Vision assessment timed out." (Retry/Retake); unavailable → "Vision assessment unavailable." (Retry); invalid → "Vision assessment returned an invalid result." (Retry); upload (bad/oversized/undecodable image) → friendly detail (Retake/Choose Another). Camera permission denied → "Camera access was not granted." with Upload Photo still available; unsupported camera → upload fallback notice. **No raw OS/network/stack traces shown.** Preview is an in-memory data URL (`previewData`) — never written to disk/localStorage/backend/logs.
  - **State (`frontend/src/mmse/state.ts`):** new `CopyingState { status: empty|photo|analyzing|assessed|error, previewData, previewName, aiScore, reviewRequired, reviewed, manual, errorKind, errorDetail }`; `MMSEState.copying` changed from `ScoreMark` to `CopyingState`. New helpers `copyingEffective()` (manual wins over AI) and `isCopyingFinalized()` (manual set, or AI finalized unless low-confidence-unreviewed); `computeScores().copying`, `isSectionComplete('copying')`, and `sectionResponseCounts().copying` all use them. MMSE total stays integer 0–30; Q1–10, `/mmse/evaluate` batch, and `/predict` untouched.
  - **API (`frontend/src/api.ts`):** `evaluateCopyingImage()` (120s timeout) + `CopyingEvaluateResponse` type + `classifyCopyingError()` (400-blank vs 400-upload vs 504 vs 502 vs 503 vs client-timeout).
  - **Deleted `frontend/src/components/mmse/DrawingCanvas.tsx`** (and its barrel export) — no longer used; no mouse/touch drawing anywhere in Q11.
  - **Verified:** `npm run build` passes. Live backend on `127.0.0.1:8000`: GOOD synthetic copy → HTTP 200 `{correct:false, score:0, confidence:0.6, reason:"Only one figure is present...", review_required:true}`; BLANK → HTTP 400 "No drawing detected..." in 0.0s (provider never invoked); oversized/undecodable → HTTP 400 friendly detail. Error-classification paths (blank/upload/timeout/invalid/unavailable) map to the exact UI strings.
- **Q11 blank-submission pre-check (fix(mmse): reject blank q11 submissions before vision):** Added a deterministic image-content sanity gate in `backend/src/vision_image.py` (`has_drawing_content`) that runs inside `prepare_patient_image` BEFORE any vision provider call. It measures the fraction of pixels that are meaningfully darker than the estimated paper level (90th percentile of the grayscale histogram). Conservative thresholds: `VISION_BLANK_INK_DELTA` (default 20 gray levels below paper) and `VISION_BLANK_MIN_INK_FRACTION` (default 0.5% of pixels). Blank/near-blank/ultra-low-contrast images → HTTP 400 "No drawing detected. Please submit a clear photo of the patient's drawing." — the provider is NEVER invoked. Faint-but-visible drawings deliberately pass (low threshold avoids over-rejection). Verified empirically: blank 0.0% ink, near-blank (single pixel) <0.5%, tiny 0.52%, good 2.09%, faint-235 ~1.5%, poor 1.02%. Real Ollama test: BLANK → 400 in 0.0s (no `[vision_eval]` log entry — Gemma never called), GOOD → 200 44.8s `{correct:true, score:1, confidence:0.95}`, POOR → 200 35.1s `{correct:false, score:0, confidence:0.6, review_required:true}`. New tests: `TestBlankPreCheck` (7 cases: blank rejected, near-blank rejected, blank never calls provider, simple/faint/good/poor drawings reach provider) — 29/29 backend tests pass. Frontend untouched; `/predict` + `/mmse/evaluate` re-verified HTTP 200; `npm run build` passes.
- **Q11 vision-assisted figure copying — backend/infrastructure milestone (feat(mmse): add q11 vision evaluation service):** New dedicated endpoint `POST /mmse/copying/evaluate` for MMSE Question 11 ONLY. Frontend UI untouched (no camera/upload/preview/reviewer — those belong to a separate later UI/UX prompt). Details:
  - **Provider abstraction (`backend/src/vision_eval.py`):** built around an OpenAI-compatible multimodal `chat/completions` contract. Exactly ONE provider runs per assessment (`VISION_PROVIDER=ollama|gemini|openai`); no voting, no auto-fallback. Shared layer owns the MMSE copying criterion prompt, normalized schema, strict validation, confidence/review, error normalization, and timeout semantics; provider adapters (`OllamaVisionProvider` via `/v1/chat/completions`, `GeminiVisionProvider`, `OpenAIVisionProvider`) only set base URL/model/auth/payload/response extraction. Config backend-only: `VISION_TIMEOUT` (default 120s — deliberately separate from the text-MMSE 175/180s budgets), `GEMINI_API_KEY`/`GEMINI_VISION_MODEL`, `OPENAI_API_KEY`/`OPENAI_VISION_MODEL`/`OPENAI_BASE_URL`, `VISION_MAX_UPLOAD_BYTES`, `VISION_MAX_IMAGE_DIMENSION`, `COPYING_REFERENCE_PATH`.
  - **Image handling (`backend/src/vision_image.py`):** in-memory only (Pillow, already pinned). Validates MIME (JPEG/PNG/WebP) + size (≤10 MB), decodes, EXIF-orients, preserves aspect ratio, downscales >2048 px, re-encodes to JPEG base64 data URLs. Patient images are NEVER persisted, logged, or returned. Upload as raw image bytes (`Content-Type: image/...`) or JSON `{"image": "data:...;base64,..."}` — no multipart (python-multipart not installed; no new deps added).
  - **Trusted reference:** loaded server-side from `frontend/public/mmse-copying-figure.png` (or `COPYING_REFERENCE_PATH`) — the same exact asset the frontend shows; never accepted from the client, never duplicated/regenerated.
  - **Structured result:** `{correct, score 0/1 agreeing with correct, confidence 0–1, reason non-empty, review_required}`. Malformed provider output → NO score + HTTP 502 "Vision assessment returned an invalid result." Timeout → 504 "Vision assessment timed out." Provider unreachable/not configured → 503 "Vision assessment unavailable." Bad uploads → 400 friendly messages. Low confidence (`< AI_CONFIDENCE_REVIEW_THRESHOLD` 0.7) → `review_required: true`. Criterion: geometric structure, presence of both figures, required overlap/intersection ONLY — no aesthetics/artistic/line-thickness/cleanliness scoring, no invented clinical criteria.
  - **Tests (`backend/tests/test_vision_eval.py`, stdlib unittest, synthetic images only):** 22 checks — valid JPEG/PNG accepted, empty/unsupported-MIME/oversized/undecodable rejected, reference loaded from trusted asset, OpenAI-compatible multimodal payload (reference + patient image_url parts), valid/malformed/missing/mismatch/out-of-range validation, low vs high confidence flagging, end-to-end score 0/1 mapping, timeout and unavailable kinds, provider selection default, `/predict` + `/mmse/evaluate` + new route present. **All pass.**
  - **Real Ollama + Gemma 3 4B test (synthetic images only):** good copy (two overlapping figures) → `{correct: true, score: 1, confidence: 0.95, review_required: false}` in 53.7s; poor copy (single ellipse) → `{correct: false, score: 0, confidence: 0.6, review_required: true}` in 37.5s. A blank white 640×480 canvas was (incorrectly) judged correct — a real model limitation, reported honestly, NOT a claim of clinical accuracy. Gemini/OpenAI: **not live-tested because credentials are not configured**; adapters verified by unit tests (request construction, auth header, payload, error handling).
- **Single-call Ollama batch evaluation (perf milestone):** replaced up to 30 sequential per-item Ollama `/api/chat` calls with exactly **ONE** `/api/chat` call that returns structured JSON for every AI-evaluable MMSE item at once (`backend/src/ai_eval.py`). The two-phase batch architecture and the frontend contract are unchanged — the browser still sends exactly ONE `POST /mmse/evaluate`. Details:
  - `BATCH_SYSTEM_PROMPT` + `build_batch_messages()` build a single system+user pair where each item keeps its section-specific rules (`_prompt_for`), so the one call preserves per-item semantics.
  - `_evaluate_ollama_batch()` sends one request with `OLLAMA_BATCH_TIMEOUT` (default 175s, fits under the 180s browser timeout) and validates every requested key via `parse_batch_result()`. Missing/malformed entries become per-item `errors` — never silently scored. Invalid keys / unsupported sections / empty responses are rejected server-side before the call. A flat `{key: result}` response (without the `items` wrapper) is accepted as a fallback.
  - `evaluate_mmse_batch()` is provider-aware: Ollama → single call; Gemini → unchanged per-item `ThreadPoolExecutor` path. `_max_concurrency()` now only serves the Gemini path.
  - Dev-side timing log (`[ai_eval] AI_PROVIDER=... items=... provider_calls=... elapsed=...s results=... errors=...`) goes to the backend console only — never shown in the UI.
  - **Measured bottleneck (honest, not faked):** the 30-item realistic batch generates ~900–1500 output tokens at ~7.4 tok/s (Gemma 3 4B split 54% CPU / 46% GPU on a GTX 1650 4 GB, so the 3.5 GB model can't sit fully on the GPU). Measured single-call timings: 3-item ~9.5–14.7s, 10-item ~60–70s, full 30-item ~139–158s when it completes, but generation variance occasionally pushes a run past the 175s backend / 180s browser window (observed 170s and 175s timeouts → 503). The user's ~30–45s target is **not achievable on this hardware with this model**; the single-call design removes the per-item call overhead but the token-generation rate is the floor. This is reported rather than hidden, per the task instructions.
  - Accuracy regression (12-item batch, 75.2s, HTTP 200, 0 errors) passed: correctly judged "2026"/"twenty twenty-six", Maharashtra, apple, 93, 65 ("sixty-five"), wristwatch/watch, correct repetition as correct; 1999, Delhi vs Mumbai, banana, 92, "I don't know" as incorrect. `npm run build` passes; `/predict` still HTTP 200.
- **Exact copying figure wired into Section 11:** supplied `frontend/public/mmse-copying-figure.png` referenced via `COPYING_REFERENCE_IMAGE = '/mmse-copying-figure.png'` in `frontend/src/mmse/config.ts`. The Copying section (already rendering "Please copy this picture." then the `<img>` when the constant is set) now shows the exact figure as the ONLY stimulus; drawing canvas and examiner scoring for Q11 kept as-is. No vision (Ollama/Gemma/Gemini/OpenAI), no camera upload, no automated figure scoring. Backend, `/predict`, ML, SHAP, MMSE scoring, and AI batch architecture untouched.
- **Provider-aware batch concurrency + 180s timeout (`2abab40`):** Real runtime test of Ollama + Gemma 3 4B showed a single `/api/chat` call completes in ~3–15 s, but the full MMSE batch timed out at the frontend's 90 s because the backend evaluated up to 8 items concurrently on a single local GPU (GTX 1650 4 GB). Fix (optimization ONLY — no MMSE structure/scoring/`/predict` changes):
  - **Backend `ai_eval.py`:** provider-aware concurrency via `_max_concurrency()` — `OLLAMA_MAX_CONCURRENCY` (default `1`, sequential) for local Ollama to avoid GPU/CPU contention; `GEMINI_MAX_CONCURRENCY` (default `8`, unchanged cloud behavior). New env vars documented in the module docstring. The two-phase batch architecture is unchanged — the browser still sends exactly ONE `POST /mmse/evaluate`.
  - **Frontend `api.ts`:** default batch timeout `90000 → 180000` ms. Per-item `AI_TIMEOUT` (30 s) is untouched; timeout protection kept — the UI still shows "AI assessment timed out." instead of hanging.
- **Runtime verification (real Gemma 3 4B via Ollama):** single `/api/chat` 14.86 s (incl. cold model load); 3-item `/mmse/evaluate` 16.23 s (HTTP 200, structured results); 30-item realistic full batch **171.99 s (HTTP 200, all 30 items, 0 errors)** — completes within the new 180 s window. `/predict` still 200. Structured output `{correct, score, confidence, reason}` confirmed.
- **MMSE UX refinements (post-batch):** Fixed the issues found in manual testing without redesigning or touching the batch architecture.
  - **Location-aware Orientation to Place:** replaced empty per-item `expected` config with an examiner-configured "Assessment location" form (state / county / town / building / floor) at the top of the section. Those values become the reference answers in the batch and are never shown to the patient. Removed all `src/mmse/config.ts` / "Expected answer not configured" developer text.
  - **Friendly error states:** raw OS/network/backend errors never appear in the normal UI. Provider failure → "AI assessment unavailable" + "The selected AI provider is currently unavailable." + [Retry]; timeout → "AI assessment timed out." + [Retry]; partial failure → "Some responses could not be assessed." + [Review items]. Raw detail lives behind a collapsed "View technical details" disclosure (both on the Summary and per item).
  - **Observation sections:** Three-Step Command / Reading / Copying are labelled "Observation-based" with an "AI vision assistance will be added in a later milestone" note — not presented as examiner-as-permanent-scorer.
  - **Response validation display:** results show "✓ Correct response / ✕ Incorrect response / ⚠ Review required / Response required"; the backend system prompt now explicitly evaluates *meaning*, not formatting (case/punctuation/spacing/number-as-words/synonyms are not penalized).
  - No vision work (Q8/Q9/Q11), no figure invented, no canvas redesign; copying placeholder kept but cleaned of developer text.
- **AI-assisted MMSE batch scoring (two-phase)** (prior milestone, committed `8cadbc5`): Replaced per-question auto-assessment with a two-phase batch workflow. Phase 1 records responses only (typing/speech/Enter never trigger AI); Phase 2 sends ONE batch `POST /mmse/evaluate` with all applicable answered responses, applies per-item validated results ("✓ Correct response / ✕ Incorrect response / AI confidence / ⚠ Review required"), updates the 0–30 total, then "Continue to Analysis" → existing `/predict`. Backend: `evaluate_mmse_batch` (503 AI disabled/all-provider-fail; 422 empty batch; per-item `errors` — never a silent score). `/predict` contract, model, and SHAP untouched.
- **Deterministic Orientation to Time scoring (fix(mmse): correct deterministic orientation scoring):** The five `orientation_time.*` items (year / season / date / day / month) are now scored 100% deterministically on the backend and NEVER reach the AI provider.
  - **Root cause of the reported regression:** no hybrid/deterministic evaluator existed — `orientation_time` was in `AI_SECTIONS` and all five items were sent to Ollama. The model's batch JSON omitted four keys, so `parse_batch_result` turned them into per-item errors (`Model did not return a result for item 'orientation_time.season'`). With no `aiScore`, the items were not finalized and the Summary stuck at "Resolve the flagged items to continue".
  - **New `backend/src/mmse_rules.py`:** `evaluate_orientation_time(item_key, response)` uses the server's current date/time (authoritative) and returns `{correct, score, confidence: 1.0, reason}`. Accepts semantic equivalents: year digits or English words ("2026", "twenty twenty-six", "two thousand twenty-six"), season case-insensitive ("fall"/"autumn"), date digits/ordinal/words ("13", "13th", "thirteen"), weekday and month case-insensitive incl. abbreviations and "the month of August". Unparseable responses score `correct: false` (never a per-item error). `confidence` is `1.0` (a real number) so the frontend's `confidence < 0.7` review flag never trips and the Summary always loads.
  - **`backend/src/ai_eval.py`:** added `DETERMINISTIC_SECTIONS = {"orientation_time"}` and removed `orientation_time` from `AI_SECTIONS`. `evaluate_mmse_batch` splits the batch: deterministic items are scored first without any provider; only AI-required items go to the provider (Ollama: one smaller batch call; Gemini: per-item). `AI_PROVIDER=none` still 503s for AI items but deterministic-only requests succeed. Deleted the dead `_orientation_time_expected` helper and its `_prompt_for` branch.
  - **Frontend:** no changes needed — `buildBatchItems` still sends the five items with `expected: ""`, and the backend routes them deterministically; the existing `applyBatchResultsToDraft` finalizes them (confidence 1.0 ⇒ not flagged). One assessment action, zero extra model calls, no concurrency/timeout changes.
  - **Verified:** new `backend/tests/test_mmse_rules.py` (17 checks: 5/5 correct, 0/5 incorrect, semantic equivalents, unparseable → incorrect not error, zero provider calls for orientation items, orientation keys absent from the model prompt, missing AI result → per-item error, deterministic-only succeeds with `AI_PROVIDER=none`, AI-only still 503 with `AI_PROVIDER=none`, empty batch 422, section sets disjoint). Full backend suite 46/46. `npm run build` passes. Live: `/mmse/evaluate` with all-correct orientation answers → 5/5 `confidence: 1.0` no errors; mixed batch (orientation + naming) → 7/7, 0 errors; full 25-AI-item batch → HTTP 200, 0 errors. `/predict` unchanged (79.5% Demented). No Q11 vision, `/predict`, ML, SHAP, UI, or demographic-flow changes.
- **Deterministic Serial-7s scoring (fix(mmse): score serial seven deterministically):** The five `attention_serial7.*` items are now scored deterministically on the backend and NEVER reach the AI provider.
  - **Root cause:** the live full-MMSE batch showed Gemma classifying all five Serial-7s answers (93/86/79/72/65) as incorrect — an LLM is the wrong tool for deterministic arithmetic, and the error degraded the Attention & Calculation score.
  - **`backend/src/mmse_rules.py`:** new `SERIAL_7_EXPECTED = (93, 86, 79, 72, 65)` and `evaluate_attention_serial7(item_key, response)`. Parses the response as a number (digits, surrounding whitespace, or English number words via the shared `_parse_number`, e.g. "ninety-three") and compares exactly. Returns `{correct, score, confidence: 1.0, reason}`; unparseable → `correct: false` (never a per-item error, never sent to AI). Keys match the frontend's 1-indexed contract (`attention_serial7.1`..`.5` map to SERIAL_7_EXPECTED positions; `"0"` accepted defensively as the first item).
  - **`backend/src/ai_eval.py`:** `attention_serial7` added to `DETERMINISTIC_SECTIONS`, removed from `AI_SECTIONS`; new `DETERMINISTIC_EVALUATORS` dispatcher so `evaluate_mmse_batch` routes both deterministic sections without any provider involvement. Deleted the dead serial-7 `_prompt_for` branch. Frontend and workflow unchanged — the same batch is sent; serial-7 keys are simply split out server-side.
  - **Verified:** new tests in `backend/tests/test_mmse_rules.py` (Serial7RulesTest + routing tests: 5/5 correct, 0/5 wrong, mixed 93/86/80/72/65 → 4/5 (1 1 0 1 1), number-words → 5/5, unparseable → incorrect not error, zero provider calls for serial-7, serial-7 keys absent from the model prompt, deterministic-only succeeds with `AI_PROVIDER=none`). Full backend suite 57/57; `npm run build` passes. Live: serial-7 correct set → 5/5 `confidence: 1.0` in ~0s (log: `deterministic=5, ai=0, provider_calls=0`); mixed → 4/5; number-words → 5/5; orientation regression still 5/5. **Full realistic 30-item MMSE benchmark: 10 deterministic + 20 AI, ONE provider call, elapsed 145.86s, 30/30 correct, 0 errors** (previously 5 deterministic + 25 AI at ~172s baseline with serial-7 misjudged). `/predict` unchanged (79.5% Demented). No Q11 vision, ML, SHAP, UI, or demographic-flow changes.
- **Expanded deterministic scoring coverage (perf(mmse): expand deterministic scoring coverage):** The hybrid MMSE architecture now scores 29 of the 30 items deterministically on a clean response (writing is the only AI item), so a full 30-item batch needs a single tiny Ollama call instead of 20.

  - **Fully deterministic, never AI:** `orientation_time` (5), `attention_serial7` (5), `attention_spell_world` (5 — `SPELL_WORLD_EXPECTED = ("D","L","R","O","W")`, case-insensitive, single-letter responses from the frontend's per-letter split; a longer string is compared at that item's position, e.g. "DLROW" → item 4 = "O").
  - **Hybrid (deterministic when safely provable, AMBIGUOUS → AI otherwise):** `orientation_place` (5), `registration` (3), `delayed_recall` (3), `naming` (2), `repetition` (1). `mmse_rules.py` helpers: `_normalize_text` (lowercase, strip punctuation/apostrophes, collapse whitespace), `_strip_articles`, `_is_uncertain` (`_UNCERTAIN_PHRASES`: "I don't know", "not sure"...), `_is_generic_or_referential` (`_GENERIC_WORDS`: fruit, thing, object, one, it...), `_EQUIVALENTS = {"wristwatch": {"watch", "wrist watch"}}` (bounded equivalence only), `_object_verdict` (MATCH: normalized equality / article-stripped / whole-word keyword containment / bounded equivalence; NO_MATCH: no word overlap → incorrect; AMBIGUOUS: uncertain, generic/referential, partial word overlap, or unconfigured place expected), `_repetition_verdict` (exact normalized match → correct; uncertain or any word overlap → AMBIGUOUS; no overlap → incorrect). Return sentinels `MATCH` / `NO_MATCH` / `AMBIGUOUS`.
  - **`backend/src/ai_eval.py`:** `DETERMINISTIC_EVALUATORS` maps all 8 sections; `FULLY_DETERMINISTIC_SECTIONS = {"orientation_time", "attention_serial7", "attention_spell_world"}` (disjoint from `AI_SECTIONS`); `AI_SECTIONS` = hybrid sections + `writing` so AMBIGUOUS items still route to AI. `evaluate_mmse_batch` calls `evaluator(item_key, entry.response, entry.expected)` per item: dict → deterministic result, `AMBIGUOUS` → that single item joins the AI batch, `None` → per-item error. Log line: `(deterministic=X, ambiguous=Y, ai=Z)`.
  - **Frontend:** no changes — `buildBatchItems` already sends the same keys with `expected`; the backend routes them.
  - **Verified:** `backend/tests/test_mmse_rules.py` extended (SpellWorldRulesTest, HybridRulesTest, HybridBatchRoutingTest + reworked routing tests: 29 deterministic + 1 AI on a clean batch, zero provider calls for every clear hybrid answer, ambiguous items still reach AI, `AI_PROVIDER=none` still 503s for AI items incl. ambiguous routing). Full backend suite 85/85; `npm run build` passes. **Live benchmark: clean 30-item batch → deterministic=29, ambiguous=0, ai=1, ONE provider call, 21.66s, 30/30 correct, 0 errors** (previous baseline 10 deterministic + 20 AI at 145.86s); wrong-answer batch → 7 incorrect deterministically, 0 errors; ambiguous batch (naming "the thing used to tell time", registration "that fruit") → deterministic=27, ambiguous=2, ai=3, 30/30. `/predict` unchanged (HTTP 200, `rule_applied=false`, probabilities returned). No Q11 vision, ML, SHAP, UI, demographic-flow, or timeout changes.
- Earlier milestones: `590ff38` examiner-scored MMSE questionnaire; `ccc45d3` contextual MMSE right panel; `73ef09c` patient-response recording separated from scoring; `1235955` per-item AI-assisted scoring (superseded by the batch workflow).

### Current State
- Working tree: the session-persistence milestone is **verified end-to-end** (41/41 CDP checks) and being committed as `fix(ux): persist and restore active assessment session`; the prior UX-fix milestone is committed and pushed (`fe4d253`). `npm run build` passes; backend tests 85/85 pass (no backend changes); `/predict` contract unchanged. Ollama 0.32.9 + `gemma3:4b` (aliased `gemma3:latest`); vision Q11 round-trip live-verified (analyze ~9s warm; batch ~8–10s warm).
- **Session persistence:** single versioned key `alzheimers_assessment_session_v1` (`frontend/src/mmse/session.ts`) restores the exact phase/step/state on refresh; one authoritative restore in `App.tsx`; 400ms-debounced saves (skipped when empty); Q11 images never stored (restored as "Choose/Take Photo", AI score kept → total stays accurate); corrupt sessions cleared safely; legacy `alzheimers_assessment_details_v1` migrated once; "Restart Assessment" = `clearSession()` → Assessment Mode; MMSE "Re-take" preserves mode+details.
- **Patient-mode Next (root cause + decision):** the disabled Next with 5 filled place responses is caused by `sectionResponseCounts().orientationPlace` requiring BOTH a configured location field AND a response — pure Patient mode never sets the location, so `done` stays 0/5. **Decision (option A):** Patient mode intentionally stays examiner-dependent for Orientation to Place (location = reference answers) and Copying (photo analysis). The block is kept, no scoring fabricated; the UI now EXPLAINS it (in-section "Examiner required" notices + nav message "An examiner is required for this section"). Examiner + Patient remains the full flow (verified to the analysis form).
- **Persistence fix (prior milestone):** Re-take cleared the localStorage key but the save-on-change effect re-wrote an empty record; fixed in `frontend/src/App.tsx` with a `detailsValid` guard on the save effect (shipped in `e4cfe16`).

### Next Recommended Task
- Commit `fix(ux): persist and restore active assessment session` (`frontend/src/App.tsx`, `frontend/src/components/AssessmentDetails.tsx`, `frontend/src/components/mmse/MMSEAssessment.tsx`, `frontend/src/mmse/session.ts`, `AI_CONTEXT.md`), push, verify clean tree, and deliver the final report.
- Then consider whether the local Ollama latency is acceptable for production. The full 30-item batch sits close to the 180s browser timeout on this hardware. Options to revisit deliberately (never by removing items): a larger GPU, a smaller/quantized model, streaming progress, or a split-batch strategy. Any of these needs explicit user instruction.
- Then consider live per-patient SHAP as the next feature.

### Files Recently Changed
- `frontend/src/mmse/session.ts` (NEW: versioned full-session persistence — `SESSION_KEY`/`SESSION_VERSION`, `AssessmentSession`/`MmseSessionPart` types, structural validators, Q11 storage/restore sanitizers, legacy `alzheimers_assessment_details_v1` migration, `loadSession` (clears corrupt/incompatible keys), `saveSession` (strips image data), `clearSession`)
- `frontend/src/App.tsx` (single authoritative startup restore; 400ms-debounced save effect with `isEmptySession` guard; `handleMmseComplete`/`handleMmseStateChange`/`handleRetake`/`handleRestart`; MMSEAssessment wired with `initialStep`/`initialPhase`/`initialState`/`onSessionStateChange`; "Restart Assessment" button on the Analysis form)
- `frontend/src/components/mmse/MMSEAssessment.tsx` (optional restore props `initialStep`/`initialPhase`/`initialState` + `onSessionStateChange`; `useState` initialized from props; 400ms-debounced state callback)
- `frontend/src/components/AssessmentDetails.tsx` (`onRestart` prop + "Restart Assessment" button next to "← Back")
- `frontend/src/components/mmse/primitives.tsx` (NEW `MicIcon` + `MicButton` — icon + "Mic"/"Stop" in the same control; NEW `SectionNavigationContext` + `useSectionNavigation` + `PatientResponseInput` Enter handling; single-line input right padding `pr-16 → pr-20`)
- `frontend/src/components/mmse/MMSEAssessment.tsx` (shared `goToNext` used by Next button AND Enter; `SectionNavigationContext.Provider` wrapping section + nav; `patientBlocked` + explicit "An examiner is required for this section" nav message)
- `frontend/src/components/mmse/sections.tsx` (patient-mode amber "Examiner required" notices on Orientation to Place + Copying; SpellWorldBlock + AIScoredResponse use `MicButton`)
- `frontend/src/mmse/mode.ts` (NEW: AssessmentMode types, ASSESSMENT_MODES, AssessmentModeContext, useAssessmentMode)
- `frontend/src/mmse/details.ts` (NEW: DetailsDraft string-based, DETAILS_FIELDS, sanitizeNumericInput, detailsValid, detailsToPatientInput, localStorage persistence)
- `frontend/src/mmse/namingLibrary.ts` (NEW: 12-object Q6 library, DEFAULT_NAMING_SELECTION, getNamingObject, pickNamingObjects)
- `frontend/src/mmse/imageUtils.ts` (NEW: Q11 processPhoto validation/compression, PhotoInfo, PHOTO_* constants)
- `frontend/public/objects/*.svg` (NEW: 12 local object assets)
- `frontend/src/components/AssessmentModePicker.tsx` (NEW: mode step UI)
- `frontend/src/App.tsx` (phase 'mode'|'details'|'mmse'|'form'; mode/details state; localStorage restore/save/clear; AssessmentModeContext.Provider; /predict via detailsToPatientInput)
- `frontend/src/components/AssessmentDetails.tsx` (DETAILS_FIELDS-driven, Back, immediate validation)
- `frontend/src/components/InputField.tsx` (numeric → type text + inputMode numeric)
- `frontend/src/components/index.ts` (AssessmentModePicker export)
- `frontend/src/components/mmse/primitives.tsx` (mode gating: ExaminerInstructions, AIResultPanel, AIScoredResponse, SectionShell)
- `frontend/src/components/mmse/sections.tsx` (NamingSection object library UI; OrientationPlace + Copying mode gating)
- `frontend/src/components/mmse/MMSESummary.tsx` (mode gating: breakdown/technical/review text)
- `frontend/src/components/mmse/Q11PhotoAssessment.tsx` (uses processPhoto, photoInfo display)
- `frontend/src/mmse/state.ts` (NamingState watchObject/pencilObject; CopyingState.photoInfo)
- `frontend/src/mmse/batch.ts` (naming expected via getNamingObject)
- `frontend/src/mmse/config.ts` (removed unused NAMING_ITEMS)
- `frontend/src/api.ts` (VISION_CLIENT_TIMEOUT = 120000 named constant)
- `backend/src/vision_eval.py` (pre-existing uncommitted: VISION_TIMEOUT 120s)
- `frontend/vite.config.ts` (pre-existing uncommitted: dev host/allowedHosts)
- `backend/src/ai_eval.py` (`DETERMINISTIC_EVALUATORS` for 8 sections, `FULLY_DETERMINISTIC_SECTIONS`, `AI_SECTIONS` = hybrid + writing, batch split routes dict/AMBIGUOUS/None, `(deterministic=X, ambiguous=Y, ai=Z)` log line, dead spell-world prompt branch deleted)
- `backend/tests/test_mmse_rules.py` (SpellWorldRulesTest, HybridRulesTest, HybridBatchRoutingTest, reworked routing/AI-disabled/missing-AI tests, disjointness test on `FULLY_DETERMINISTIC_SECTIONS`)
- `backend/src/mmse_rules.py` (deterministic Serial-7s: `SERIAL_7_EXPECTED`, `evaluate_attention_serial7`, number parsing via `_parse_number` incl. whitespace normalization)
- `backend/src/mmse_rules.py` (NEW: deterministic Orientation-to-Time evaluator — server date/time authoritative, English number-word parser, semantic-equivalence normalizers, `confidence` 1.0)
- `backend/src/ai_eval.py` (`DETERMINISTIC_SECTIONS`, `orientation_time` removed from `AI_SECTIONS`, deterministic/AI batch split in `evaluate_mmse_batch`, `_orientation_time_expected` + its prompt branch deleted)
- `backend/tests/test_mmse_rules.py` (NEW: 17 checks — deterministic routing, zero provider calls, prompt exclusion, per-item error contract, `/predict` contract import)
- `frontend/src/components/AssessmentDetails.tsx` (NEW: pre-MMSE Assessment Details step — Age/Sex/Education/SES, Continue gated on validity)
- `frontend/src/App.tsx` (phase `'details' | 'mmse' | 'form'` defaulting to 'details'; `handleRestart` clears demographics and returns to details; read-only assessment summary on the Analysis screen; contextual right-panel EmptyState for the details phase; `/predict` payload unchanged)
- `frontend/src/components/index.ts` (added `AssessmentDetails` barrel export)
- `frontend/src/components/mmse/Q11PhotoAssessment.tsx` (NEW: Q11 photo flow — capture/upload/preview/analyze/result/review/errors)
- `frontend/src/components/mmse/DrawingCanvas.tsx` (DELETED — replaced by the photo flow)
- `frontend/src/components/mmse/index.ts` (removed `DrawingCanvas` barrel export)
- `frontend/src/components/mmse/sections.tsx` (`CopyingSection` now renders `Q11PhotoAssessment`; instructions updated)
- `frontend/src/mmse/state.ts` (`CopyingState`, `CopyingStatus`, `CopyingErrorKind`; `copyingEffective`, `isCopyingFinalized`; `MMSEState.copying` → `CopyingState`)
- `frontend/src/mmse/batch.ts` (`sectionResponseCounts().copying` uses `isCopyingFinalized`)
- `frontend/src/api.ts` (`evaluateCopyingImage`, `CopyingEvaluateResponse`, `classifyCopyingError`)
- `backend/src/vision_image.py` (blank/near-blank pre-check: `has_drawing_content`, `VISION_BLANK_INK_DELTA`, `VISION_BLANK_MIN_INK_FRACTION`, hook into `prepare_patient_image`)
- `backend/tests/test_vision_eval.py` (`TestBlankPreCheck`: 7 pre-check cases + synthetic blank/near-blank/simple/faint/poor helpers)
- `backend/src/vision_eval.py` (NEW: Q11 vision provider abstraction + evaluation service)
- `backend/src/vision_image.py` (NEW: Q11 image validation/normalization/encoding + trusted reference loader)
- `backend/src/api.py` (`POST /mmse/copying/evaluate` endpoint; raw-binary or JSON-base64 body)
- `backend/tests/test_vision_eval.py` (NEW: 22 stdlib-unittest checks, synthetic images only)
- `backend/src/ai_eval.py` (single-call Ollama batch: `BATCH_SYSTEM_PROMPT`, `build_batch_messages`, `_parse_json_object`/`_validate_item`/`parse_batch_result` validation refactor, `_evaluate_ollama_batch`, provider-aware `evaluate_mmse_batch`, `OLLAMA_BATCH_TIMEOUT` env, dev-side `[ai_eval]` timing log)
- `frontend/src/mmse/state.ts` (+`AssessmentLocation`; `MMSEState.location`)
- `frontend/src/mmse/config.ts` (`PLACE_ITEMS` without hardcoded expected; `LOCATION_FIELDS` for the examiner location form)
- `frontend/src/mmse/batch.ts` (`buildBatchItems` uses `state.location` for orientation_place; location-aware `sectionResponseCounts`; `collectAiItems`, `countAssessmentErrors`)
- `frontend/src/components/mmse/primitives.tsx` (`SectionShell` AI/Observation badge; `AIResultPanel` "✓ Correct response / ✕ Incorrect response / Response required" + per-item friendly error with "View technical details"; `AIScoredResponse` `disabledNotice` replaces `aiEnabled`)
- `frontend/src/components/mmse/sections.tsx` (OrientationPlace location form; observation wording for command/reading/copying; cleaned copying placeholder)
- `frontend/src/components/mmse/MMSESummary.tsx` (friendly error state + "View technical details"; "Some responses could not be assessed." + [Review items])
- `frontend/src/components/mmse/MMSEAssessment.tsx` (`BatchErrorInfo {title, subtitle, detail}`; `countAssessmentErrors`)
- `frontend/src/components/mmse/MMSEIntroduction.tsx` (two-phase + observation note)
- `backend/src/ai_eval.py` (system prompt: semantic evaluation — no formatting-only penalties)
- Earlier: batch workflow files (`backend/src/api.py`, `frontend/src/api.ts`, `frontend/src/mmse/batch.ts`, etc.) — committed in `8cadbc5`.

### Tests Verified
- **UX-fix browser verification (COMPLETE — 32/32 PASS):** headless Chrome via CDP (`C:\Users\sumed\AppData\Local\Temp\opencode\cdp-uxfix.mjs`, Node built-in WebSocket, zero new deps; a fake `SpeechRecognition` is installed so the Mic control can be exercised in headless Chrome).
  - **Mic control:** icon (SVG) AND "Mic" text inside the SAME button (not icon-only); accessible label; clickable; toggles `[🎙 Mic] → [🎙 Stop] → [🎙 Mic]`; verified at 375px mobile width (icon + text both visible, not clipped).
  - **Enter navigation (Examiner):** 4/5 responses + Enter → does NOT advance; 5/5 + Enter → section 2 (Place); Place complete + Enter → section 3 (Registration); Registration 3 + Enter → section 4 (Attention); Attention complete + Enter → section 5 (Delayed Recall); Writing textarea + Enter → no advance, value preserved (multiline not broken); full traversal to collect-phase summary; last-section action caps at Summary.
  - **Patient mode:** Enter advances on non-examiner section (Time → Place); location config hidden; in-section "Examiner required" notice shown; Next stays blocked at Place with 5 responses; nav message explains the block; Enter does NOT bypass it.
  - **Examiner + Patient regression:** location config visible; naming dual UI (2 selects + expected answers); Q11 analyze round-trip; batch assessment completed (~8–10s warm); full flow reaches analysis form with `/ 30`.
  - **Fallback:** unsupported-speech message preserved (browser natively supports speech in headless, so the fallback path is verified by code, not a live run).
- **UX milestone browser verification (COMPLETE — 68/68 PASS):** headless Chrome via CDP (`C:\Users\sumed\AppData\Local\Temp\opencode\cdp-verify.mjs`). All three modes exercised:
  - **Mode picker:** shows Patient / Examiner / Examiner+Patient.
  - **Patient:** numeric sanitization (`021` → `21`, empty allowed, Continue disabled until valid); localStorage writes ONLY approved keys (`{mode:'patient', age:'65', sex:1, education_years:'21', ses:'3'}`); reload restores; examiner UI hidden (no instructions, no badge, no location config); Orientation-to-Time Next enabled after 5 responses; place section shows only the note and Next is blocked (examiner-only location — **by design**, user-confirmed).
  - **Examiner:** location + examiner instructions visible; Q6 shows 2 object images, ≥2 selects, ≥2 `Expected answer:` entries, "What is this?" prompt; Q11 pipeline: low-res → EXACT message + "Choose Another"; invalid → "Image could not be decoded."; valid 1000×800 → no `(optimized)` suffix; oversized 3000×2000 → 2048×1365 `(optimized)`; blank → backend 400 "No drawing detected."; valid photo → Ollama vision round-trip (elapsed recorded); batch assess completed; form phase reached; **Re-take → localStorage `stored = null`, back at mode picker**.
  - **Both:** naming shows examiner controls AND patient prompt; full flow through batch + analysis form.
  - **All 12 `/objects/*.svg` + `mmse-copying-figure.png` return HTTP 200.**
- **Bug found & fixed by the browser run:** after Re-take the save-on-change effect re-wrote an empty record. Fix in `frontend/src/App.tsx`: save effect guarded with `if (detailsValid(details)) saveStoredDetails(storedDetails);`. Post-fix run confirmed `stored = null` and clean return to the mode picker. `npm run build` passes after the fix.
- **UX milestone (build + backend):** `npm run build` (`tsc && vite build`) — passes, no TypeScript errors (501 modules, ~5.25s; only pre-existing chunk-size warning). Backend suite `python -m unittest tests.test_mmse_rules tests.test_vision_eval` — 85/85 pass (backend unchanged except pre-existing `vision_eval.py` timeout). `/predict` contract unchanged.
- `npm run build` (`tsc && vite build`) — passes, no TypeScript errors (verified after the Assessment Details milestone).
- Live `POST /predict` with `{age: 70, sex: 1, education_years: 12, mmse: 28, ses: 2}` → HTTP 200, contract unchanged.
- **Q11 blank pre-check (synthetic, stdlib unittest):** `python -m unittest backend.tests.test_vision_eval` — 29/29 pass. New: completely blank white → rejected (400 message); near-blank (single pixel) → rejected; blank NEVER calls the provider (verified by mocking `get_provider` to fail if reached); simple visible drawing / faint 235-gray drawing / good copy / poor copy all reach the provider and score 0/1. Existing image-validation and result-validation suites still pass.
- **Real Ollama + Gemma 3 4B Q11 vision test (post pre-check):** BLANK 640×480 white → **HTTP 400 in 0.0s** ("No drawing detected...") with NO `[vision_eval]` log entry (Gemma never invoked); GOOD synthetic copy → HTTP 200 in 44.8s `{correct: true, score: 1, confidence: 0.95, review_required: false}`; POOR copy (single figure) → HTTP 200 in 35.1s `{correct: false, score: 0, confidence: 0.6, review_required: true}`. In the browser run the first vision Analyze took ~50s, subsequent ~9s (model warm). Gemini/OpenAI: not live-tested (no credentials).
- **Single-call Ollama runtime (real Gemma 3 4B, `gemma3:latest`):** 3-item `/mmse/evaluate` ~9.5–14.7s; 10-item ~60–70s; **30-item full batch ~139–158s HTTP 200 with all 30 structured results when it completes** — but generation variance occasionally exceeds the 175s backend / 180s browser window (observed 170s & 175s timeouts → 503). 12-item accuracy regression HTTP 200, 0 errors, all judgments sensible. Direct single-call diagnostics: 916–1533 output tokens, ~7.4 tok/s, prompt_eval ~600–2300 tokens. `/predict` 200 (unchanged). Structured output schema `{correct, score, confidence, reason}` verified; flat (no `items` wrapper) response shape also accepted.
- Frontend batch logic verified by Node harness (compiled `state.ts` + `batch.ts`): location-unconfigured → no `orientation_place` items sent; location-configured → 5 place items sent with the examiner's values as `expected`; place response counts require location + response; `countAssessmentErrors` counts only `error` items; `applyBatchResultsToDraft` still applies place scores; totals remain integer 0–30.
- Backend (from batch milestone): 11 validation checks + live batch via mock Gemini provider → 200; provider-unreachable → 503; `/predict` 200 (unchanged contract).

### Commit
- `fe4d253` — `fix(ux): improve mic controls and keyboard navigation` (primitives.tsx + MMSEAssessment.tsx + sections.tsx + AI_CONTEXT.md)
- `e4cfe16` — `fix(ux): complete assessment mode and input safeguards` (App.tsx persistence guard + api.ts VISION_CLIENT_TIMEOUT 120000 + AI_CONTEXT.md)
- `907483a` — `feat(ux): add assessment modes persistence and input safeguards` (UX milestone, incl. `backend/src/vision_eval.py` Q11 `VISION_TIMEOUT` 120s + `frontend/vite.config.ts` dev host/allowedHosts — deliberate inclusion)
- `d717ad3` — `perf(mmse): expand deterministic scoring coverage`
- `2b0dc75` — `docs: record deterministic coverage commit hash`
- `e6fd581` — `fix(ui): start assessment details empty`

### Push Status
- Pushed to `origin/main` successfully through `fe4d253` (mic controls + keyboard navigation).
- Working tree clean.

### Important Warnings
- Assessment Details values live in App-level state AND in localStorage (`alzheimers_assessment_details_v1`) — persist ONLY the approved keys (mode, age, sex, education_years, ses), and ONLY when the draft is valid. Restore is a `detailsValid`-guarded read on app start; "Restart Assessment"/Re-take clears the key and returns to the Mode step. NEVER store MMSE answers, Q11 images/blobs, payloads, or API keys in localStorage. Numeric details are string-sanitized (digits only, no leading zeros); convert to numbers ONLY in `detailsToPatientInput` for `/predict`.
- Assessment Details uses the SAME field semantics/ranges as the original form (Age 50–100, Education 0–25, SES 1–5, Sex 1=Male/0=Female). The `/predict` payload must stay exactly `{age, sex, education_years, mmse, ses}` — do not add fields or change order.
- No live SHAP endpoint exists — do not assume per-patient SHAP.
- Q11 blank/near-blank submissions are rejected deterministically BEFORE any vision provider call (HTTP 400, `has_drawing_content` in `vision_image.py`). This fixes the earlier blank-canvas false-positive. The gate is conservative (0.5% ink / 20 gray-level delta) so faint-but-visible drawings pass — do NOT tighten it without explicit instruction or you will over-reject legitimate faint pencil drawings.
- Q11 vision backend exists (`POST /mmse/copying/evaluate`) AND the frontend consumes it (`evaluateCopyingImage` + `Q11PhotoAssessment.tsx`). Q11 is now fully AI-assisted with examiner accept/override.
- Q11 photo preview is an in-memory data URL held in React state (`CopyingState.previewData`) and is never written to disk, localStorage, `public/`, Git, project files, or logs. Do not add image persistence. `photoInfo` (original/optimized dims + sizes) is also in-memory only.
- Q11 images are validated + optimized client-side BEFORE upload (`processPhoto` in `imageUtils.ts`): accepted types jpeg/png/webp, ≤10 MB, min 800×600 (rejected with the EXACT low-resolution message), downscale long side >2048 (never upscale), re-encoded JPEG q0.82. Do NOT raise the min-resolution or size thresholds without explicit instruction (it would over-reject legitimate drawings); do NOT lower the quality cap meaningfully (the backend vision model relies on a legible image). The frontend timeout `VISION_CLIENT_TIMEOUT` (120s) must stay at/above the backend `VISION_TIMEOUT` (120s).
- Q6 Naming uses a 12-object library; the batch slot keys MUST stay `naming.wristwatch`/`naming.pencil` (backend `evaluate_naming` only accepts those keys) — only the `expected` value changes per selected object. Do not add new batch keys or change Q6's 2-point total.
- **Patient-mode blocking at Orientation to Place and Copying is INTENTIONAL and user-approved.** `sectionResponseCounts().orientationPlace` requires a configured location + response, so pure Patient mode can never reach 5/5 there — that is the design, and the UI now explains it ("An examiner is required for this section" + in-section "Examiner required" notices). Do NOT remove/bypass the block or fabricate scores to enable Next; the full flow is Examiner + Patient.
- **Enter-key navigation:** the Next button and the Enter key share ONE `goToNext` path (`MMSEAssessment.tsx` → `SectionNavigationContext` in `primitives.tsx`). Enter advances only from single-line patient response inputs and only when `canAdvance` (section complete AND not AI-assessing). Textarea inputs (Writing) never intercept Enter (newline preserved). If you add a new text input that should NOT trigger section-advance, render it as a plain `<input>` (not via `PatientResponse`) or handle the key explicitly.
- Camera "Take Photo" uses native `<input capture="environment">` + a `getUserMedia` support probe. Camera permission-denied/unsupported handling is best-effort (browser-dependent); Upload Photo always remains available. Do NOT claim the camera was live-tested unless a real device was used.
- Q11 reference figure is loaded server-side from `frontend/public/mmse-copying-figure.png` (or `COPYING_REFERENCE_PATH`). Never accept the reference from the client; never duplicate/regenerate it. Note: the file bytes are JPEG despite the `.png` extension — Pillow and browsers content-sniff so it works; do not "fix" the extension.
- `VISION_PROVIDER` defaults to `ollama`; Gemini/OpenAI require backend `GEMINI_API_KEY`/`OPENAI_API_KEY` and were not live-tested. Provider selection is ONE provider per assessment (no voting/fallback by design).
- `VISION_TIMEOUT` (120s) is separate from the text-MMSE `OLLAMA_BATCH_TIMEOUT` (175s) and frontend 180s budget — do not merge them.
- Q11 vision accuracy is NOT validated: a real blank-canvas false positive was observed. Confidence is a model signal, not clinical certainty; `review_required` must gate final scoring. Do not claim clinical accuracy.
- Copying figure is supplied at `frontend/public/mmse-copying-figure.png` (referenced via `COPYING_REFERENCE_IMAGE`); do not substitute or redraw it.
- `probabilities` in `/predict` are position-indexed (`[0,1,2]`), not keyed by `model.classes_`.
- Do not retrain the model or modify the `/predict` contract without explicit instruction.
- AI results are an assist signal, never a diagnosis; confidence is a model signal, not clinical certainty. Keep these disclaimers in the UI.
- AI secrets (`GEMINI_API_KEY`, `OPENAI_API_KEY`, etc.) must stay backend-only (env/`.env`); never ship them in React or commit them.
- Real AI-model correctness was verified on this machine (Ollama 0.32.9 + Gemma 3 4B): full MMSE batch now evaluates in a SINGLE model call. **Latency warning:** the full 30-item batch takes ~139–158s when it completes and can exceed the 175s backend / 180s browser timeout due to generation variance (Gemma 3 4B runs ~7.4 tok/s split 54% CPU / 46% GPU on a 4 GB GTX 1650). Do not raise the frontend timeout above what the user configured without instruction; never "fix" latency by removing items or lowering quality. Options (larger GPU, smaller/quantized model, streaming, split-batch) require explicit user approval.
- `README.md` has stale claims (e.g., Dataset.csv "not committed") — treat code as authoritative.
