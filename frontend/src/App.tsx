import { useCallback, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { PredictionResponse } from './api';
import { predictAlzheimers } from './api';
import {
  Layout,
  FormPanel,
  AnalyzeButton,
  ResultCard,
  ErrorMessage,
  EmptyState,
  AssessmentDetails,
  AssessmentModePicker,
  MMSEAssessment,
} from './components';
import { AssessmentModeContext } from './mmse/mode';
import type { AssessmentMode } from './mmse/mode';
import { DEFAULT_ASSESSMENT_MODE } from './mmse/mode';
import { MMSE_SECTIONS } from './mmse/config';
import {
  EMPTY_DETAILS,
  detailsToPatientInput,
  detailsValid,
  sanitizeNumericInput,
  type DetailsDraft,
} from './mmse/details';
import type { MMSEState, MmsePhase } from './mmse/state';
import {
  clearSession,
  loadSession,
  saveSession,
  SESSION_VERSION,
  type AppPhase,
  type MmseSessionPart,
} from './mmse/session';

const SUMMARY_STEP = MMSE_SECTIONS.length + 1;

function App() {
  // ------------------ Assessment Mode (first step, persisted) -------------
  const [mode, setMode] = useState<AssessmentMode>(DEFAULT_ASSESSMENT_MODE);

  // ------------------ Assessment Details (string-based, persisted) --------
  const [details, setDetails] = useState<DetailsDraft>(EMPTY_DETAILS);

  const [phase, setPhase] = useState<AppPhase>('mode');
  const [mmseScore, setMmseScore] = useState<number | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // In-progress MMSE position (step / collect-assess phase / full response
  // state). Set by MMSEAssessment via onSessionStateChange.
  const [mmse, setMmse] = useState<MmseSessionPart | null>(null);
  const [restored, setRestored] = useState(false);

  // THE single authoritative startup restore. Reads the versioned session and
  // restores the exact logical location (mode, details, phase, MMSE step,
  // score, prediction). A single effect makes the one startup decision so no
  // other effect can fight it. `restored` gates the save effect until the
  // restore has landed (avoids immediately overwriting the restored session).
  useEffect(() => {
    const session = loadSession();
    if (session) {
      setMode(session.mode);
      setDetails(session.details);
      setPhase(session.appPhase);
      setMmseScore(session.mmseScore);
      setResult(session.result);
      setMmse(session.mmse);
    }
    setRestored(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist the whole session whenever meaningful state changes. Debounced so
  // text-entry churn (details + MMSE responses) does not write on every
  // keystroke; a refresh still restores the latest meaningful state.
  // A completely empty session (fresh start / after full "Restart Assessment")
  // is NOT re-written, so the cleared key stays gone.
  const isEmptySession =
    phase === 'mode' &&
    mode === DEFAULT_ASSESSMENT_MODE &&
    details.age === '' &&
    details.sex === '' &&
    details.education_years === '' &&
    details.ses === '' &&
    mmse === null &&
    mmseScore === null &&
    result === null;
  useEffect(() => {
    if (!restored || isEmptySession) return;
    const timer = window.setTimeout(() => {
      saveSession({
        version: SESSION_VERSION,
        mode,
        details,
        appPhase: phase,
        mmse,
        mmseScore,
        result,
      });
    }, 400);
    return () => window.clearTimeout(timer);
  }, [restored, isEmptySession, mode, details, phase, mmse, mmseScore, result]);

  const handleDetailsChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    const isNumeric = name === 'age' || name === 'education_years' || name === 'ses';
    const next =
      isNumeric && e.target instanceof HTMLInputElement
        ? sanitizeNumericInput(value)
        : value;
    setDetails((prev) => ({
      ...prev,
      [name]: isNumeric ? next : next === '' ? '' : Number(next),
    }));
  };

  const handleMmseComplete = (score: number) => {
    setMmseScore(score);
    setPhase('form');
  };

  const handleMmseStateChange = useCallback(
    (step: number, phase: MmsePhase, state: MMSEState) => {
      setMmse({ step, phase, state });
    },
    []
  );

  // "Re-take MMSE": clear ONLY the MMSE part (responses, AI results, score) —
  // PRESERVE the selected mode and assessment details — and return to the MMSE
  // intro. This must NOT return to Assessment Mode and must NOT erase the whole
  // session. (Separate from "Restart Assessment": do not conflate the two.)
  const handleRetake = () => {
    setMmse(null);
    setMmseScore(null);
    setResult(null);
    setPhase('mmse');
    // Flush immediately so an instant refresh also lands on the fresh MMSE intro.
    saveSession({
      version: SESSION_VERSION,
      mode,
      details,
      appPhase: 'mmse',
      mmse: null,
      mmseScore: null,
      result: null,
    });
  };

  // "Hand off to examiner" (Patient mode, pending-examiner sections): switch the
  // session to Examiner mode WITHOUT losing any collected patient responses, so
  // the examiner can configure the location, run the AI batch, upload the Q11
  // photo, and finalize the score. Persist immediately (like re-take).
  const handleHandoffToExaminer = () => {
    setMode('examiner');
    saveSession({
      version: SESSION_VERSION,
      mode: 'examiner',
      details,
      appPhase: phase,
      mmse,
      mmseScore,
      result,
    });
  };

  // THE single authoritative full-reset: wipe the ENTIRE persisted session and
  // every piece of application state, then return to the FIRST step
  // (Assessment Mode). Every "Restart Assessment" control reuses this function
  // so restart always means restart. A subsequent refresh stays on Assessment
  // Mode because the cleared session key is never re-written (isEmptySession).
  const handleRestartAssessment = () => {
    clearSession();
    setMode(DEFAULT_ASSESSMENT_MODE);
    setDetails(EMPTY_DETAILS);
    setMmse(null);
    setPhase('mode');
    setMmseScore(null);
    setResult(null);
    setError(null);
  };

  // "Back" from the Analysis form: return to the MMSE Summary WITHOUT clearing
  // anything. The MMSE part is pinned to the summary step so the score and all
  // responses are still shown (phase stays 'mmse'; mmseScore/result untouched).
  const handleBackToSummary = () => {
    setMmse((prev) => (prev ? { ...prev, step: SUMMARY_STEP } : prev));
    setPhase('mmse');
  };

  const handleAnalyze = async () => {
    if (mmseScore === null) {
      setError('Complete the MMSE assessment before analyzing.');
      return;
    }
    if (!detailsValid(details)) {
      setError('Please complete the assessment details correctly.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Convert validated strings to numbers ONLY for the final /predict
      // request. /predict contract is unchanged (numbers only).
      const payload = detailsToPatientInput(details, mmseScore);
      const prediction = await predictAlzheimers(payload);
      setResult(prediction);
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
          'Failed to get prediction. Make sure the backend is running on http://127.0.0.1:8000'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    handleAnalyze();
  };

  const detailsValidState = detailsValid(details);

  return (
    <AssessmentModeContext.Provider value={mode}>
      <Layout compactHero={phase === 'form'}>
        <AnimatePresence>
          {error && (
            <ErrorMessage
              message={error}
              onDismiss={() => setError(null)}
            />
          )}
        </AnimatePresence>

        {/* ----------- FIXED PERFECT CENTERED LAYOUT ----------- */}

        {/* Assessment flow phases (Mode / Details / MMSE): single centered column,
            no side placeholder panels. The questionnaire owns the viewport. */}
        {phase === 'mode' || phase === 'details' || phase === 'mmse' ? (
          <div className="w-full flex justify-center mt-4 px-6">
            <div className="w-full max-w-[860px]">
              {phase === 'mode' ? (
                <AssessmentModePicker
                  selected={mode}
                  onSelect={setMode}
                  onContinue={() => setPhase('details')}
                />
              ) : phase === 'details' ? (
                <AssessmentDetails
                  formData={details}
                  onChange={handleDetailsChange}
                  onBack={() => setPhase('mode')}
                  onContinue={() => setPhase('mmse')}
                  onRestart={handleRestartAssessment}
                />
              ) : (
                <MMSEAssessment
                  onComplete={handleMmseComplete}
                  initialStep={mmse?.step}
                  initialPhase={mmse?.phase}
                  initialState={mmse?.state}
                  onSessionStateChange={handleMmseStateChange}
                  onHandoffToExaminer={handleHandoffToExaminer}
                  onRetake={handleRetake}
                  onRestartAssessment={handleRestartAssessment}
                  onExitToDetails={() => setPhase('details')}
                />
              )}
            </div>
          </div>
        ) : (
          /* Analysis phase: form (left) + result (right) composition. Local full-bleed
             so the analysis row can exceed the global max-w-7xl while staying centered.
             Auto width + negative margins expand past the Layout padding to the
             viewport; the inner max caps at 1400px (or viewport minus a safe margin). */
          <div className="flex justify-center mt-10 lg:-mt-11 lg:-mx-[max(0px,calc(50vw-600px))] lg:px-10">
            <div className="w-full max-w-[min(1400px,calc(100vw-2.5rem))] grid grid-cols-1 xl:grid-cols-[minmax(460px,520px)_minmax(0,1fr)] gap-12 items-start">

              {/* Left: analysis form */}
              <div className="w-full min-w-0">
                <FormPanel onSubmit={handleSubmit}>
                  {/* Navigation row (above the MMSE Score card) */}
                  <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
                    <button
                      type="button"
                      onClick={handleBackToSummary}
                      className="text-sm text-gray-400 hover:text-white transition-colors"
                    >
                      ← Back
                    </button>
                    <button
                      type="button"
                      onClick={handleRetake}
                      className="text-sm text-gray-400 hover:text-white transition-colors"
                    >
                      Re-take MMSE
                    </button>
                    <button
                      type="button"
                      onClick={handleRestartAssessment}
                      className="text-sm text-gray-500 hover:text-rose-300 transition-colors"
                    >
                      Restart Assessment
                    </button>
                  </div>

                  <div className="rounded-xl border border-white/10 bg-white/5 p-4 mb-7 flex items-center justify-between gap-4">
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.08em]">
                        MMSE Score
                      </p>
                      <p className="text-lg font-semibold text-white mt-0.5">
                        {mmseScore} / 30
                      </p>
                    </div>
                  </div>

                  <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 mb-7">
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.08em] mb-3">
                      Assessment Details
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-sm text-gray-400">Age</span>
                        <span className="text-sm font-semibold text-white">{details.age}</span>
                      </div>
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-sm text-gray-400">Sex</span>
                        <span className="text-sm font-semibold text-white">
                          {details.sex === 1 ? 'Male' : details.sex === 0 ? 'Female' : '—'}
                        </span>
                      </div>
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-sm text-gray-400">Education</span>
                        <span className="text-sm font-semibold text-white">
                          {details.education_years} years
                        </span>
                      </div>
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-sm text-gray-400">SES</span>
                        <span className="text-sm font-semibold text-white">{details.ses}</span>
                      </div>
                    </div>
                  </div>

                  <div className="pt-10 border-t border-white/10">
                    <AnalyzeButton
                      onClick={handleAnalyze}
                      loading={loading}
                      disabled={!detailsValidState}
                    />
                  </div>
                </FormPanel>
              </div>

              {/* Right: Results */}
              <div className="w-full min-w-0 xl:sticky xl:top-10 xl:h-fit">
                {loading ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5 }}
                    className="relative bg-black/40 backdrop-blur-2xl rounded-[2rem] shadow-[0_8px_32px_rgba(0,0,0,0.4)] border border-white/10 p-12 min-h-[600px] flex items-center justify-center"
                  >
                    <div className="inline-block w-16 h-16 border-4 border-white/20 border-t-blue-400 rounded-full animate-spin" />
                  </motion.div>
                ) : result ? (
                  <ResultCard result={result} />
                ) : (
                  <EmptyState onAnalyze={handleAnalyze} loading={loading} />
                )}
              </div>

            </div>
          </div>
        )}
      </Layout>
    </AssessmentModeContext.Provider>
  );
}

export default App;
