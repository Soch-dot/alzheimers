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

  // "Re-take Assessment": clear MMSE-specific state (responses, step, AI
  // results, score), PRESERVE the selected mode and assessment details, and
  // return to the MMSE intro. This must NOT erase the whole session.
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

  // "Restart Assessment": clear the FULL persisted session and return to
  // Assessment Mode. Verified by a refresh landing on Assessment Mode.
  const handleRestart = () => {
    clearSession();
    setMode(DEFAULT_ASSESSMENT_MODE);
    setDetails(EMPTY_DETAILS);
    setMmse(null);
    setPhase('mode');
    setMmseScore(null);
    setResult(null);
    setError(null);
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
      <Layout>
        <AnimatePresence>
          {error && (
            <ErrorMessage
              message={error}
              onDismiss={() => setError(null)}
            />
          )}
        </AnimatePresence>

        {/* ----------- FIXED PERFECT CENTERED LAYOUT ----------- */}
        <div className="w-full flex justify-center mt-10 px-6">
          <div className="w-full max-w-[1350px] flex flex-col lg:flex-row items-start justify-between gap-12">

            {/* Left: Mode, Assessment Details, MMSE Assessment, or Analysis */}
            <div className="flex-1 max-w-[520px] w-full">
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
                  onRestart={handleRestart}
                />
              ) : phase === 'mmse' ? (
                <MMSEAssessment
                  onComplete={handleMmseComplete}
                  initialStep={mmse?.step}
                  initialPhase={mmse?.phase}
                  initialState={mmse?.state}
                  onSessionStateChange={handleMmseStateChange}
                />
              ) : (
                <FormPanel onSubmit={handleSubmit}>
                  <div className="rounded-xl border border-white/10 bg-white/5 p-4 mb-7 flex items-center justify-between gap-4">
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.08em]">
                        MMSE Score
                      </p>
                      <p className="text-lg font-semibold text-white mt-0.5">
                        {mmseScore} / 30
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={handleRetake}
                      className="text-sm text-gray-400 hover:text-white transition-colors"
                    >
                      Re-take Assessment
                    </button>
                    <button
                      type="button"
                      onClick={handleRestart}
                      className="text-sm text-gray-500 hover:text-rose-300 transition-colors"
                    >
                      Restart Assessment
                    </button>
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

                  <div className="pt-12 border-t border-white/10">
                    <AnalyzeButton
                      onClick={handleAnalyze}
                      loading={loading}
                      disabled={!detailsValidState}
                    />
                  </div>
                </FormPanel>
              )}
            </div>

            {/* Right: Results */}
            <div className="flex-1 max-w-[520px] w-full lg:sticky lg:top-10 lg:h-fit">
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
              ) : phase === 'mode' ? (
                <EmptyState
                  onAnalyze={handleAnalyze}
                  loading={loading}
                  title="Assessment Mode"
                  description={
                    <>
                      Choose who is performing this screening to begin.
                    </>
                  }
                  showAnalyze={false}
                />
              ) : phase === 'details' ? (
                <EmptyState
                  onAnalyze={handleAnalyze}
                  loading={loading}
                  title="Assessment Details"
                  description={
                    <>
                      Enter the patient&apos;s assessment details to begin the
                      screening.
                    </>
                  }
                  showAnalyze={false}
                />
              ) : phase === 'mmse' ? (
                <EmptyState
                  onAnalyze={handleAnalyze}
                  loading={loading}
                  title="MMSE Assessment"
                  description={
                    <>
                      Complete the assessment to generate your screening result.
                    </>
                  }
                  showAnalyze={false}
                />
              ) : (
                <EmptyState onAnalyze={handleAnalyze} loading={loading} />
              )}
            </div>

          </div>
        </div>
      </Layout>
    </AssessmentModeContext.Provider>
  );
}

export default App;
