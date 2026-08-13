import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { PatientInput, PredictionResponse } from './api';
import { predictAlzheimers } from './api';
import {
  Layout,
  FormPanel,
  AnalyzeButton,
  ResultCard,
  ErrorMessage,
  EmptyState,
  AssessmentDetails,
  MMSEAssessment,
} from './components';

function App() {
  const [formData, setFormData] = useState<PatientInput>({
    age: 70,
    sex: 1,
    education_years: 12,
    mmse: 0,
    ses: 2,
  });

  const [phase, setPhase] = useState<'details' | 'mmse' | 'form'>('details');
  const [mmseScore, setMmseScore] = useState<number | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: Number(value),
    }));
  };

  const handleMmseComplete = (score: number) => {
    setMmseScore(score);
    setFormData((prev) => ({
      ...prev,
      mmse: score,
    }));
    setPhase('form');
  };

  const handleRestart = () => {
    setPhase('details');
    setMmseScore(null);
    setResult(null);
    setError(null);
    setFormData({
      age: 70,
      sex: 1,
      education_years: 12,
      mmse: 0,
      ses: 2,
    });
  };

  const handleAnalyze = async () => {
    if (mmseScore === null) {
      setError('Complete the MMSE assessment before analyzing.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: PatientInput = {
        age: formData.age,
        sex: formData.sex,
        education_years: formData.education_years,
        mmse: formData.mmse,
        ses: formData.ses,
      };

      console.log("Payload:", payload);
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

  const detailsValid =
    formData.age >= 50 &&
    formData.age <= 100 &&
    (formData.sex === 0 || formData.sex === 1) &&
    formData.education_years >= 0 &&
    formData.education_years <= 25 &&
    formData.ses >= 1 &&
    formData.ses <= 5;

  return (
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

          {/* Left: Assessment Details, MMSE Assessment, or Analysis */}
          <div className="flex-1 max-w-[520px] w-full">
            {phase === 'details' ? (
              <AssessmentDetails
                formData={formData}
                onChange={handleChange}
                onContinue={() => setPhase('mmse')}
              />
            ) : phase === 'mmse' ? (
              <MMSEAssessment onComplete={handleMmseComplete} />
            ) : (
              <FormPanel onSubmit={handleSubmit}>
                <div className="rounded-xl border border-white/10 bg-white/5 p-4 mb-7 flex items-center justify-between gap-4">
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.08em]">
                      MMSE Score
                    </p>
                    <p className="text-lg font-semibold text-white mt-0.5">
                      {formData.mmse} / 30
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleRestart}
                    className="text-sm text-gray-400 hover:text-white transition-colors"
                  >
                    Re-take Assessment
                  </button>
                </div>

                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 mb-7">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.08em] mb-3">
                    Assessment Details
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm text-gray-400">Age</span>
                      <span className="text-sm font-semibold text-white">{formData.age}</span>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm text-gray-400">Sex</span>
                      <span className="text-sm font-semibold text-white">
                        {formData.sex === 1 ? 'Male' : 'Female'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm text-gray-400">Education</span>
                      <span className="text-sm font-semibold text-white">
                        {formData.education_years} years
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm text-gray-400">SES</span>
                      <span className="text-sm font-semibold text-white">{formData.ses}</span>
                    </div>
                  </div>
                </div>

                <div className="pt-12 border-t border-white/10">
                  <AnalyzeButton onClick={handleAnalyze} loading={loading} disabled={!detailsValid} />
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
  );
}

export default App;
