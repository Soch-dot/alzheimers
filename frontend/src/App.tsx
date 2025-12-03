import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { PatientInput, PredictionResponse } from './api';
import { predictAlzheimers } from './api';
import {
  Layout,
  FormPanel,
  InputField,
  SelectField,
  AnalyzeButton,
  ResultCard,
  ErrorMessage,
  EmptyState,
} from './components';

function App() {
  const [formData, setFormData] = useState<PatientInput>({
    age: 70,
    sex: 1,
    education_years: 12,
    mmse: 28,
    cdr: 0,
    ses: 2,
  });

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

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: PatientInput = {
        age: formData.age,
        sex: formData.sex,
        education_years: formData.education_years,
        mmse: formData.mmse,
        cdr: formData.cdr,
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

          {/* Left: Form */}
          <div className="flex-1 max-w-[520px] w-full">
            <FormPanel onSubmit={handleSubmit}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-7">

                <InputField label="Age" name="age" value={formData.age} onChange={handleChange} min={50} max={100} required />

                <SelectField
                  label="Gender"
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                  options={[
                    { value: 1, label: 'Male' },
                    { value: 0, label: 'Female' },
                  ]}
                  required
                />

                <InputField
                  label="MMSE Score"
                  name="mmse"
                  value={formData.mmse}
                  onChange={handleChange}
                  min={0}
                  max={30}
                  required
                  hint="Range: 0-30"
                />

                <InputField
                  label="CDR Score"
                  name="cdr"
                  value={formData.cdr}
                  onChange={handleChange}
                  min={0}
                  max={2}
                  step={0.5}
                  required
                  hint="Range: 0-2"
                />

                <InputField
                  label="Education"
                  name="education_years"
                  value={formData.education_years}
                  onChange={handleChange}
                  min={0}
                  max={25}
                  required
                  hint="Years"
                />

                <InputField
                  label="SES"
                  name="ses"
                  value={formData.ses}
                  onChange={handleChange}
                  min={1}
                  max={5}
                  required
                  hint="Range: 1-5"
                />

              </div>

              <div className="pt-12 border-t border-white/10">
                <AnalyzeButton onClick={handleAnalyze} loading={loading} />
              </div>
            </FormPanel>
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
