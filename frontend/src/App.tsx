import { useState } from 'react';
import type { PatientInput, PredictionResponse } from './api';
import { predictAlzheimers } from './api';

function App() {
  const [formData, setFormData] = useState<PatientInput>({
    visit: 1,
    mr_delay: 0,
    sex: 1,
    hand: 1,
    age: 70,
    education_years: 12,
    ses: 2,
    mmse: 28,
    cdr: 0,
    etiv: 1500,
    nwbv: 0.75,
    asf: 1.1,
  });

  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'sex' || name === 'hand' ? Number(value) : Number(value),
    }));
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await predictAlzheimers(formData);
      setResult(prediction);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get prediction. Make sure the backend API is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    handleAnalyze();
  };

  const loadSampleData = () => {
    setFormData({
      visit: 1,
      mr_delay: 0,
      sex: 1,
      hand: 1,
      age: 75,
      education_years: 12,
      ses: 2,
      mmse: 24,
      cdr: 0.5,
      etiv: 1700,
      nwbv: 0.72,
      asf: 1.1,
    });
  };

  const loadRandomData = () => {
    // Helper function to generate random integer in range [min, max]
    const randomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;
    
    // Helper function to generate random float in range [min, max] with step
    const randomFloat = (min: number, max: number, step: number = 0.001) => {
      const steps = Math.floor((max - min) / step);
      return Math.round((min + randomInt(0, steps) * step) * 1000) / 1000;
    };

    setFormData({
      visit: randomInt(1, 5),
      mr_delay: randomInt(0, 365),
      sex: Math.random() < 0.5 ? 0 : 1,
      hand: Math.random() < 0.5 ? 0 : 1,
      age: randomInt(50, 100),
      education_years: randomInt(0, 25),
      ses: randomInt(1, 5),
      mmse: randomInt(0, 30),
      cdr: randomFloat(0, 2, 0.5),
      etiv: randomInt(1000, 2500),
      nwbv: randomFloat(0.5, 1.0, 0.001),
      asf: randomFloat(0.8, 1.6, 0.001),
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-20">
        {/* Header */}
        <div className="mb-12 md:mb-16 text-center">
          <h1 className="text-4xl md:text-5xl font-semibold text-gray-900 mb-3 tracking-tight">
            Alzheimer's Risk Assessment
          </h1>
          <p className="text-base text-gray-600">
            Clinical data analysis using machine learning
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-50 border-l-4 border-red-500 rounded-r-lg p-5 shadow-md">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-semibold text-red-800 mb-1">Error</h3>
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Main Content: Form and Results Side by Side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          {/* Left Side: Form Card */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6 md:p-10">
          <form onSubmit={handleSubmit} className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Age */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  Age
                </label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="50"
                  max="100"
                />
              </div>

              {/* Sex */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  Gender
                </label>
                <select
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all cursor-pointer"
                >
                  <option value={1}>Male</option>
                  <option value={0}>Female</option>
                </select>
              </div>

              {/* MMSE Score */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  MMSE Score
                </label>
                <input
                  type="number"
                  name="mmse"
                  value={formData.mmse}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="0"
                  max="30"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 0-30</p>
              </div>

              {/* CDR Score */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  CDR Score
                </label>
                <input
                  type="number"
                  name="cdr"
                  value={formData.cdr}
                  onChange={handleChange}
                  step="0.5"
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="0"
                  max="2"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 0-2</p>
              </div>

              {/* Education Years */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  Education
                </label>
                <input
                  type="number"
                  name="education_years"
                  value={formData.education_years}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="0"
                  max="25"
                />
                <p className="text-xs text-gray-500 mt-1">Years</p>
              </div>

              {/* SES */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  SES
                </label>
                <input
                  type="number"
                  name="ses"
                  value={formData.ses}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="1"
                  max="5"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 1-5</p>
              </div>

              {/* Visit */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  Visit
                </label>
                <input
                  type="number"
                  name="visit"
                  value={formData.visit}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="1"
                  max="5"
                />
              </div>

              {/* Hand */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  Handedness
                </label>
                <select
                  name="hand"
                  value={formData.hand}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all cursor-pointer"
                >
                  <option value={1}>Right</option>
                  <option value={0}>Left</option>
                </select>
              </div>

              {/* MR Delay */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  MR Delay
                </label>
                <input
                  type="number"
                  name="mr_delay"
                  value={formData.mr_delay}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="0"
                />
                <p className="text-xs text-gray-500 mt-1">Days</p>
              </div>

              {/* eTIV */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  eTIV
                </label>
                <input
                  type="number"
                  name="etiv"
                  value={formData.etiv}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="1000"
                  max="2500"
                />
              </div>

              {/* nWBV */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  nWBV
                </label>
                <input
                  type="number"
                  name="nwbv"
                  value={formData.nwbv}
                  onChange={handleChange}
                  step="0.001"
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="0.5"
                  max="1.0"
                />
              </div>

              {/* ASF */}
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  ASF
                </label>
                <input
                  type="number"
                  name="asf"
                  value={formData.asf}
                  onChange={handleChange}
                  step="0.001"
                  className="w-full px-3 py-2.5 text-base text-gray-900 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent transition-all"
                  required
                  min="0.8"
                  max="1.6"
                />
              </div>
            </div>

            {/* Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 pt-6 border-t border-gray-200">
              <button
                type="submit"
                disabled={loading}
                className="flex-1 px-8 py-3.5 bg-gray-900 text-white text-base font-semibold rounded-lg hover:bg-gray-800 active:bg-gray-950 transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-gray-900"
              >
                {loading ? 'Processing...' : 'Analyze'}
              </button>
              <button
                type="button"
                onClick={loadSampleData}
                className="px-6 py-3.5 text-gray-700 text-base font-medium rounded-lg border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50 active:bg-gray-100 transition-all"
              >
                Sample Data
              </button>
              <button
                type="button"
                onClick={loadRandomData}
                className="px-6 py-3.5 text-gray-700 text-base font-medium rounded-lg border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50 active:bg-gray-100 transition-all"
              >
                Random
              </button>
            </div>
          </form>
          </div>

          {/* Right Side: Results Card */}
          <div className="lg:sticky lg:top-8 lg:h-fit">
            {result ? (
              <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6 md:p-10">
                <div className="mb-8">
                  <h2 className="text-2xl md:text-3xl font-semibold text-gray-900 mb-6">Results</h2>
                  
                  <div className="mb-10">
                    <p className="text-sm font-medium text-gray-600 mb-3 uppercase tracking-wide">Predicted Class</p>
                    <div className={`inline-block px-6 py-3 rounded-lg text-base font-semibold shadow-sm ${
                      result.class_index === 0 
                        ? 'text-green-800 bg-green-100 border-2 border-green-300' 
                        : result.class_index === 1
                        ? 'text-amber-800 bg-amber-100 border-2 border-amber-300'
                        : 'text-red-800 bg-red-100 border-2 border-red-300'
                    }`}>
                      {result.predicted_class}
                    </div>
                  </div>

                  <div className="space-y-6">
                    <p className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-4">Class Probabilities</p>
                    
                    {/* Nondemented */}
                    <div>
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-base font-medium text-gray-700">Nondemented</span>
                        <span className="text-base font-bold text-gray-900">
                          {(result.probabilities.Nondemented * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-green-500 to-green-600 transition-all duration-700 ease-out rounded-full"
                          style={{ width: `${result.probabilities.Nondemented * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Converted */}
                    <div>
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-base font-medium text-gray-700">Converted</span>
                        <span className="text-base font-bold text-gray-900">
                          {(result.probabilities.Converted * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-amber-400 to-amber-500 transition-all duration-700 ease-out rounded-full"
                          style={{ width: `${result.probabilities.Converted * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Demented */}
                    <div>
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-base font-medium text-gray-700">Demented</span>
                        <span className="text-base font-bold text-gray-900">
                          {(result.probabilities.Demented * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-red-500 to-red-600 transition-all duration-700 ease-out rounded-full"
                          style={{ width: `${result.probabilities.Demented * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6 md:p-10 h-full flex items-center justify-center min-h-[400px]">
                <div className="text-center">
                  <svg className="mx-auto h-16 w-16 mb-6 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-base font-medium text-gray-600 mb-2">Results will appear here</p>
                  <p className="text-sm text-gray-400 mb-6">Fill in the form and click analyze to see predictions</p>
                  <button
                    type="button"
                    onClick={handleAnalyze}
                    disabled={loading}
                    className="px-8 py-3 bg-gray-900 text-white text-base font-semibold rounded-lg hover:bg-gray-800 active:bg-gray-950 transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-gray-900"
                  >
                    {loading ? 'Processing...' : 'Analyze'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

