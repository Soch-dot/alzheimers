import { useState } from 'react';
import type { PatientInput, PredictionResponse } from './api';
import { predictAlzheimers } from './api';
import './style.css';

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Alzheimer's Risk Prediction
          </h1>
          <p className="text-gray-600 text-lg">
            Enter patient clinical data to assess Alzheimer's risk using machine learning
          </p>
        </div>

        {/* Form Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Age */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Age
                </label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="50"
                  max="100"
                />
              </div>

              {/* Sex */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Gender
                </label>
                <select
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={1}>Male</option>
                  <option value={0}>Female</option>
                </select>
              </div>

              {/* MMSE Score */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  MMSE Score (0-30)
                </label>
                <input
                  type="number"
                  name="mmse"
                  value={formData.mmse}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="0"
                  max="30"
                />
                <p className="text-xs text-gray-500 mt-1">Mini-Mental State Examination</p>
              </div>

              {/* CDR Score */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  CDR Score
                </label>
                <input
                  type="number"
                  name="cdr"
                  value={formData.cdr}
                  onChange={handleChange}
                  step="0.5"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="0"
                  max="2"
                />
                <p className="text-xs text-gray-500 mt-1">Clinical Dementia Rating</p>
              </div>

              {/* Education Years */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Education (Years)
                </label>
                <input
                  type="number"
                  name="education_years"
                  value={formData.education_years}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="0"
                  max="25"
                />
              </div>

              {/* SES */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  SES (Socio-Economic Status)
                </label>
                <input
                  type="number"
                  name="ses"
                  value={formData.ses}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="1"
                  max="5"
                />
              </div>

              {/* Visit */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Visit Number
                </label>
                <input
                  type="number"
                  name="visit"
                  value={formData.visit}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="1"
                  max="5"
                />
              </div>

              {/* Hand */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Handedness
                </label>
                <select
                  name="hand"
                  value={formData.hand}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={1}>Right</option>
                  <option value={0}>Left</option>
                </select>
              </div>

              {/* MR Delay */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  MR Delay (days)
                </label>
                <input
                  type="number"
                  name="mr_delay"
                  value={formData.mr_delay}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="0"
                />
              </div>

              {/* eTIV */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  eTIV (Estimated Total Intracranial Volume)
                </label>
                <input
                  type="number"
                  name="etiv"
                  value={formData.etiv}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="1000"
                  max="2500"
                />
              </div>

              {/* nWBV */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  nWBV (Normalized Whole Brain Volume)
                </label>
                <input
                  type="number"
                  name="nwbv"
                  value={formData.nwbv}
                  onChange={handleChange}
                  step="0.001"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="0.5"
                  max="1.0"
                />
              </div>

              {/* ASF */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  ASF (Atlas Scaling Factor)
                </label>
                <input
                  type="number"
                  name="asf"
                  value={formData.asf}
                  onChange={handleChange}
                  step="0.001"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  min="0.8"
                  max="1.6"
                />
              </div>
            </div>

            {/* Buttons */}
            <div className="flex gap-4 pt-4">
              <button
                type="submit"
                disabled={loading}
                className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Predicting...' : 'Get Prediction'}
              </button>
              <button
                type="button"
                onClick={loadSampleData}
                className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
              >
                Load Sample Data
              </button>
            </div>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 rounded-lg p-4 mb-8">
            <p className="font-medium">Error:</p>
            <p>{error}</p>
          </div>
        )}

        {/* Results Card */}
        {result && (
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Prediction Results</h2>
            
            <div className="mb-6">
              <div className={`inline-block px-6 py-3 rounded-lg font-semibold text-lg ${
                result.class_index === 0 
                  ? 'bg-green-100 text-green-800' 
                  : result.class_index === 1
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-red-100 text-red-800'
              }`}>
                Predicted: {result.predicted_class}
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-700 mb-4">Class Probabilities:</h3>
              
              {/* Nondemented */}
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Nondemented</span>
                  <span className="text-sm font-semibold text-gray-900">
                    {(result.probabilities.Nondemented * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-green-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${result.probabilities.Nondemented * 100}%` }}
                  ></div>
                </div>
              </div>

              {/* Converted */}
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Converted (Early Stage)</span>
                  <span className="text-sm font-semibold text-gray-900">
                    {(result.probabilities.Converted * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-yellow-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${result.probabilities.Converted * 100}%` }}
                  ></div>
                </div>
              </div>

              {/* Demented */}
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Demented</span>
                  <span className="text-sm font-semibold text-gray-900">
                    {(result.probabilities.Demented * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-red-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${result.probabilities.Demented * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

