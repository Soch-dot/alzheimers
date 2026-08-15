import React from 'react';
import { motion } from 'framer-motion';
import type { PredictionResponse } from '../api';
import { PredictionPieChart } from './PredictionPieChart';
import { isExaminerView, useAssessmentMode } from '../mmse/mode';

interface ResultCardProps {
  result: PredictionResponse;
}

/** Consistent patient-facing display rule: round to a whole percent. */
function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result }) => {
  const mode = useAssessmentMode();
  const examinerView = isExaminerView(mode);

  // Raw probability is the DECISION source (threshold 0.40). The calibrated
  // value is DISPLAY ONLY; patients see the calibrated number, never both.
  const rawProbability = result.screening_probability;
  const hasCalibrated =
    typeof result.calibrated_screening_probability === 'number' &&
    result.calibration?.display_calibrated === true &&
    result.calibration?.decision_uses_raw_probability === true;
  const displayProbability = hasCalibrated
    ? (result.calibrated_screening_probability as number)
    : rawProbability;

  const isPositive = result.screening_result === 'positive';

  const probabilityBars = [
    {
      label: 'Nondemented',
      value: 1 - displayProbability,
      color: 'from-emerald-500 via-emerald-400 to-emerald-500',
      dot: 'bg-emerald-400'
    },
    {
      label: 'Dementia-related outcome',
      value: displayProbability,
      color: 'from-rose-500 via-red-400 to-rose-500',
      dot: 'bg-rose-400'
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 40, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.6 }}
      className="relative w-full max-w-[520px]"
    >
      <div className="relative bg-black/40 backdrop-blur-2xl rounded-[2rem] shadow-xl border border-white/10 p-10 md:p-12 overflow-hidden">

        {/* Heading */}
        <h2 className="text-3xl md:text-4xl font-semibold text-white mb-10 tracking-tight">
          Results
        </h2>

        {/* Screening Status */}
        <div className="mb-10">
          <p className="text-xs font-semibold text-gray-400 mb-4 uppercase tracking-widest">
            Screening Result
          </p>

          <span
            className={`px-6 py-3 rounded-xl font-semibold shadow border 
              ${isPositive ? 'text-rose-400 bg-rose-500/20 border-rose-400/30' :
                'text-emerald-400 bg-emerald-500/20 border-emerald-400/30'}`}
          >
            {isPositive ? "Positive Screening Result" : "Negative Screening Result"}
          </span>

          <p className="text-sm text-gray-400 mt-4">
            {result.interpretation.label}:
            <span className="text-white font-semibold ml-2">
              {formatPercent(displayProbability)}
            </span>
          </p>

          <p className="text-sm text-gray-500 mt-2">
            Screening threshold: {formatPercent(result.screening_threshold)}
          </p>
        </div>

        {/* Predicted Class */}
        <div className="mb-12">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-4">
            Predicted Outcome
          </p>

          <span className="px-7 py-3 rounded-xl bg-white/5 border border-white/10 text-white font-semibold shadow-lg">
            {result.predicted_class === 'dementia_related'
              ? 'Dementia-related outcome'
              : result.predicted_class}
          </span>
        </div>

        {/* Horizontal Pie Chart + Legend */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex items-center justify-between gap-10 mb-16"
        >
          {/* Pie Chart */}
          <PredictionPieChart screeningProbability={displayProbability} />

          {/* Legend */}
          <div className="flex flex-col gap-5 text-gray-300 text-sm">
            {probabilityBars.map((p) => (
              <div key={p.label} className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${p.dot}`} />
                <span>
                  {p.label} — {formatPercent(p.value)}
                </span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Bars */}
        <div className="space-y-9">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
            Screening Probability
          </p>

          {probabilityBars.map((bar, idx) => (
            <div key={bar.label} className="space-y-3">
              <div className="flex justify-between text-gray-300">
                <span>{bar.label}</span>
                <span className="font-semibold text-white">
                  {formatPercent(bar.value)}
                </span>
              </div>

              <div className="w-full h-3 bg-black/30 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${bar.value * 100}%` }}
                  transition={{ delay: 0.2 + idx * 0.1, duration: 1 }}
                  className={`h-full bg-gradient-to-r ${bar.color} rounded-full`}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Examiner-only technical details: raw probability must never be shown
            to patients (display-only calibration). */}
        {examinerView && hasCalibrated && (
          <div className="mt-8 p-4 rounded-xl bg-white/5 border border-white/10">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-3">
              Technical Details
            </p>
            <dl className="space-y-1.5 text-xs text-gray-400">
              <div className="flex justify-between">
                <dt>Raw model probability</dt>
                <dd className="text-gray-200 font-mono">
                  {(rawProbability * 100).toFixed(1)}%
                </dd>
              </div>
              <div className="flex justify-between">
                <dt>Calibrated display probability</dt>
                <dd className="text-gray-200 font-mono">
                  {formatPercent(displayProbability)}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt>Decision threshold (raw)</dt>
                <dd className="text-gray-200 font-mono">
                  {formatPercent(result.screening_threshold)}
                </dd>
              </div>
            </dl>
            <p className="text-[10px] text-gray-500 mt-2 leading-relaxed">
              The screening decision uses the raw probability against the 0.40
              threshold. The calibrated value is a display-only transformation.
            </p>
          </div>
        )}

        {/* Non-diagnostic disclaimer */}
        <div className="mt-10 p-4 rounded-xl bg-white/5 border border-white/10">
          <p className="text-xs text-gray-400 leading-relaxed">
            This is a <span className="text-white font-medium">screening result</span>, not a
            diagnosis. The probability is a model-estimated screening probability of a
            dementia-related outcome. It is not clinically validated and does not predict
            future conversion.
          </p>
        </div>

      </div>
    </motion.div>
  );
};