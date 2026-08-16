import React from 'react';
import { motion } from 'framer-motion';
import type { PredictionResponse } from '../api';
import { PredictionPieChart } from './PredictionPieChart';

interface ResultCardProps {
  result: PredictionResponse;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result }) => {
  const probabilityBars = [
    {
      label: 'Nondemented',
      value: result.probabilities.Nondemented,
      color: 'from-emerald-500 via-emerald-400 to-emerald-500',
      dot: 'bg-emerald-400'
    },
    {
      label: 'Converted',
      value: result.probabilities.Converted,
      color: 'from-amber-500 via-orange-400 to-amber-500',
      dot: 'bg-amber-400'
    },
    {
      label: 'Demented',
      value: result.probabilities.Demented,
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

        {/* Detection Status */}
        <div className="mb-10">
          <p className="text-xs font-semibold text-gray-400 mb-4 uppercase tracking-widest">
            Detection Status
          </p>

          <span
            className={`px-6 py-3 rounded-xl font-semibold shadow border 
              ${result.alzheimers_detected ? 'text-rose-400 bg-rose-500/20 border-rose-400/30' :
                'text-emerald-400 bg-emerald-500/20 border-emerald-400/30'}`}
          >
            {result.alzheimers_detected ? "⚠️ Alzheimer's Detected" : "✅ No Alzheimer’s"}
          </span>

          <p className="text-sm text-gray-400 mt-4">
            Detection Confidence:
            <span className="text-white font-semibold ml-2">
              {result.detection_percentage.toFixed(1)}%
            </span>
          </p>
        </div>

        {/* Predicted Class */}
        <div className="mb-12">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-4">
            Predicted Class
          </p>

          <span className="px-7 py-3 rounded-xl bg-white/5 border border-white/10 text-white font-semibold shadow-lg">
            {result.predicted_class}
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
          <PredictionPieChart probabilities={result.probabilities} />

          {/* Legend */}
          <div className="flex flex-col gap-5 text-gray-300 text-sm">
            {probabilityBars.map((p) => (
              <div key={p.label} className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${p.dot}`} />
                <span>
                  {p.label} — {(p.value * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Bars */}
        <div className="space-y-9">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
            Class Probabilities
          </p>

          {probabilityBars.map((bar, idx) => (
            <div key={bar.label} className="space-y-3">
              <div className="flex justify-between text-gray-300">
                <span>{bar.label}</span>
                <span className="font-semibold text-white">
                  {(bar.value * 100).toFixed(1)}%
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

      </div>
    </motion.div>
  );
};
