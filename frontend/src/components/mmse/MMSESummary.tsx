import React from 'react';
import { motion } from 'framer-motion';
import type { MMSEScores } from '../../mmse/state';
import { GlassCard } from './primitives';

interface MMSESummaryProps {
  total: number;
  scores: MMSEScores;
  onContinue: () => void;
  onRestart: () => void;
  onBack: () => void;
}

const BREAKDOWN: { title: string; key: keyof MMSEScores; max: number }[] = [
  { title: 'Orientation - Time', key: 'orientationTime', max: 5 },
  { title: 'Orientation - Place', key: 'orientationPlace', max: 5 },
  { title: 'Registration', key: 'registration', max: 3 },
  { title: 'Attention', key: 'attention', max: 5 },
  { title: 'Delayed Recall', key: 'delayedRecall', max: 3 },
  { title: 'Naming', key: 'naming', max: 2 },
  { title: 'Repetition', key: 'repetition', max: 1 },
  { title: 'Three-Step Command', key: 'command', max: 3 },
  { title: 'Reading', key: 'reading', max: 1 },
  { title: 'Writing', key: 'writing', max: 1 },
  { title: 'Copying', key: 'copying', max: 1 },
];

export const MMSESummary: React.FC<MMSESummaryProps> = ({
  total,
  scores,
  onContinue,
  onRestart,
  onBack,
}) => {
  const rows = BREAKDOWN.map((row) => ({
    ...row,
    value: scores[row.key],
  }));

  return (
    <motion.div
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <GlassCard>
        <div className="text-center mb-8">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.12em] mb-3">
            MMSE Score
          </p>
          <p className="text-6xl font-semibold text-white tracking-tight">
            {total}
            <span className="text-2xl text-gray-500 font-normal"> / 30</span>
          </p>
        </div>

        <div className="space-y-2 mb-8">
          {rows.map((row) => (
            <div
              key={row.title}
              className="flex items-center justify-between gap-4 rounded-lg border border-white/10 bg-white/[0.03] px-4 py-2.5"
            >
              <span className="text-sm text-gray-300">{row.title}</span>
              <span className="text-sm font-semibold text-white">
                {row.value} <span className="text-gray-500 font-normal">/ {row.max}</span>
              </span>
            </div>
          ))}
        </div>

        <p className="text-xs text-gray-500 text-center mb-6">
          This score is used as one clinical feature for the risk screening. It
          is a research prototype and not a diagnosis.
        </p>

        <div className="flex flex-col gap-3">
          <motion.button
            type="button"
            onClick={onContinue}
            whileHover={{ scale: 1.01, y: -2 }}
            whileTap={{ scale: 0.99, y: 0 }}
            className="w-full px-8 py-3.5 bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white text-base font-semibold rounded-xl hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 transition-all duration-200 shadow-[0_4px_16px_rgba(59,130,246,0.4)] hover:shadow-[0_6px_24px_rgba(59,130,246,0.5)]"
          >
            Continue to Analysis
          </motion.button>
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={onBack}
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              ← Back to Copying
            </button>
            <button
              type="button"
              onClick={onRestart}
              className="text-sm text-gray-500 hover:text-rose-300 transition-colors"
            >
              Restart Assessment
            </button>
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
};
