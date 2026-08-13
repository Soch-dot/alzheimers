import React from 'react';
import { motion } from 'framer-motion';
import type { MMSEScores, MmsePhase, SectionId } from '../../mmse/state';
import type { SectionResponseCount } from '../../mmse/batch';
import { GlassCard } from './primitives';

interface MMSESummaryProps {
  phase: MmsePhase;
  total: number;
  scores: MMSEScores;
  responseCounts: Record<SectionId, SectionResponseCount>;
  batchError: string | null;
  /** AI-scored items that still need (re)assessment. */
  needsReassessCount: number;
  allFinalized: boolean;
  onAssess: () => void;
  onReview: () => void;
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

const assessButtonClass =
  'w-full px-8 py-3.5 bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white text-base font-semibold rounded-xl hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 transition-all duration-200 shadow-[0_4px_16px_rgba(59,130,246,0.4)] hover:shadow-[0_6px_24px_rgba(59,130,246,0.5)] disabled:opacity-40 disabled:cursor-not-allowed';

const ghostButtonClass =
  'w-full px-6 py-3 text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 rounded-xl transition-all duration-200 hover:bg-white/10';

export const MMSESummary: React.FC<MMSESummaryProps> = ({
  phase,
  total,
  scores,
  responseCounts,
  batchError,
  needsReassessCount,
  allFinalized,
  onAssess,
  onReview,
  onContinue,
  onRestart,
  onBack,
}) => {
  const footerLinks = (
    <div className="flex items-center justify-between">
      <button
        type="button"
        onClick={onBack}
        className="text-sm text-gray-400 hover:text-white transition-colors"
      >
        ← Back
      </button>
      <button
        type="button"
        onClick={onRestart}
        className="text-sm text-gray-500 hover:text-rose-300 transition-colors"
      >
        Restart Assessment
      </button>
    </div>
  );

  if (phase === 'collect') {
    return (
      <motion.div
        initial={{ opacity: 0, x: 24 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        <GlassCard>
          <div className="text-center mb-8">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.12em] mb-3">
              MMSE Assessment
            </p>
            <p className="text-2xl md:text-3xl font-semibold text-emerald-300 tracking-tight">
              ✓ Responses complete
            </p>
            <p className="text-sm text-gray-400 font-light mt-2">
              All required responses are recorded. The AI is not called while the
              patient answers — assessment begins when you tap the button below.
            </p>
          </div>

          <div className="space-y-2 mb-8">
            {BREAKDOWN.map((row) => {
              const count = responseCounts[row.key];
              return (
                <div
                  key={row.title}
                  className="flex items-center justify-between gap-4 rounded-lg border border-white/10 bg-white/[0.03] px-4 py-2.5"
                >
                  <span className="text-sm text-gray-300">{row.title}</span>
                  <span className="text-sm font-semibold text-white">
                    {count.done} <span className="text-gray-500 font-normal">/ {count.max} responses</span>
                  </span>
                </div>
              );
            })}
          </div>

          <div className="flex flex-col gap-3">
            <button type="button" onClick={onAssess} className={assessButtonClass}>
              Assess MMSE with AI
            </button>
            <p className="text-xs text-gray-500 text-center">
              One batch evaluation of all AI-scored responses. You stay in control:
              low-confidence results are flagged for review and every item can be
              overridden manually.
            </p>
            {footerLinks}
          </div>
        </GlassCard>
      </motion.div>
    );
  }

  if (phase === 'assessing') {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <GlassCard>
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <span className="inline-block w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            <p className="text-lg font-semibold text-white tracking-tight mt-5">Assessing MMSE…</p>
            <p className="text-sm text-gray-400 mt-2">
              One batch request is evaluating all recorded responses. Please wait —
              this usually takes a few seconds.
            </p>
          </div>
        </GlassCard>
      </motion.div>
    );
  }

  if (phase === 'error') {
    return (
      <motion.div
        initial={{ opacity: 0, x: 24 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        <GlassCard>
          <div className="text-center mb-8">
            <p className="text-lg font-semibold text-amber-300 tracking-tight">
              AI assessment unavailable.
            </p>
            {batchError && <p className="text-xs text-gray-400 mt-2">{batchError}</p>}
            <p className="text-sm text-gray-400 font-light mt-3">
              No score was assigned. You can retry, or score the items manually in
              each section and continue.
            </p>
          </div>
          <div className="flex flex-col gap-3">
            <button type="button" onClick={onAssess} className={assessButtonClass}>
              Retry
            </button>
            {footerLinks}
          </div>
        </GlassCard>
      </motion.div>
    );
  }

  // assessed
  if (!allFinalized) {
    return (
      <motion.div
        initial={{ opacity: 0, x: 24 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        <GlassCard>
          <div className="text-center mb-8">
            <p className="text-lg font-semibold text-amber-300 tracking-tight">
              {needsReassessCount > 0
                ? `${needsReassessCount} item(s) need (re)assessment`
                : 'Review the flagged items to finish'}
            </p>
            <p className="text-sm text-gray-400 font-light mt-2">
              {needsReassessCount > 0
                ? 'Some responses changed or were not scored. Run the AI assessment again, or score them manually.'
                : 'Low-confidence AI results and any missing items must be accepted or overridden before continuing.'}
            </p>
          </div>
          <div className="flex flex-col gap-3">
            {needsReassessCount > 0 && (
              <button type="button" onClick={onAssess} className={assessButtonClass}>
                Assess MMSE with AI
              </button>
            )}
            <button type="button" onClick={onReview} className={ghostButtonClass}>
              Review items
            </button>
            {footerLinks}
          </div>
        </GlassCard>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <GlassCard>
        <div className="text-center mb-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.12em] mb-3">
            AI assessment complete
          </p>
          <p className="text-6xl font-semibold text-white tracking-tight">
            {total}
            <span className="text-2xl text-gray-500 font-normal"> / 30</span>
          </p>
        </div>

        <div className="space-y-2 mb-8">
          {BREAKDOWN.map((row) => (
            <div
              key={row.title}
              className="flex items-center justify-between gap-4 rounded-lg border border-white/10 bg-white/[0.03] px-4 py-2.5"
            >
              <span className="text-sm text-gray-300">{row.title}</span>
              <span className="text-sm font-semibold text-white">
                {scores[row.key]} <span className="text-gray-500 font-normal">/ {row.max}</span>
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
            className={assessButtonClass}
          >
            Continue to Analysis
          </motion.button>
          {footerLinks}
        </div>
      </GlassCard>
    </motion.div>
  );
};