import React from 'react';
import { motion } from 'framer-motion';
import type { ScoreMark } from '../../mmse/state';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
}

export const GlassCard: React.FC<GlassCardProps> = ({ children, className = '' }) => {
  return (
    <div
      className={`relative bg-black/40 backdrop-blur-2xl rounded-[2rem] shadow-[0_8px_32px_rgba(0,0,0,0.4)] border border-white/10 p-6 md:p-8 overflow-hidden ${className}`}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-black/20 pointer-events-none" />
      <div className="absolute inset-0 bg-gradient-to-t from-blue-500/5 via-transparent to-transparent pointer-events-none" />
      <div className="relative">{children}</div>
    </div>
  );
};

interface ExaminerInstructionsProps {
  children: React.ReactNode;
}

export const ExaminerInstructions: React.FC<ExaminerInstructionsProps> = ({ children }) => {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4 md:p-5">
      <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-blue-300/80 mb-2">
        Examiner instructions <span className="normal-case tracking-normal text-gray-500">— for the examiner, not the patient</span>
      </p>
      <div className="text-sm text-gray-300 leading-relaxed">{children}</div>
    </div>
  );
};

export const inputClass =
  'w-full px-4 py-2.5 text-sm text-white bg-white/5 border border-white/10 rounded-lg focus:ring-2 focus:ring-blue-400/30 focus:border-blue-400/50 focus:bg-white/10 transition-all duration-200 placeholder:text-gray-600';

const labelClass =
  'text-[11px] font-semibold uppercase tracking-[0.08em] mb-2';

interface PatientResponseProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  multiline?: boolean;
}

/**
 * Records the patient's actual response. Never auto-scores the text — the
 * examiner decides correctness separately via ExaminerScoring.
 */
export const PatientResponse: React.FC<PatientResponseProps> = ({
  value,
  onChange,
  placeholder,
  multiline = false,
}) => {
  return (
    <div>
      <p className={`${labelClass} text-gray-400`}>Patient response</p>
      {multiline ? (
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          rows={4}
          className={`${inputClass} resize-none`}
        />
      ) : (
        <input
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          className={inputClass}
        />
      )}
    </div>
  );
};

interface ExaminerScoringProps {
  /** Short section label. Defaults to "Examiner scoring". */
  label?: string;
  /** For action rows, render the label as a normal text row instead of a micro-label. */
  labelVariant?: 'label' | 'row';
  hint?: string;
  correctLabel?: string;
  incorrectLabel?: string;
  value: ScoreMark;
  onChange: (value: ScoreMark) => void;
}

/**
 * Examiner-controlled Correct / Incorrect mark, visually separated from the
 * patient's response. Clicking the selected option clears the mark.
 */
export const ExaminerScoring: React.FC<ExaminerScoringProps> = ({
  label = 'Examiner scoring',
  labelVariant = 'label',
  hint,
  correctLabel = 'Correct',
  incorrectLabel = 'Incorrect',
  value,
  onChange,
}) => {
  const select = (next: boolean) => {
    onChange(value === next ? null : next);
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="min-w-0">
        <p
          className={
            labelVariant === 'row'
              ? 'text-sm font-medium text-white'
              : `${labelClass} text-blue-300/80`
          }
        >
          {label}
        </p>
        {hint && <p className="text-xs text-gray-500 mt-0.5">{hint}</p>}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <button
          type="button"
          onClick={() => select(true)}
          className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all duration-200 border ${
            value === true
              ? 'bg-emerald-500/20 border-emerald-400/40 text-emerald-300'
              : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
          }`}
        >
          {correctLabel}
        </button>
        <button
          type="button"
          onClick={() => select(false)}
          className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all duration-200 border ${
            value === false
              ? 'bg-rose-500/20 border-rose-400/40 text-rose-300'
              : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
          }`}
        >
          {incorrectLabel}
        </button>
      </div>
    </div>
  );
};

interface ScoredResponseProps {
  /** Optional content rendered above the prompt (e.g. an object card). */
  top?: React.ReactNode;
  prompt: string;
  hint?: string;
  response: string;
  onResponseChange: (value: string) => void;
  responsePlaceholder?: string;
  responseMultiline?: boolean;
  score: ScoreMark;
  onScoreChange: (value: ScoreMark) => void;
  scoringLabel?: string;
  correctLabel?: string;
  incorrectLabel?: string;
}

/**
 * Standard item layout: question → patient response → examiner scoring,
 * with a clear visual separation between the two.
 */
export const ScoredResponse: React.FC<ScoredResponseProps> = ({
  top,
  prompt,
  hint,
  response,
  onResponseChange,
  responsePlaceholder,
  responseMultiline,
  score,
  onScoreChange,
  scoringLabel,
  correctLabel,
  incorrectLabel,
}) => {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4">
      {top}
      <div>
        <p className="text-sm font-medium text-white">{prompt}</p>
        {hint && <p className="text-xs text-gray-500 mt-1">{hint}</p>}
      </div>
      <PatientResponse
        value={response}
        onChange={onResponseChange}
        placeholder={responsePlaceholder}
        multiline={responseMultiline}
      />
      <div className="pt-4 border-t border-white/10">
        <ExaminerScoring
          label={scoringLabel}
          correctLabel={correctLabel}
          incorrectLabel={incorrectLabel}
          value={score}
          onChange={onScoreChange}
        />
      </div>
    </div>
  );
};

interface SectionShellProps {
  title: string;
  score: number;
  maxScore: number;
  instructions?: React.ReactNode;
  children: React.ReactNode;
}

export const SectionShell: React.FC<SectionShellProps> = ({
  title,
  score,
  maxScore,
  instructions,
  children,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <GlassCard>
        <div className="flex items-start justify-between gap-4 mb-6">
          <h3 className="text-xl md:text-2xl font-semibold text-white tracking-tight">{title}</h3>
          <span className="shrink-0 text-sm font-semibold text-white/80 bg-white/5 border border-white/10 rounded-xl px-3 py-1.5">
            {score} <span className="text-gray-500">/ {maxScore}</span>
          </span>
        </div>
        {instructions && <div className="mb-6">{instructions}</div>}
        <div className="space-y-3">{children}</div>
      </GlassCard>
    </motion.div>
  );
};
