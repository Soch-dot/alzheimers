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

interface ExaminerToggleProps {
  label: string;
  hint?: string;
  value: ScoreMark;
  onChange: (value: ScoreMark) => void;
}

export const ExaminerToggle: React.FC<ExaminerToggleProps> = ({ label, hint, value, onChange }) => {
  const select = (next: boolean) => {
    onChange(value === next ? null : next);
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
      <div className="min-w-0">
        <p className="text-sm font-medium text-white">{label}</p>
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
          Correct
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
          Incorrect
        </button>
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
