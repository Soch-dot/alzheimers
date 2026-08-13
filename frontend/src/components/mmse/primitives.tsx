import React, { useCallback, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import type { ItemState, ScoreMark } from '../../mmse/state';
import { effectiveCorrect } from '../../mmse/state';
import { AI_CONFIDENCE_REVIEW_THRESHOLD } from '../../mmse/config';
import { evaluateMmseItem, extractApiError } from '../../api';

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
  onKeyDown?: (event: React.KeyboardEvent) => void;
  rightSlot?: React.ReactNode;
}

/** Records the patient's actual response. Never auto-scores the text. */
export const PatientResponse: React.FC<PatientResponseProps> = ({
  value,
  onChange,
  placeholder,
  multiline = false,
  onKeyDown,
  rightSlot,
}) => {
  return (
    <div>
      <p className={`${labelClass} text-gray-400`}>Patient response</p>
      {multiline ? (
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          rows={4}
          className={`${inputClass} resize-none`}
        />
      ) : (
        <div className="relative">
          <input
            type="text"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            className={`${inputClass} ${rightSlot ? 'pr-16' : ''}`}
          />
          {rightSlot && (
            <div className="absolute right-2 top-1/2 -translate-y-1/2">{rightSlot}</div>
          )}
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Speech capture (Web Speech API, no new dependency). Graceful degradation:
// when unsupported, the examiner types the response instead.
// ---------------------------------------------------------------------------
type SpeechRecognitionLike = {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  continuous: boolean;
  start: () => void;
  stop: () => void;
  onresult: ((event: any) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: any) => void) | null;
};

export function useSpeechRecognition(onFinalTranscript: (text: string) => void) {
  const [supported] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    const w = window as any;
    return !!(w.SpeechRecognition || w.webkitSpeechRecognition);
  });
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const cbRef = useRef(onFinalTranscript);
  cbRef.current = onFinalTranscript;

  const stop = useCallback(() => {
    recognitionRef.current?.stop?.();
    setListening(false);
  }, []);

  const start = useCallback(() => {
    const w = window as any;
    const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!Ctor) {
      setError('Speech input is not supported in this browser.');
      return;
    }
    setError(null);
    try {
      const rec: SpeechRecognitionLike = new Ctor();
      rec.lang = 'en-US';
      rec.interimResults = false;
      rec.maxAlternatives = 1;
      rec.continuous = false;
      rec.onresult = (event: any) => {
        const transcript = event?.results?.[0]?.[0]?.transcript ?? '';
        if (transcript) cbRef.current(transcript);
      };
      rec.onend = () => setListening(false);
      rec.onerror = (e: any) => {
        setListening(false);
        setError(e?.error ? `Speech input failed: ${e.error}` : 'Speech input failed.');
      };
      recognitionRef.current = rec;
      rec.start();
      setListening(true);
    } catch {
      setError('Could not start speech recognition.');
      setListening(false);
    }
  }, []);

  return { supported, listening, error, start, stop };
}

// ---------------------------------------------------------------------------
// Examiner scoring (manual sections / manual review)
// ---------------------------------------------------------------------------
interface ExaminerScoringProps {
  label?: string;
  labelVariant?: 'label' | 'row';
  hint?: string;
  correctLabel?: string;
  incorrectLabel?: string;
  value: ScoreMark;
  onChange: (value: ScoreMark) => void;
}

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

// ---------------------------------------------------------------------------
// AI assessment result panel (shared by scored items and spelling letter rows)
// ---------------------------------------------------------------------------
const primaryButtonClass =
  'px-4 py-2 rounded-lg text-sm font-semibold bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white transition-all duration-200 hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed';

const ghostButtonClass =
  'px-4 py-1.5 rounded-lg text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 transition-all duration-200 hover:bg-white/10';

interface AIAssessmentPanelProps {
  item: ItemState;
  update: (patch: Partial<ItemState>) => void;
  onAssess: () => void;
  idleEmptyHint?: string;
}

export const AIAssessmentPanel: React.FC<AIAssessmentPanelProps> = ({
  item,
  update,
  onAssess,
  idleEmptyHint = "Enter the patient's response, then assess.",
}) => {
  const [showManual, setShowManual] = useState(false);
  const effective = effectiveCorrect(item);

  const manualControls = (
    <div className="flex items-center gap-2 mt-3">
      <span className="text-xs text-gray-500">Manual score:</span>
      <button
        type="button"
        onClick={() => update({ manual: true })}
        className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all duration-200 border ${
          item.manual === true
            ? 'bg-emerald-500/20 border-emerald-400/40 text-emerald-300'
            : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
        }`}
      >
        Correct
      </button>
      <button
        type="button"
        onClick={() => update({ manual: false })}
        className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all duration-200 border ${
          item.manual === false
            ? 'bg-rose-500/20 border-rose-400/40 text-rose-300'
            : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
        }`}
      >
        Incorrect
      </button>
    </div>
  );

  if (item.status === 'idle') {
    if (!item.response.trim()) {
      return <p className="text-xs text-gray-500">{idleEmptyHint}</p>;
    }
    return (
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-gray-500">Patient response captured</p>
        <button type="button" onClick={onAssess} className={primaryButtonClass}>
          Assess with AI
        </button>
      </div>
    );
  }

  if (item.status === 'assessing') {
    return (
      <div className="flex items-center gap-3">
        <span className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-gray-400">Assessing response…</p>
      </div>
    );
  }

  if (item.status === 'error') {
    return (
      <div>
        <p className="text-sm font-semibold text-amber-300">AI assessment unavailable.</p>
        {item.error && <p className="text-xs text-gray-400 mt-1">{item.error}</p>}
        <div className="flex flex-wrap items-center gap-2 mt-3">
          <button type="button" onClick={onAssess} className={ghostButtonClass}>
            Retry
          </button>
          <button
            type="button"
            onClick={() => setShowManual((v) => !v)}
            className={ghostButtonClass}
          >
            Manual review
          </button>
        </div>
        {showManual && manualControls}
      </div>
    );
  }

  const ai = item.aiScore;
  if (!ai) return null;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <span className={`text-sm font-semibold ${ai.correct ? 'text-emerald-300' : 'text-rose-300'}`}>
          {ai.correct ? '✓ Correct' : '✗ Incorrect'}
        </span>
        {item.reviewRequired && !item.reviewed && (
          <span className="text-xs font-semibold text-amber-300">⚠ Review required</span>
        )}
      </div>
      <div className="flex items-center gap-4 mt-1">
        <p className="text-xs text-gray-400">AI confidence: {Math.round(ai.confidence * 100)}%</p>
        <p className="text-xs text-gray-500">
          Score: {effective === null ? '—' : effective ? '1' : '0'} / 1
        </p>
      </div>
      {ai.reason && (
        <p className="text-xs text-gray-500 mt-1.5 italic">“{ai.reason}”</p>
      )}
      {item.reviewRequired && !item.reviewed ? (
        <div className="flex flex-wrap items-center gap-2 mt-3">
          <button
            type="button"
            onClick={() => update({ reviewed: true, status: 'assessed' })}
            className="px-4 py-1.5 rounded-lg text-sm font-semibold border border-emerald-400/30 bg-emerald-500/10 text-emerald-300 transition-all duration-200 hover:bg-emerald-500/20"
          >
            Accept AI result
          </button>
          <button type="button" onClick={() => setShowManual((v) => !v)} className={ghostButtonClass}>
            Override
          </button>
        </div>
      ) : (
        <div className="mt-2">
          {item.manual !== null ? (
            <p className="text-xs text-gray-400">
              Examiner override:{' '}
              <span className={item.manual ? 'text-emerald-300' : 'text-rose-300'}>
                {item.manual ? 'Correct' : 'Incorrect'}
              </span>
            </p>
          ) : (
            <button
              type="button"
              onClick={() => setShowManual((v) => !v)}
              className="text-xs text-gray-500 hover:text-white transition-colors"
            >
              {showManual ? 'Hide override' : 'Review / Override'}
            </button>
          )}
          {showManual && manualControls}
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// AI-scored item card: question → patient response → AI assessment → score
// ---------------------------------------------------------------------------
interface AIScoredResponseProps {
  section: string;
  itemKey: string;
  question: string;
  hint?: string;
  expected: string;
  /** When false (e.g. expected answer not configured), the item is manual-only. */
  aiEnabled?: boolean;
  item: ItemState;
  update: (patch: Partial<ItemState>) => void;
  speech?: boolean;
  placeholder?: string;
  multiline?: boolean;
  top?: React.ReactNode;
}

export const AIScoredResponse: React.FC<AIScoredResponseProps> = ({
  section,
  itemKey,
  question,
  hint,
  expected,
  aiEnabled = true,
  item,
  update,
  speech = true,
  placeholder = "Patient's response",
  multiline = false,
  top,
}) => {
  const busyRef = useRef(false);

  const onResponseChange = (value: string) => {
    if (value === item.response) return;
    update({
      response: value,
      status: 'idle',
      aiScore: null,
      reviewRequired: false,
      reviewed: false,
      manual: null,
      error: null,
    });
  };

  const assess = useCallback(
    async (text: string) => {
      if (busyRef.current) return;
      if (!text.trim()) return;
      busyRef.current = true;
      update({ status: 'assessing', error: null });
      try {
        const result = await evaluateMmseItem({
          section,
          item_key: itemKey,
          question,
          response: text,
          expected,
        });
        busyRef.current = false;
        update({
          status: 'assessed',
          aiScore: {
            correct: result.correct,
            confidence: result.confidence,
            reason: result.reason,
          },
          reviewRequired: result.confidence < AI_CONFIDENCE_REVIEW_THRESHOLD,
          reviewed: false,
          error: null,
        });
      } catch (err) {
        busyRef.current = false;
        update({ status: 'error', error: extractApiError(err) });
      }
    },
    [section, itemKey, question, expected, update]
  );

  const speechHook = useSpeechRecognition((transcript) => {
    onResponseChange(transcript);
    window.setTimeout(() => {
      void assess(transcript);
    }, 0);
  });
  const { supported: speechSupported, listening, error: speechError, start, stop } = speechHook;

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !multiline && !event.shiftKey) {
      event.preventDefault();
      if (item.response.trim() && item.status !== 'assessing') {
        void assess(item.response);
      }
    }
  };

  const micButton = (
    <button
      type="button"
      onClick={() => (listening ? stop() : start())}
      className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all duration-200 ${
        listening
          ? 'bg-rose-500/20 border-rose-400/40 text-rose-300'
          : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
      }`}
    >
      {listening ? 'Stop' : 'Mic'}
    </button>
  );

  const responseField = (
    <div>
      <PatientResponse
        value={item.response}
        onChange={onResponseChange}
        placeholder={placeholder}
        multiline={multiline}
        onKeyDown={handleKeyDown}
        rightSlot={speech && speechSupported && !multiline ? micButton : undefined}
      />
      {speech && !multiline && !speechSupported && (
        <p className="text-[11px] text-gray-500 mt-1.5">
          Speech input not supported in this browser — type the response instead.
        </p>
      )}
      {listening && (
        <p className="text-[11px] text-blue-300/80 mt-1.5">Listening… ask the patient to speak now.</p>
      )}
      {speechError && <p className="text-[11px] text-rose-300/80 mt-1.5">{speechError}</p>}
    </div>
  );

  if (!aiEnabled) {
    return (
      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4">
        {top}
        <div>
          <p className="text-sm font-medium text-white">{question}</p>
          {hint && <p className="text-xs text-gray-500 mt-1">{hint}</p>}
        </div>
        <PatientResponse value={item.response} onChange={onResponseChange} placeholder={placeholder} />
        <div className="pt-4 border-t border-white/10">
          <p className="text-xs text-amber-300/90 mb-2">
            Expected answer not configured for this item — score manually.
          </p>
          <ExaminerScoring value={item.manual} onChange={(value) => update({ manual: value })} />
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4">
      {top}
      <div>
        <p className="text-sm font-medium text-white">{question}</p>
        {hint && <p className="text-xs text-gray-500 mt-1">{hint}</p>}
      </div>
      {responseField}
      <div className="pt-4 border-t border-white/10">
        <AIAssessmentPanel
          item={item}
          update={update}
          onAssess={() => void assess(item.response)}
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
