import React, { useCallback, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import type { ItemState, MmsePhase, ScoreMark } from '../../mmse/state';
import { effectiveCorrect } from '../../mmse/state';
import { useAssessmentMode } from '../../mmse/mode';

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
  const mode = useAssessmentMode();
  if (mode === 'patient') return null;
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

/** Records the patient's actual response. Never auto-scores or auto-evaluates the text. */
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
// IMPORTANT: a transcript only populates the response field. It never triggers
// AI evaluation — assessment starts only from the explicit "Assess MMSE with
// AI" batch action.
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
// Examiner scoring (manual sections / manual review / override)
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
// AI result panel (shown AFTER the batch assessment; never during response
// collection). No per-item "Assessing…" state exists anymore — the batch has
// one global loading state.
// ---------------------------------------------------------------------------
const ghostButtonClass =
  'px-4 py-1.5 rounded-lg text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 transition-all duration-200 hover:bg-white/10';

interface AIResultPanelProps {
  item: ItemState;
  update: (patch: Partial<ItemState>) => void;
  /** Re-run the batch for still-unscored items. */
  onRetry?: () => void;
  notAssessedHint?: string;
}

export const AIResultPanel: React.FC<AIResultPanelProps> = ({
  item,
  update,
  onRetry,
  notAssessedHint = 'Not assessed by AI yet.',
}) => {
  const [showManual, setShowManual] = useState(false);
  const mode = useAssessmentMode();
  const examinerView = mode !== 'patient';
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

  // Not yet assessed (or the item failed and has no score).
  if (!item.aiScore) {
    const empty = (item.response ?? '').trim() === '';
    // Patient mode: no technical/AI failure details — keep it neutral.
    if (!examinerView && empty) return <p className="text-xs text-gray-500">Response required</p>;
    return (
      <div>
        {empty ? (
          <p className="text-xs text-gray-500">Response required</p>
        ) : item.error ? (
          examinerView ? (
            <>
              <p className="text-xs text-amber-300">AI assessment unavailable for this item.</p>
              <details className="mt-1">
                <summary className="cursor-pointer text-[11px] text-gray-600 hover:text-gray-400 transition-colors">
                  View technical details
                </summary>
                <pre className="mt-1 text-[10px] leading-relaxed text-gray-600 whitespace-pre-wrap break-words max-h-28 overflow-auto rounded-md bg-white/[0.03] p-2">
                  {item.error}
                </pre>
              </details>
            </>
          ) : (
            <p className="text-xs text-amber-300">This response could not be assessed.</p>
          )
        ) : (
          <p className="text-xs text-gray-500">{notAssessedHint}</p>
        )}
        <div className="flex flex-wrap items-center gap-2 mt-3">
          {examinerView && onRetry && item.manual === null && (
            <button type="button" onClick={onRetry} className={ghostButtonClass}>
              Retry
            </button>
          )}
          {examinerView && (
            <button type="button" onClick={() => setShowManual((v) => !v)} className={ghostButtonClass}>
              Score manually
            </button>
          )}
        </div>
        {examinerView && showManual && manualControls}
      </div>
    );
  }

  const ai = item.aiScore;
  // Patient mode: hide AI score details, confidence, and manual override.
  if (!examinerView) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-emerald-300">✓</span>
        <p className="text-xs text-gray-400">Response assessed</p>
      </div>
    );
  }
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <span className={`text-sm font-semibold ${ai.correct ? 'text-emerald-300' : 'text-rose-300'}`}>
          {ai.correct ? '✓ Correct response' : '✕ Incorrect response'}
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
// AI-scored item card.
// Phase 'collect'      : question → patient response (typed or speech) →
//                        "Response recorded". No AI, no scores, no confidence.
// Phase 'assessed/error': question → response → AI result panel.
// ---------------------------------------------------------------------------
interface AIScoredResponseProps {
  question: string;
  hint?: string;
  /**
   * When set (e.g. an Orientation to Place item whose location field has not
   * been configured yet), the item records the response but is not AI-evaluated
   * and shows this examiner-facing notice instead of manual scoring.
   */
  disabledNotice?: string;
  item: ItemState;
  update: (patch: Partial<ItemState>) => void;
  speech?: boolean;
  placeholder?: string;
  multiline?: boolean;
  top?: React.ReactNode;
  phase: MmsePhase;
  onRetry?: () => void;
}

export const AIScoredResponse: React.FC<AIScoredResponseProps> = ({
  question,
  hint,
  disabledNotice,
  item,
  update,
  speech = true,
  placeholder = "Patient's response",
  multiline = false,
  top,
  phase,
  onRetry,
}) => {
  const mode = useAssessmentMode();
  const examinerView = mode !== 'patient';
  // Expected answers are examiner-only — never shown to the patient.
  const visibleHint = examinerView ? hint : undefined;

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

  const speechHook = useSpeechRecognition((transcript) => {
    onResponseChange(transcript);
  });
  const { supported: speechSupported, listening, error: speechError, start, stop } = speechHook;

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

  if (disabledNotice) {
    return (
      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4">
        {top}
        <div>
          <p className="text-sm font-medium text-white">{question}</p>
          {visibleHint && <p className="text-xs text-gray-500 mt-1">{visibleHint}</p>}
        </div>
        <PatientResponse value={item.response} onChange={onResponseChange} placeholder={placeholder} />
        {phase === 'collect' && item.response.trim() ? (
          <div className="pt-3 border-t border-white/10 flex items-center gap-2">
            <span className="text-emerald-300">✓</span>
            <p className="text-xs text-gray-400">Response recorded</p>
          </div>
        ) : null}
        <p className="text-xs text-amber-300/80">{disabledNotice}</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 md:p-5 space-y-4">
      {top}
      <div>
        <p className="text-sm font-medium text-white">{question}</p>
        {visibleHint && <p className="text-xs text-gray-500 mt-1">{visibleHint}</p>}
      </div>
      {responseField}
      {phase === 'collect' ? (
        item.response.trim() ? (
          <div className="pt-3 border-t border-white/10 flex items-center gap-2">
            <span className="text-emerald-300">✓</span>
            <p className="text-xs text-gray-400">Response recorded</p>
          </div>
        ) : (
          <div className="pt-3 border-t border-white/10 flex items-center gap-2">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-600" />
            <p className="text-xs text-gray-600">Response required</p>
          </div>
        )
      ) : (
        <div className="pt-4 border-t border-white/10">
          <AIResultPanel
            item={item}
            update={update}
            onRetry={onRetry}
            notAssessedHint="Not assessed by AI yet."
          />
        </div>
      )}
    </div>
  );
};

interface SectionShellProps {
  title: string;
  score: number;
  maxScore: number;
  instructions?: React.ReactNode;
  children: React.ReactNode;
  phase: MmsePhase;
  responseCount?: number;
  /** 'ai' sections are scored by the AI batch; 'observation' sections are recorded by the examiner (vision later). */
  kind?: 'ai' | 'observation';
}

export const SectionShell: React.FC<SectionShellProps> = ({
  title,
  score,
  maxScore,
  instructions,
  children,
  phase,
  responseCount = 0,
  kind,
}) => {
  const mode = useAssessmentMode();
  const examinerView = mode !== 'patient';
  return (
    <motion.div
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <GlassCard>
        <div className="flex items-start justify-between gap-4 mb-6">
          <div>
            <h3 className="text-xl md:text-2xl font-semibold text-white tracking-tight">{title}</h3>
            {kind && (
              <span
                className={`inline-block mt-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] px-2 py-0.5 rounded-md border ${
                  kind === 'ai'
                    ? 'text-blue-300 bg-blue-500/10 border-blue-400/20'
                    : 'text-amber-300 bg-amber-500/10 border-amber-400/20'
                }`}
              >
                {kind === 'ai' ? 'AI-assessed' : 'Observation-based'}
              </span>
            )}
          </div>
          {examinerView ? (
            phase === 'collect' ? (
              <span className="shrink-0 text-sm font-semibold text-white/80 bg-white/5 border border-white/10 rounded-xl px-3 py-1.5">
                {responseCount} <span className="text-gray-500">/ {maxScore} responses</span>
              </span>
            ) : (
              <span className="shrink-0 text-sm font-semibold text-white/80 bg-white/5 border border-white/10 rounded-xl px-3 py-1.5">
                {score} <span className="text-gray-500">/ {maxScore}</span>
              </span>
            )
          ) : null}
        </div>
        {examinerView && instructions && <div className="mb-6">{instructions}</div>}
        <div className="space-y-3">{children}</div>
      </GlassCard>
    </motion.div>
  );
};