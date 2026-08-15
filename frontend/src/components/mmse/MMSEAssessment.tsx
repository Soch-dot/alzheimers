import React, { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import type { MMSEState, MmsePhase, SectionId } from '../../mmse/state';
import {
  computeScores,
  computeTotal,
  createInitialMMSEState,
  isSectionComplete,
} from '../../mmse/state';
import {
  applyBatchResultsToDraft,
  buildBatchItems,
  countAssessmentErrors,
  isSectionResponseComplete,
  sectionResponseCounts,
} from '../../mmse/batch';
import { MMSE_SECTIONS } from '../../mmse/config';
import { evaluateMmseBatch, extractApiError, isTimeoutError } from '../../api';
import { useAssessmentMode } from '../../mmse/mode';
import { MMSEIntroduction } from './MMSEIntroduction';
import { MMSESummary } from './MMSESummary';
import { SectionNavigationContext } from './primitives';
import {
  AttentionSection,
  CopyingSection,
  DelayedRecallSection,
  NamingSection,
  OrientationPlaceSection,
  OrientationTimeSection,
  ReadingSection,
  RegistrationSection,
  RepetitionSection,
  ThreeStepCommandSection,
  WritingSection,
  type SectionProps,
} from './sections';

interface MMSEAssessmentProps {
  onComplete: (totalScore: number) => void;
  /** Restored step (0 = intro, 1..11 = sections, 12 = summary). */
  initialStep?: number;
  /** Restored MMSE phase. */
  initialPhase?: MmsePhase;
  /** Restored response/AI state for the in-progress session. */
  initialState?: MMSEState;
  /** Reports the current MMSE position/state so the session can be persisted. */
  onSessionStateChange?: (step: number, phase: MmsePhase, state: MMSEState) => void;
}

const SECTION_COMPONENTS: Record<SectionId, React.FC<SectionProps>> = {
  orientationTime: OrientationTimeSection,
  orientationPlace: OrientationPlaceSection,
  registration: RegistrationSection,
  attention: AttentionSection,
  delayedRecall: DelayedRecallSection,
  naming: NamingSection,
  repetition: RepetitionSection,
  command: ThreeStepCommandSection,
  reading: ReadingSection,
  writing: WritingSection,
  copying: CopyingSection,
};

const INTRO_STEP = 0;
const SUMMARY_STEP = MMSE_SECTIONS.length + 1;

/** User-facing batch error info. The technical detail stays behind a disclosure. */
export interface BatchErrorInfo {
  title: string;
  subtitle: string;
  detail: string | null;
}

export const MMSEAssessment: React.FC<MMSEAssessmentProps> = ({
  onComplete,
  initialStep,
  initialPhase,
  initialState,
  onSessionStateChange,
}) => {
  const [step, setStep] = useState(initialStep ?? INTRO_STEP);
  const [state, setState] = useState<MMSEState>(
    initialState ?? createInitialMMSEState
  );
  const [phase, setPhase] = useState<MmsePhase>(initialPhase ?? 'collect');
  const [batchError, setBatchError] = useState<BatchErrorInfo | null>(null);

  // Debounce persistence of the MMSE position/state so typing does not write on
  // every keystroke, while a refresh still restores the latest meaningful state.
  useEffect(() => {
    if (!onSessionStateChange) return;
    const timer = window.setTimeout(() => {
      onSessionStateChange(step, phase, state);
    }, 400);
    return () => window.clearTimeout(timer);
  }, [step, phase, state, onSessionStateChange]);

  // Shared "go to next section" path used by BOTH the Next button and the Enter
  // key (see SectionNavigationContext / PatientResponseInput). Enter must not
  // advance while the section is incomplete or while AI assessment is running.
  const goToNext = useCallback(() => {
    setStep((current) => Math.min(current + 1, SUMMARY_STEP));
  }, []);

  const mode = useAssessmentMode();

  const update = (updater: (draft: MMSEState) => void) => {
    setState((prev) => {
      const draft = structuredClone(prev);
      updater(draft);
      return draft;
    });
  };

  const restart = () => {
    if (window.confirm('Restart the MMSE assessment? All recorded responses and scores will be cleared.')) {
      setState(createInitialMMSEState());
      setStep(INTRO_STEP);
      setPhase('collect');
      setBatchError(null);
    }
  };

  /**
   * The ONLY AI trigger in the whole questionnaire. Sends one batch request with
   * every collected response that does not already have a score, applies the
   * per-item results, then jumps to the first section that still needs attention.
   */
  const runBatchAssessment = async () => {
    const items = buildBatchItems(state, true);
    if (Object.keys(items).length === 0) {
      setPhase('assessed');
      setBatchError(null);
      return;
    }

    setPhase('assessing');
    setBatchError(null);

    try {
      const outcome = await evaluateMmseBatch(items);
      const next = structuredClone(state);
      applyBatchResultsToDraft(next, outcome);
      setState(next);
      setPhase('assessed');

      const firstIncomplete = MMSE_SECTIONS.findIndex((s) => !isSectionComplete(s.id, next));
      if (firstIncomplete !== -1) {
        setStep(firstIncomplete + 1);
      }
    } catch (err) {
      setPhase('error');
      const detail = extractApiError(err) || null;
      if (isTimeoutError(err)) {
        setBatchError({
          title: 'AI assessment timed out.',
          subtitle:
            'The assessment request took too long and was cancelled. Please retry, or score the items manually.',
          detail,
        });
      } else {
        setBatchError({
          title: 'AI assessment unavailable',
          subtitle: 'The selected AI provider is currently unavailable.',
          detail,
        });
      }
    }
  };

  const total = computeTotal(state);
  const scores = computeScores(state);
  const responseCounts = sectionResponseCounts(state);
  const pendingAssessCount = Object.keys(buildBatchItems(state, true)).length;
  const errorItemCount = countAssessmentErrors(state);
  const allFinalized = MMSE_SECTIONS.every((s) => isSectionComplete(s.id, state));

  const jumpToReview = () => {
    const firstIncomplete = MMSE_SECTIONS.findIndex((s) => !isSectionComplete(s.id, state));
    setStep(firstIncomplete !== -1 ? firstIncomplete + 1 : SUMMARY_STEP);
  };

  if (step === INTRO_STEP) {
    return <MMSEIntroduction onStart={() => setStep(1)} />;
  }

  if (step === SUMMARY_STEP) {
    return (
      <MMSESummary
        phase={phase}
        total={total}
        scores={scores}
        responseCounts={responseCounts}
        batchError={batchError}
        needsReassessCount={pendingAssessCount}
        errorItemCount={errorItemCount}
        allFinalized={allFinalized}
        onAssess={() => void runBatchAssessment()}
        onReview={jumpToReview}
        onContinue={() => onComplete(total)}
        onRestart={restart}
        onBack={() => setStep(SUMMARY_STEP - 1)}
      />
    );
  }

  const section = MMSE_SECTIONS[step - 1];
  const SectionComponent = SECTION_COMPONENTS[section.id];
  const assessing = phase === 'assessing';
  const sectionComplete =
    phase === 'collect'
      ? isSectionResponseComplete(section.id, state)
      : isSectionComplete(section.id, state);

  const canAdvance = sectionComplete && !assessing;
  const navigationValue = { goToNext, canAdvance };

  // Pure Patient mode intentionally cannot complete examiner-dependent sections
  // (Orientation to Place needs the examiner-set location; Copying needs the
  // examiner's photo analysis). The blocking logic stays — the message below
  // explains WHY instead of a generic "Complete all items to continue".
  const patientBlocked =
    mode === 'patient' && (section.id === 'orientationPlace' || section.id === 'copying');

  let centerMessage: React.ReactNode;
  if (assessing) {
    centerMessage = 'AI assessment in progress…';
  } else if (patientBlocked && !sectionComplete) {
    centerMessage = 'An examiner is required for this section';
  } else if (phase === 'collect') {
    centerMessage = sectionComplete ? (
      <span className="text-gray-500">Responses complete</span>
    ) : (
      'Complete all items to continue'
    );
  } else {
    centerMessage = sectionComplete ? (
      <span className="text-gray-500">Section scored</span>
    ) : (
      'Resolve the flagged items to continue'
    );
  }

  return (
    <div className="relative w-full">
      <div className="mb-5">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.08em]">
            MMSE Assessment · {step} of {MMSE_SECTIONS.length}
          </p>
          <button
            type="button"
            onClick={restart}
            className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            Restart
          </button>
        </div>
        <div className="h-1 bg-white/10 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full"
            initial={false}
            animate={{ width: `${(step / MMSE_SECTIONS.length) * 100}%` }}
            transition={{ duration: 0.4, ease: 'easeInOut' }}
          />
        </div>
      </div>

      <SectionNavigationContext.Provider value={navigationValue}>
        <motion.div key={step} initial={false}>
          {assessing && (
            <div className="mb-4 flex items-center gap-3 rounded-xl border border-blue-400/20 bg-blue-500/10 px-4 py-3">
              <span className="inline-block w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-blue-200">Assessing MMSE… please wait.</p>
            </div>
          )}
          <SectionComponent
            state={state}
            update={update}
            phase={phase}
            onRetry={() => void runBatchAssessment()}
          />
        </motion.div>

        <div className="flex items-center justify-between mt-6 gap-3">
          <button
            type="button"
            onClick={() => setStep((current) => Math.max(current - 1, 1))}
            disabled={step === 1 || assessing}
            className="px-5 py-2.5 rounded-lg text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 transition-all duration-200 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Back
          </button>
          <p className="text-xs text-gray-500 text-center">{centerMessage}</p>
          <button
            type="button"
            onClick={goToNext}
            disabled={!sectionComplete || assessing}
            className="px-5 py-2.5 rounded-lg text-sm font-semibold bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white transition-all duration-200 hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {step === MMSE_SECTIONS.length ? 'View Summary' : 'Next'}
          </button>
        </div>
      </SectionNavigationContext.Provider>
    </div>
  );
};