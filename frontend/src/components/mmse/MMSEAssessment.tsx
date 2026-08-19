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
  canNavigateSection,
  countAssessmentErrors,
  getSectionStatus,
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
  /** Hand the assessment over to the examiner (mode -> 'examiner'), preserving all data. */
  onHandoffToExaminer?: () => void;
  /** "Re-take MMSE": MMSE-only reset (preserves mode + details, back to the intro). */
  onRetake?: () => void;
  /** "Restart Assessment": full session reset -> Assessment Mode (App-level authority). */
  onRestartAssessment?: () => void;
  /** "Back" from the MMSE intro: return to Assessment Details (no session clearing). */
  onExitToDetails?: () => void;
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
  onHandoffToExaminer,
  onRetake,
  onRestartAssessment,
  onExitToDetails,
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

  // "Re-take MMSE" (header control): MMSE-only reset. Clears the recorded
  // responses/AI scores locally AND tells the App to clear the persisted MMSE
  // part + score (mode/details stay). Never returns to Assessment Mode.
  const handleReTake = () => {
    if (window.confirm('Re-take the MMSE? All recorded responses and scores will be cleared.')) {
      setState(createInitialMMSEState());
      setStep(INTRO_STEP);
      setPhase('collect');
      setBatchError(null);
      onRetake?.();
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
  const responseCounts = sectionResponseCounts(state, mode);
  const pendingAssessCount = Object.keys(buildBatchItems(state, true)).length;
  const errorItemCount = countAssessmentErrors(state);
  const allFinalized = MMSE_SECTIONS.every((s) => isSectionComplete(s.id, state));
  const pendingSections = MMSE_SECTIONS.filter(
    (s) => getSectionStatus(s.id, state, mode) === 'pending-examiner'
  ).map((s) => ({ id: s.id, title: s.title }));

  const jumpToReview = () => {
    const firstIncomplete = MMSE_SECTIONS.findIndex((s) => !isSectionComplete(s.id, state));
    setStep(firstIncomplete !== -1 ? firstIncomplete + 1 : SUMMARY_STEP);
  };

  if (step === INTRO_STEP) {
    return (
      <div className="mx-auto w-full max-w-[860px]">
        <MMSEIntroduction
          onStart={() => setStep(1)}
          onBack={onExitToDetails}
        />
      </div>
    );
  }

  if (step === SUMMARY_STEP) {
    return (
      <div className="mx-auto w-full max-w-[1250px]">
        <MMSESummary
          phase={phase}
          total={total}
          scores={scores}
          responseCounts={responseCounts}
          batchError={batchError}
          needsReassessCount={pendingAssessCount}
          errorItemCount={errorItemCount}
          allFinalized={allFinalized}
          pendingSections={pendingSections}
          onAssess={() => void runBatchAssessment()}
          onReview={jumpToReview}
          onContinue={() => onComplete(total)}
          onRestart={onRestartAssessment ?? handleReTake}
          onBack={() => setStep(SUMMARY_STEP - 1)}
          onHandoffToExaminer={onHandoffToExaminer}
        />
      </div>
    );
  }

  const section = MMSE_SECTIONS[step - 1];
  const SectionComponent = SECTION_COMPONENTS[section.id];
  const assessing = phase === 'assessing';

  // Navigation (shared by the Next button and the Enter key). Patient mode
  // lets the patient CONTINUE through examiner-dependent sections once their
  // input is complete (pending examiner verification); missing input still
  // blocks navigation.
  const navigable = canNavigateSection(section.id, state, mode, phase);
  const canAdvance = navigable && !assessing;
  const navigationValue = { goToNext, canAdvance };

  const pendingExaminer =
    mode === 'patient' && getSectionStatus(section.id, state, mode) === 'pending-examiner';

  let centerMessage: React.ReactNode;
  if (assessing) {
    centerMessage = 'AI assessment in progress…';
  } else if (mode === 'patient' && (section.id === 'orientationPlace' || section.id === 'copying')) {
    centerMessage = pendingExaminer ? (
      <span className="text-amber-300">Pending examiner verification</span>
    ) : (
      'Complete all items to continue'
    );
  } else if (phase === 'collect') {
    centerMessage = navigable ? (
      <span className="text-gray-500">Responses complete</span>
    ) : (
      'Complete all items to continue'
    );
  } else {
    centerMessage = navigable ? (
      <span className="text-gray-500">Section scored</span>
    ) : (
      'Resolve the flagged items to continue'
    );
  }

  return (
    <div className="relative w-full">
      {/* Centered content column: header/progress, question cards, and navigation
          all share the same readable width, centered inside the wide MMSE shell.
          The page scrolls normally (no internal scroll container). */}
      <div className="mx-auto w-full max-w-[1250px]">
        {/* Unified assessment header: step, controls, progress */}
        <div className="mb-5">
          <div className="flex items-center justify-between gap-4 mb-2.5">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-[0.08em]">
              MMSE Assessment · {step} of {MMSE_SECTIONS.length}
            </p>
            <div className="flex items-center gap-4">
              <button
                type="button"
                onClick={handleReTake}
                className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                Re-take MMSE
              </button>
              {onRestartAssessment && (
                <button
                  type="button"
                  onClick={onRestartAssessment}
                  className="text-xs text-gray-500 hover:text-rose-300 transition-colors"
                >
                  Restart Assessment
                </button>
              )}
            </div>
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
              onHandoffToExaminer={onHandoffToExaminer}
            />
          </motion.div>
        </SectionNavigationContext.Provider>

        {/* Stable navigation: reachable at the end of the section content */}
        <div className="mt-5 pt-4 border-t border-white/10 flex items-center justify-between gap-3">
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
            disabled={!navigable || assessing}
            className="px-5 py-2.5 rounded-lg text-sm font-semibold bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white transition-all duration-200 hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {step === MMSE_SECTIONS.length ? 'View Summary' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};