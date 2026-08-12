import React, { useState } from 'react';
import { motion } from 'framer-motion';
import type { MMSEState, SectionId } from '../../mmse/state';
import {
  computeScores,
  computeTotal,
  createInitialMMSEState,
  isSectionComplete,
} from '../../mmse/state';
import { MMSE_SECTIONS } from '../../mmse/config';
import { MMSEIntroduction } from './MMSEIntroduction';
import { MMSESummary } from './MMSESummary';
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

export const MMSEAssessment: React.FC<MMSEAssessmentProps> = ({ onComplete }) => {
  const [step, setStep] = useState(INTRO_STEP);
  const [state, setState] = useState<MMSEState>(createInitialMMSEState);

  const update = (updater: (draft: MMSEState) => void) => {
    setState((prev) => {
      const draft = structuredClone(prev);
      updater(draft);
      return draft;
    });
  };

  const restart = () => {
    if (window.confirm('Restart the MMSE assessment? All scored answers will be cleared.')) {
      setState(createInitialMMSEState());
      setStep(INTRO_STEP);
    }
  };

  const total = computeTotal(state);

  if (step === INTRO_STEP) {
    return <MMSEIntroduction onStart={() => setStep(1)} />;
  }

  if (step === SUMMARY_STEP) {
    return (
      <MMSESummary
        total={total}
        scores={computeScores(state)}
        onContinue={() => onComplete(total)}
        onRestart={restart}
        onBack={() => setStep(SUMMARY_STEP - 1)}
      />
    );
  }

  const section = MMSE_SECTIONS[step - 1];
  const SectionComponent = SECTION_COMPONENTS[section.id];
  const sectionComplete = isSectionComplete(section.id, state);

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

      <motion.div key={step} initial={false}>
        <SectionComponent state={state} update={update} />
      </motion.div>

      <div className="flex items-center justify-between mt-6 gap-3">
        <button
          type="button"
          onClick={() => setStep((current) => Math.max(current - 1, 1))}
          disabled={step === 1}
          className="px-5 py-2.5 rounded-lg text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 transition-all duration-200 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Back
        </button>
        <p className="text-xs text-gray-500 text-center">
          {sectionComplete ? (
            <span className="text-gray-500">Section scored</span>
          ) : (
            'Complete all items to continue'
          )}
        </p>
        <button
          type="button"
          onClick={() => setStep((current) => Math.min(current + 1, SUMMARY_STEP))}
          disabled={!sectionComplete}
          className="px-5 py-2.5 rounded-lg text-sm font-semibold bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white transition-all duration-200 hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {step === MMSE_SECTIONS.length ? 'View Summary' : 'Next'}
        </button>
      </div>
    </div>
  );
};
