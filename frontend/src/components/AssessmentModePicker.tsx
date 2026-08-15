import React from 'react';
import { motion } from 'framer-motion';
import type { AssessmentMode } from '../mmse/mode';
import { ASSESSMENT_MODES } from '../mmse/mode';
import { FormPanel } from './';

interface AssessmentModePickerProps {
  selected: AssessmentMode;
  onSelect: (mode: AssessmentMode) => void;
  onContinue: () => void;
}

/**
 * First step of the flow: choose who is performing the assessment.
 *
 * The selection gates which functional interactions are available later
 * (Patient hides examiner-only UI). It is persisted (along with the Assessment
 * Details fields) in localStorage and restored on the next visit. The choice
 * itself never changes MMSE scoring.
 */
export const AssessmentModePicker: React.FC<AssessmentModePickerProps> = ({
  selected,
  onSelect,
  onContinue,
}) => {
  return (
    <FormPanel
      onSubmit={(e) => {
        e.preventDefault();
        onContinue();
      }}
    >
      <div>
        <h2 className="text-xl md:text-2xl font-semibold text-white tracking-tight">
          Who is performing this assessment?
        </h2>
        <p className="text-sm text-gray-400 font-light mt-2">
          Choose the assessment mode. The mode controls which controls are shown —
          it does not change how the assessment is scored.
        </p>
      </div>

      <div className="space-y-3">
        {ASSESSMENT_MODES.map((mode) => {
          const active = selected === mode.id;
          return (
            <motion.button
              key={mode.id}
              type="button"
              onClick={() => onSelect(mode.id)}
              whileTap={{ scale: 0.99 }}
              className={`w-full text-left rounded-xl border p-4 transition-all duration-200 ${
                active
                  ? 'border-blue-400/50 bg-blue-500/10'
                  : 'border-white/10 bg-white/5 hover:bg-white/10'
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-white">{mode.title}</p>
                  <p className="text-xs text-gray-400 font-light mt-1">
                    {mode.description}
                  </p>
                </div>
                <span
                  className={`shrink-0 w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                    active ? 'border-blue-400' : 'border-white/20'
                  }`}
                >
                  {active && (
                    <span className="w-2 h-2 rounded-full bg-blue-400" />
                  )}
                </span>
              </div>
            </motion.button>
          );
        })}
      </div>

      <div className="pt-12 border-t border-white/10">
        <motion.button
          type="submit"
          whileHover={{ scale: 1.02, y: -2 }}
          whileTap={{ scale: 0.98, y: 0 }}
          className="w-full px-8 py-4 bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white text-base font-semibold rounded-xl hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 transition-all duration-200 shadow-[0_4px_16px_rgba(59,130,246,0.4)] hover:shadow-[0_6px_24px_rgba(59,130,246,0.5)]"
        >
          Continue
        </motion.button>
      </div>
    </FormPanel>
  );
};
