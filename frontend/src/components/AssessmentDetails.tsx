import React from 'react';
import { motion } from 'framer-motion';
import type { DetailsDraft } from '../mmse/details';
import { DETAILS_FIELDS, detailsValid } from '../mmse/details';
import { FormPanel, InputField, SelectField } from './';

interface AssessmentDetailsProps {
  formData: DetailsDraft;
  onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void;
  onBack: () => void;
  onContinue: () => void;
  /** Full session reset — clears everything and returns to Assessment Mode. */
  onRestart: () => void;
}

/**
 * Pre-MMSE demographic collection (Assessment Details).
 * The values live in App-level `details` so they survive the whole flow:
 * Mode → Assessment Details → MMSE → AI assessment → Summary → Analysis, and are
 * restored from localStorage on the next visit.
 *
 * Numeric fields are stored as raw strings (empty is a valid draft state), so
 * the Continue button is enabled IMMEDIATELY once every required field is valid —
 * there is no stale "0" placeholder. Conversion to numbers happens only when the
 * final /predict request is built.
 */
export const AssessmentDetails: React.FC<AssessmentDetailsProps> = ({
  formData,
  onChange,
  onBack,
  onContinue,
  onRestart,
}) => {
  const valid = detailsValid(formData);

  return (
    <FormPanel
      onSubmit={(e) => {
        e.preventDefault();
        if (valid) onContinue();
      }}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl md:text-2xl font-semibold text-white tracking-tight">
            Assessment Details
          </h2>
          <p className="text-sm text-gray-400 font-light mt-2">
            Enter the information required for the risk assessment.
          </p>
        </div>
        <div className="flex items-center gap-4 shrink-0">
          <button
            type="button"
            onClick={onRestart}
            className="text-sm text-gray-500 hover:text-rose-300 transition-colors"
          >
            Restart Assessment
          </button>
          <button
            type="button"
            onClick={onBack}
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            ← Back
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-7">
        {DETAILS_FIELDS.map((field) => {
          const value = formData[field.name];
          const common = {
            label: field.required ? `${field.label} *` : field.label,
            name: field.name,
            value,
            onChange,
            required: field.required,
          };
          if (field.type === 'select') {
            return (
              <SelectField
                key={field.name}
                {...common}
                options={field.options ?? []}
              />
            );
          }
          return (
            <InputField
              key={field.name}
              {...common}
              type="number"
              min={field.min}
              max={field.max}
              hint={field.hint}
            />
          );
        })}
      </div>

      <div className="pt-12 border-t border-white/10">
        <motion.button
          type="submit"
          disabled={!valid}
          whileHover={valid ? { scale: 1.02, y: -2 } : {}}
          whileTap={valid ? { scale: 0.98, y: 0 } : {}}
          className="w-full px-8 py-4 bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white text-base font-semibold rounded-xl hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 transition-all duration-200 shadow-[0_4px_16px_rgba(59,130,246,0.4)] hover:shadow-[0_6px_24px_rgba(59,130,246,0.5)] disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Continue to MMSE
        </motion.button>
      </div>
    </FormPanel>
  );
};
