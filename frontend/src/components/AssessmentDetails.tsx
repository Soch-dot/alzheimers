import React from 'react';
import { motion } from 'framer-motion';
import type { PatientInput } from '../api';
import { FormPanel, InputField, SelectField } from './';

interface AssessmentDetailsProps {
  formData: PatientInput;
  onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void;
  onContinue: () => void;
}

/**
 * Pre-MMSE demographic collection (Assessment Details).
 * The values live in App-level `formData` so they survive the whole flow:
 * Assessment Details → MMSE → AI assessment → Summary → Analysis.
 * Continue stays disabled until every field is within the existing ranges.
 */
export const AssessmentDetails: React.FC<AssessmentDetailsProps> = ({
  formData,
  onChange,
  onContinue,
}) => {
  const valid =
    formData.age >= 50 &&
    formData.age <= 100 &&
    (formData.sex === 0 || formData.sex === 1) &&
    formData.education_years >= 0 &&
    formData.education_years <= 25 &&
    formData.ses >= 1 &&
    formData.ses <= 5;

  return (
    <FormPanel onSubmit={(e) => {
      e.preventDefault();
      if (valid) onContinue();
    }}>
      <div>
        <h2 className="text-xl md:text-2xl font-semibold text-white tracking-tight">
          Assessment Details
        </h2>
        <p className="text-sm text-gray-400 font-light mt-2">
          Enter the information required for the risk assessment.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-7">
        <InputField
          label="Age"
          name="age"
          value={formData.age}
          onChange={onChange}
          min={50}
          max={100}
          required
        />

        <SelectField
          label="Sex"
          name="sex"
          value={formData.sex}
          onChange={onChange}
          options={[
            { value: 1, label: 'Male' },
            { value: 0, label: 'Female' },
          ]}
          required
        />

        <InputField
          label="Education (years)"
          name="education_years"
          value={formData.education_years}
          onChange={onChange}
          min={0}
          max={25}
          required
        />

        <InputField
          label="SES"
          name="ses"
          value={formData.ses}
          onChange={onChange}
          min={1}
          max={5}
          required
          hint="Range: 1-5"
        />
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