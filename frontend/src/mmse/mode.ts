import { createContext, useContext } from 'react';

/**
 * Assessment Mode — the first step of the flow. It controls which functional
 * interactions are available:
 *
 * - 'patient'  : the patient performs the assessment (types / speaks responses).
 *                Examiner-only UI (instructions, expected answers, AI score
 *                details, manual override, location config, Q11 camera) is hidden.
 * - 'examiner' : the examiner facilitates/records/uploads/reviews. Everything is
 *                visible (the default full UI).
 * - 'both'     : both roles are active; the examiner and patient share the screen.
 *
 * The mode never changes MMSE scoring — it only gates which UI is shown.
 */
export type AssessmentMode = 'patient' | 'examiner' | 'both';

export const ASSESSMENT_MODES: { id: AssessmentMode; title: string; description: string }[] = [
  {
    id: 'patient',
    title: 'Patient',
    description:
      'The patient performs the assessment and enters their own responses by typing or speaking.',
  },
  {
    id: 'examiner',
    title: 'Examiner',
    description:
      'The examiner facilitates the session: asks the questions, records responses, uploads photos and reviews results.',
  },
  {
    id: 'both',
    title: 'Examiner + Patient',
    description:
      'Both roles are active at once — the patient performs while the examiner records, uploads and reviews.',
  },
];

export function isAssessmentMode(value: unknown): value is AssessmentMode {
  return value === 'patient' || value === 'examiner' || value === 'both';
}

/** True when the examiner-facing controls should be visible for this mode. */
export function isExaminerView(mode: AssessmentMode): boolean {
  return mode !== 'patient';
}

export const DEFAULT_ASSESSMENT_MODE: AssessmentMode = 'examiner';

export const AssessmentModeContext = createContext<AssessmentMode>(DEFAULT_ASSESSMENT_MODE);

export const useAssessmentMode = (): AssessmentMode => useContext(AssessmentModeContext);
