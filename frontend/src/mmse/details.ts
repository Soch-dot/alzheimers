import type { PatientInput } from '../api';
import type { AssessmentMode } from './mode';
import { isAssessmentMode } from './mode';

/**
 * Assessment Details draft state.
 *
 * The numeric fields (age, education_years, ses) are kept as raw STRINGS so the
 * user can clear them (no leading "0" auto-appears) and type naturally. They are
 * converted to numbers ONLY when building the final /predict payload. sex is a
 * 0/1 select and stays numeric. This fixes the "empty string -> 0" bug: an empty
 * or invalid field simply keeps the Continue button disabled.
 */
export interface DetailsDraft {
  age: string;
  sex: number | '';
  education_years: string;
  ses: string;
}

export const EMPTY_DETAILS: DetailsDraft = {
  age: '',
  sex: '',
  education_years: '',
  ses: '',
};

export interface DetailFieldConfig {
  name: keyof DetailsDraft;
  label: string;
  required: boolean;
  type: 'number' | 'select';
  min?: number;
  max?: number;
  hint?: string;
  options?: { value: number | string; label: string }[];
}

/**
 * CONFIGURABLE: required-field metadata for the Assessment Details step.
 * `required` marks which fields are mandatory before the patient can continue
 * to the MMSE. Used by the details form to render and validate the fields.
 */
export const DETAILS_FIELDS: DetailFieldConfig[] = [
  { name: 'age', label: 'Age', required: true, type: 'number', min: 50, max: 100 },
  {
    name: 'sex',
    label: 'Sex',
    required: true,
    type: 'select',
    options: [
      { value: 1, label: 'Male' },
      { value: 0, label: 'Female' },
    ],
  },
  {
    name: 'education_years',
    label: 'Education (years)',
    required: true,
    type: 'number',
    min: 0,
    max: 25,
  },
  { name: 'ses', label: 'SES', required: true, type: 'number', min: 1, max: 5, hint: 'Range: 1-5' },
];

/**
 * Sanitize a numeric text input: allow digits only, strip leading zeros
 * ("021" -> "21"), keep a lone "0", and allow an empty string (so the field can
 * be cleared instead of snapping back to "0").
 */
export function sanitizeNumericInput(value: string): string {
  const digitsOnly = value.replace(/\D/g, '');
  return digitsOnly.replace(/^0+(?=\d)/, '');
}

export function parseField(field: keyof DetailsDraft, draft: DetailsDraft): number | null {
  const raw = draft[field];
  if (raw === '' || raw === null || raw === undefined) return null;
  if (typeof raw === 'number') return raw;
  const trimmed = raw.trim();
  if (trimmed === '') return null;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : null;
}

/** Whether every required field is present and within range. */
export function detailsValid(draft: DetailsDraft): boolean {
  const age = parseField('age', draft);
  const sex = parseField('sex', draft);
  const edu = parseField('education_years', draft);
  const ses = parseField('ses', draft);

  return (
    age !== null &&
    age >= 50 &&
    age <= 100 &&
    (sex === 0 || sex === 1) &&
    edu !== null &&
    edu >= 0 &&
    edu <= 25 &&
    ses !== null &&
    ses >= 1 &&
    ses <= 5
  );
}

/** Build the numeric /predict payload from the string-based draft. */
export function detailsToPatientInput(draft: DetailsDraft, mmse: number): PatientInput {
  return {
    age: parseField('age', draft) ?? 0,
    sex: parseField('sex', draft) ?? 0,
    education_years: parseField('education_years', draft) ?? 0,
    mmse,
    ses: parseField('ses', draft) ?? 0,
  };
}

// ---------------------------------------------------------------------------
// localStorage persistence — ONLY approved fields are stored: mode, age, sex,
// education_years, ses. MMSE answers, Q11 photos, payloads and API keys are
// NEVER written to localStorage. The key is versioned.
// ---------------------------------------------------------------------------
const STORAGE_KEY = 'alzheimers_assessment_details_v1';

export interface StoredDetails {
  mode: AssessmentMode;
  age: string;
  sex: number | '';
  education_years: string;
  ses: string;
}

export function loadStoredDetails(): StoredDetails | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return null;
    const data = parsed as Record<string, unknown>;
    const mode = isAssessmentMode(data.mode) ? data.mode : undefined;
    if (!mode) return null;

    const age = typeof data.age === 'string' ? data.age : '';
    const sex = data.sex === 0 || data.sex === 1 ? (data.sex as 0 | 1) : '';
    const education_years =
      typeof data.education_years === 'string' ? data.education_years : '';
    const ses = typeof data.ses === 'string' ? data.ses : '';

    const draft: DetailsDraft = { age, sex, education_years, ses };
    if (!detailsValid(draft)) return null;
    return { mode, ...draft };
  } catch {
    return null;
  }
}

export function saveStoredDetails(stored: StoredDetails): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
  } catch {
    // localStorage unavailable (private mode / quota) — non-fatal.
  }
}

export function clearStoredDetails(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
