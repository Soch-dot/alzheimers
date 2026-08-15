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
  /** Display unit appended to the range help text (e.g. "50–100 years"). */
  unit?: string;
  options?: { value: number | string; label: string }[];
}

/**
 * CONFIGURABLE: required-field metadata for the Assessment Details step.
 * `required` marks which fields are mandatory before the patient can continue
 * to the MMSE. This array is the SINGLE SOURCE OF TRUTH for the allowed ranges:
 * the form renders them as help text and validates against them — no other
 * component hardcodes these bounds.
 */
export const DETAILS_FIELDS: DetailFieldConfig[] = [
  { name: 'age', label: 'Age', required: true, type: 'number', min: 50, max: 100, unit: 'years' },
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
    unit: 'years',
  },
  { name: 'ses', label: 'SES', required: true, type: 'number', min: 1, max: 5 },
];

/**
 * Human-readable allowed range for a numeric field, derived from its config
 * (`min`/`max`/`unit`), e.g. "50–100 years", "0–25 years", "1–5". Returns null
 * for fields without a numeric range. Used for both the field help text and the
 * out-of-range error message, so the config stays the single source of truth.
 */
export function fieldRangeLabel(field: DetailFieldConfig): string | null {
  if (field.type !== 'number' || field.min === undefined || field.max === undefined) {
    return null;
  }
  return `${field.min}–${field.max}${field.unit ? ' ' + field.unit : ''}`;
}

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

/** Whether a single field is valid per its DETAILS_FIELDS config. */
export function fieldValid(name: keyof DetailsDraft, draft: DetailsDraft): boolean {
  const cfg = DETAILS_FIELDS.find((f) => f.name === name);
  if (!cfg) return false;
  const raw = draft[name];
  if (raw === '' || raw === null || raw === undefined) return !cfg.required;
  if (cfg.type === 'select') {
    return typeof raw === 'number' && (cfg.options?.some((o) => o.value === raw) ?? false);
  }
  const n = parseField(name, draft);
  if (n === null) return false;
  if (cfg.min !== undefined && n < cfg.min) return false;
  if (cfg.max !== undefined && n > cfg.max) return false;
  return true;
}

/** User-facing validation message for a field, or null when it is valid/empty. */
export function fieldError(name: keyof DetailsDraft, draft: DetailsDraft): string | null {
  const cfg = DETAILS_FIELDS.find((f) => f.name === name);
  if (!cfg) return null;
  const raw = draft[name];
  if (raw === '' || raw === null || raw === undefined) return null;
  if (fieldValid(name, draft)) return null;
  if (cfg.type === 'select') return 'Invalid selection';
  const range = fieldRangeLabel(cfg);
  if (range) return `Enter ${range}`;
  return 'Enter a number';
}

/** Whether every required field is present and within the configured ranges. */
export function detailsValid(draft: DetailsDraft): boolean {
  return DETAILS_FIELDS.every((field) => fieldValid(field.name, draft));
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
