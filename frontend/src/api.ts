import axios from 'axios';

// Use environment variable for API URL, fallback to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export interface PatientInput {
  age: number;
  sex: number; // 1 = Male, 0 = Female
  education_years: number;
  mmse: number;
  ses: number;
}

export interface PredictionResponse {
  model_version: string;
  screening_target: string;
  screening_probability: number;
  screening_threshold: number;
  screening_result: 'positive' | 'negative';
  predicted_class: string;
  features: {
    age: number;
    sex: number;
    education_years: number;
    mmse: number;
    ses: number;
  };
  interpretation: {
    label: string;
    not_a_diagnosis: boolean;
  };
  limitations: {
    clinical_validation: boolean;
    prospective_conversion_prediction: boolean;
  };
}

export const predictAlzheimers = async (data: PatientInput): Promise<PredictionResponse> => {
  // DEBUG: Log exact payload being sent
  console.log('=== FRONTEND PAYLOAD ===');
  console.log('Sending to API:', JSON.stringify(data, null, 2));
  console.log('Payload keys:', Object.keys(data));
  console.log('Payload order: age, sex, education_years, mmse, ses');
  console.log('========================');
  
  const response = await axios.post<PredictionResponse>(`${API_BASE_URL}/predict`, data);
  return response.data;
};

// ---------------------------------------------------------------------------
// AI-assisted MMSE batch evaluation (separate service; /predict contract unchanged)
// The frontend sends ALL collected responses in ONE request when the examiner
// clicks "Assess MMSE with AI". Keys are "<section>.<item_key>".
// ---------------------------------------------------------------------------
export interface MmseBatchItem {
  question: string;
  response: string;
  expected: string;
}

export interface MmseBatchItemResult {
  correct: boolean;
  score: number;
  confidence: number;
  reason: string;
}

export interface MmseBatchResponse {
  items: Record<string, MmseBatchItemResult>;
  errors?: Record<string, string>;
}

export const evaluateMmseBatch = async (
  items: Record<string, MmseBatchItem>,
  timeoutMs = 180000
): Promise<MmseBatchResponse> => {
  const response = await axios.post<MmseBatchResponse>(
    `${API_BASE_URL}/mmse/evaluate`,
    { items },
    { timeout: timeoutMs }
  );
  return response.data;
};

export function isTimeoutError(err: unknown): boolean {
  return axios.isAxiosError(err) && err.code === 'ECONNABORTED';
}

// ---------------------------------------------------------------------------
// MMSE Question 11 — photo-based figure copying assessment.
// Sends the patient drawing as a JSON data URL to the dedicated vision endpoint
// (independent of the text-MMSE batch). The backend rejects blank drawings with
// HTTP 400 before any model call; 502/503/504 map to invalid/unavailable/timeout.
// ---------------------------------------------------------------------------
export interface CopyingEvaluateResponse {
  correct: boolean;
  score: number;
  confidence: number;
  reason: string;
  review_required: boolean;
}

/**
 * Frontend timeout (ms) for the Q11 vision endpoint. This is the Q11-only
 * budget — the text-MMSE batch keeps its own separate timeout (see
 * `evaluateMmseBatch`). Mirrors the backend `VISION_TIMEOUT` default.
 */
export const VISION_CLIENT_TIMEOUT = 120000;

export const evaluateCopyingImage = async (
  imageDataUrl: string,
  timeoutMs = VISION_CLIENT_TIMEOUT
): Promise<CopyingEvaluateResponse> => {
  const response = await axios.post<CopyingEvaluateResponse>(
    `${API_BASE_URL}/mmse/copying/evaluate`,
    { image: imageDataUrl },
    { timeout: timeoutMs }
  );
  return response.data;
};

export type CopyingErrorKind =
  | 'blank'
  | 'timeout'
  | 'unavailable'
  | 'invalid'
  | 'upload';

/**
 * Classify a Q11 vision-evaluation failure into a user-facing error kind.
 * - 400 with "No drawing detected" detail -> blank drawing
 * - other 400 -> upload problem (bad/oversized/undecodable image)
 * - 504 or client timeout -> timeout
 * - 502 -> invalid model result
 * - 503 / anything else -> provider unavailable
 */
export function classifyCopyingError(err: unknown): {
  kind: CopyingErrorKind;
  detail: string | null;
} {
  if (axios.isAxiosError(err)) {
    const status = err.response?.status;
    const detail = extractApiError(err) || null;
    if (status === 400) {
      const isBlank = typeof detail === 'string' && detail.includes('No drawing detected');
      return { kind: isBlank ? 'blank' : 'upload', detail };
    }
    if (status === 504) return { kind: 'timeout', detail };
    if (status === 502) return { kind: 'invalid', detail };
    if (status === 503) return { kind: 'unavailable', detail };
    if (isTimeoutError(err)) return { kind: 'timeout', detail };
    return { kind: 'unavailable', detail };
  }
  return { kind: 'unavailable', detail: null };
}

export function extractApiError(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail.map((d: any) => d?.msg ?? JSON.stringify(d)).join('; ');
    }
  }
  return err instanceof Error ? err.message : 'Unknown error';
}