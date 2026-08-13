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
  alzheimers_detected: boolean;
  detection_percentage: number;
  predicted_class: string;
  class_index: number;
  probabilities: {
    Nondemented: number;
    Converted: number;
    Demented: number;
  };
  rule_applied?: boolean;
  rule_usage_percentage?: number;
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
// AI-assisted MMSE item evaluation (separate service; /predict contract unchanged)
// ---------------------------------------------------------------------------
export interface MmseEvaluateRequest {
  section: string;
  item_key: string;
  question: string;
  response: string;
  expected: string;
}

export interface MmseEvaluateResult {
  correct: boolean;
  score: number;
  confidence: number;
  reason: string;
}

export const evaluateMmseItem = async (
  data: MmseEvaluateRequest
): Promise<MmseEvaluateResult> => {
  const response = await axios.post<MmseEvaluateResult>(
    `${API_BASE_URL}/mmse/evaluate`,
    data
  );
  return response.data;
};

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