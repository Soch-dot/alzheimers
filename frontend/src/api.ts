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