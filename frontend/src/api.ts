import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

export interface PatientInput {
  visit: number;
  mr_delay: number;
  sex: number; // 1 = Male, 0 = Female
  hand: number; // 1 = Right, 0 = Left
  age: number;
  education_years: number;
  ses: number;
  mmse: number;
  cdr: number;
  etiv: number;
  nwbv: number;
  asf: number;
}

export interface PredictionResponse {
  predicted_class: string;
  class_index: number;
  probabilities: {
    Nondemented: number;
    Converted: number;
    Demented: number;
  };
}

export const predictAlzheimers = async (data: PatientInput): Promise<PredictionResponse> => {
  const response = await axios.post<PredictionResponse>(`${API_BASE_URL}/predict`, data);
  return response.data;
};

