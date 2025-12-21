const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export interface RegisterUserPayload {
  name: string;
  age: number;
  gender: number;
  education: number;
  // Medical
  family_history?: number;
  diabetes?: number;
  hypertension?: number;
  depression?: number;
  head_injury?: number;
  // Lifestyle
  sleep_quality?: number;
  physical_activity?: number;
  smoking?: number;

  language: string;
  consent: boolean;
}

export interface RegisterUserResponse {
  user: {
    id: string;
    name: string;
    age: number;
    gender: number;
    education: number;
    family_history?: number;
    diabetes?: number;
    hypertension?: number;
    depression?: number;
    head_injury?: number;
    sleep_quality?: number;
    physical_activity?: number;
    smoking?: number;

    language: string;
    consent: boolean;
  };
  accessToken?: string;
}

export interface StartAssessmentResponse {
  assessmentId: string;
}

export interface CognitiveSubmissionPayload {
  assessmentId: string;
  logs: InteractionLog[];
  cognitiveScores: CognitiveScores;
  clockDrawing?: string;
}

export interface InteractionLog {
  taskId: string;
  taskType: "speech" | "cognitive";
  prompt: string;
  responseTimeMs: number;
  correct: boolean | null;
  errors?: number;
  metadata?: Record<string, unknown>;
}

export interface CognitiveScores {
  memoryScore: number;
  attentionScore: number;
  languageScore: number;
  executiveScore: number;
}

export interface AssessmentResult {
  assessmentId: string;
  riskLevel: "Low" | "Medium" | "High";
  probability: number;
  featureImportances: Array<{
    feature: string;
    contribution: number;
    direction: "positive" | "negative";
  }>;
  subScores: CognitiveScores;
  recommendations: string[];
  generatedAt: string;
}

export interface SpeechUploadResponse {
  sampleId: string;
  path: string;
  detectedLanguage?: string | null;
  languageConfidence?: number | null;
  languageMismatch?: boolean;
  transcript?: string | null;
  translation?: string | null;
  warnings: string[];
  success: boolean;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Request failed");
  }
  return response.json() as Promise<T>;
}

export async function registerUser(payload: RegisterUserPayload) {
  const response = await fetch(`${API_BASE_URL}/api/users`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  // Read raw response so we can normalize snake_case keys from backend
  const raw = await response.json().catch(() => null as unknown);

  if (!response.ok) {
    const msg = typeof raw === "string" ? raw : (raw as any)?.detail || "Request failed";
    throw new Error(msg);
  }

  const normalized: RegisterUserResponse = {
    user: (raw as any).user,
    // Backend returns access_token; frontend expects accessToken
    accessToken: (raw as any).accessToken ?? (raw as any).access_token,
  };
  return normalized;
}

export async function startAssessment(accessToken?: string) {
  const response = await fetch(`${API_BASE_URL}/api/assessments`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    },
    body: JSON.stringify({}),
  });
  const raw = await response.json().catch(() => null as unknown);
  if (!response.ok) {
    const msg = typeof raw === "string" ? raw : (raw as any)?.detail || "Request failed";
    throw new Error(msg);
  }
  const normalized: StartAssessmentResponse = {
    assessmentId: (raw as any).assessmentId ?? (raw as any).assessment_id,
  };
  return normalized;
}

export async function uploadSpeechSample(options: {
  assessmentId: string;
  taskId: string;
  blob: Blob;
  language: string;
  accessToken?: string;
  durationMs?: number;
}): Promise<SpeechUploadResponse> {
  const formData = new FormData();
  formData.append("file", options.blob, `${options.taskId}.webm`);
  // Backend expects snake_case fields
  formData.append("task_id", options.taskId);
  formData.append("language", options.language);
  // attach duration if known (in ms)
  // Note: caller already knows duration; keep compatibility by checking if provided via query
  // We only add if present on options as any
  const maybeDuration = (options as any).durationMs as number | undefined;
  if (typeof maybeDuration === "number") {
    formData.append("duration_ms", String(maybeDuration));
  }

  const response = await fetch(
    `${API_BASE_URL}/api/assessments/${options.assessmentId}/speech`,
    {
      method: "POST",
      headers: {
        ...(options.accessToken ? { Authorization: `Bearer ${options.accessToken}` } : {}),
      },
      body: formData,
    },
  );

  return handleResponse<SpeechUploadResponse>(response);
}

export async function submitCognitiveData(
  payload: CognitiveSubmissionPayload,
  accessToken?: string,
) {
  // Transform camelCase -> snake_case to match backend Pydantic models
  const logs_snake = payload.logs.map((l) => ({
    task_id: l.taskId,
    task_type: l.taskType,
    prompt: l.prompt,
    response_time_ms: Math.round(l.responseTimeMs ?? 0),
    correct: l.correct,
    errors: l.errors,
    metadata: l.metadata,
  }));

  const scores_snake = {
    memory_score: payload.cognitiveScores.memoryScore,
    attention_score: payload.cognitiveScores.attentionScore,
    language_score: payload.cognitiveScores.languageScore,
    executive_score: payload.cognitiveScores.executiveScore,
  };

  const response = await fetch(
    `${API_BASE_URL}/api/assessments/${payload.assessmentId}/cognitive`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
      },
      body: JSON.stringify({
        logs: logs_snake,
        cognitive_scores: scores_snake,
        clock_drawing: payload.clockDrawing,
      }),
    },
  );

  return handleResponse<{ success: boolean }>(response);
}

export async function requestRiskPrediction(
  assessmentId: string,
  accessToken?: string,
) {
  const response = await fetch(
    `${API_BASE_URL}/api/assessments/${assessmentId}/predict`,
    {
      method: "POST",
      headers: {
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
      },
    },
  );

  return handleResponse<{ success: boolean }>(response);
}

export async function fetchAssessmentResult(
  assessmentId: string,
  accessToken?: string,
) {
  const response = await fetch(
    `${API_BASE_URL}/api/assessments/${assessmentId}/result`,
    {
      headers: {
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
      },
    },
  );

  const raw = await response.json().catch(() => null as unknown);
  if (!response.ok) {
    const msg = typeof raw === "string" ? raw : (raw as any)?.detail || "Request failed";
    throw new Error(msg);
  }

  const rawRisk = (raw as any).riskLevel ?? (raw as any).risk_level;
  const normalizedRisk = rawRisk === "Moderate" ? "Medium" : rawRisk;
  const normalized: AssessmentResult = {
    assessmentId: (raw as any).assessmentId ?? (raw as any).assessment_id,
    riskLevel: normalizedRisk,
    probability: (raw as any).probability,
    featureImportances:
      (raw as any).featureImportances ?? (raw as any).feature_importances ?? [],
    subScores: (raw as any).subScores ?? (raw as any).sub_scores ?? {
      memoryScore: 0,
      attentionScore: 0,
      languageScore: 0,
      executiveScore: 0,
    },
    recommendations: (raw as any).recommendations ?? [],
    generatedAt: (raw as any).generatedAt ?? (raw as any).generated_at,
  } as AssessmentResult;

  // Ensure subScores keys are camelCase
  if ((raw as any)?.sub_scores) {
    const ss = (raw as any).sub_scores;
    normalized.subScores = {
      memoryScore: ss.memory_score ?? 0,
      attentionScore: ss.attention_score ?? 0,
      languageScore: ss.language_score ?? 0,
      executiveScore: ss.executive_score ?? 0,
    };
  }

  return normalized;
}
