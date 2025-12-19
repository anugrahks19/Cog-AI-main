from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator

from app.models import Assessment, PredictionResult


class AssessmentCreateResponse(BaseModel):
    assessment_id: str


class InteractionLog(BaseModel):
    task_id: str
    task_type: str
    prompt: str
    response_time_ms: int = Field(..., ge=0)
    correct: Optional[bool] = None
    errors: Optional[int] = Field(default=None, ge=0)
    metadata: Optional[dict] = None


class CognitiveScores(BaseModel):
    memory_score: float = Field(..., ge=0)
    attention_score: float = Field(..., ge=0)
    language_score: float = Field(..., ge=0)
    executive_score: float = Field(..., ge=0)


class CognitiveSubmission(BaseModel):
    logs: List[InteractionLog]
    cognitive_scores: CognitiveScores
    clock_drawing: Optional[str] = None

    @validator("logs")
    def ensure_logs_not_empty(cls, value: List[InteractionLog]) -> List[InteractionLog]:
        if not value:
            raise ValueError("At least one log is required")
        return value


class SpeechUploadResponse(BaseModel):
    sample_id: str
    path: str
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    language_mismatch: bool = False
    transcript: Optional[str] = None
    translation: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    success: bool = True


class PredictionTriggerResponse(BaseModel):
    success: bool = True


class FeatureImportance(BaseModel):
    feature: str
    contribution: float
    direction: str


class AssessmentDetail(BaseModel):
    id: str
    user_id: str
    status: str
    language: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class RiskResultResponse(BaseModel):
    assessment_id: str
    risk_level: str
    probability: float
    feature_importances: List[FeatureImportance]
    sub_scores: CognitiveScores
    recommendations: List[str]
    generated_at: datetime

    class Config:
        orm_mode = True

    @classmethod
    def from_prediction(cls, assessment: Assessment, prediction: PredictionResult) -> "RiskResultResponse":
        return cls(
            assessment_id=assessment.id,
            risk_level=prediction.risk_level,
            probability=prediction.probability,
            feature_importances=[FeatureImportance(**fi) for fi in prediction.feature_importances],
            sub_scores=CognitiveScores(
                memory_score=assessment.memory_score or 0.0,
                attention_score=assessment.attention_score or 0.0,
                language_score=assessment.language_score or 0.0,
                executive_score=assessment.executive_score or 0.0,
            ),
            recommendations=prediction.recommendations,
            generated_at=prediction.generated_at,
        )
