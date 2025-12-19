from app.models import Assessment, PredictionResult
from app.schemas.assessments import CognitiveScores, FeatureImportance, RiskResultResponse


def assemble_assessment_result(assessment: Assessment, prediction: PredictionResult) -> RiskResultResponse:
    return RiskResultResponse(
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
