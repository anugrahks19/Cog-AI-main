from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Assessment, PredictionResult
from app.schemas.assessments import (
    AssessmentCreateResponse,
    CognitiveSubmission,
    PredictionTriggerResponse,
    RiskResultResponse,
    SpeechUploadResponse,
)
from app.services.assessments import (
    create_assessment,
    fetch_assessment,
    persist_cognitive_submission,
    persist_speech_sample,
    run_prediction_pipeline,
)
from app.services.storage import store_audio_file
from app.services.transformers import assemble_assessment_result
from app.utils.auth import get_current_user

router = APIRouter()


@router.post("", response_model=AssessmentCreateResponse)
def start_assessment(
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> AssessmentCreateResponse:
    assessment = create_assessment(db, current_user_id)
    return AssessmentCreateResponse(assessment_id=assessment.id)


@router.get("/{assessment_id}", response_model=AssessmentCreateResponse)
def get_assessment_detail(
    assessment_id: str,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> AssessmentCreateResponse:
    assessment = fetch_assessment(db, assessment_id, current_user_id)
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")
    return AssessmentCreateResponse(assessment_id=assessment.id)


@router.post("/{assessment_id}/speech", response_model=SpeechUploadResponse)
async def upload_speech(
    assessment_id: str,
    file: UploadFile = File(...),
    task_id: str | None = None,
    language: str | None = None,
    duration_ms: int | None = None,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> SpeechUploadResponse:
    assessment = fetch_assessment(db, assessment_id, current_user_id)
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")

    storage_path = await store_audio_file(file, assessment.id, task_id or "speech")
    speech_sample = persist_speech_sample(
        db,
        assessment.id,
        task_id or file.filename,
        storage_path,
        duration_ms,
        language,
    )
    return SpeechUploadResponse(
        sample_id=speech_sample.id,
        path=storage_path,
        detected_language=speech_sample.detected_language,
        language_confidence=speech_sample.language_confidence,
        language_mismatch=bool(speech_sample.language_mismatch),
        transcript=speech_sample.transcript,
        translation=speech_sample.translation,
        warnings=speech_sample.processing_warnings or [],
        success=True,
    )


@router.post("/{assessment_id}/cognitive")
async def submit_cognitive_logs(
    assessment_id: str,
    submission: CognitiveSubmission,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> dict[str, bool]:
    assessment = fetch_assessment(db, assessment_id, current_user_id)
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")

    persist_cognitive_submission(db, assessment.id, submission)
    return {"success": True}


@router.post("/{assessment_id}/predict", response_model=PredictionTriggerResponse)
async def trigger_prediction(
    assessment_id: str,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> PredictionTriggerResponse:
    assessment = fetch_assessment(db, assessment_id, current_user_id)
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")

    run_prediction_pipeline(db, assessment)
    return PredictionTriggerResponse(success=True)


@router.get("/{assessment_id}/result", response_model=RiskResultResponse)
def get_prediction_result(
    assessment_id: str,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> RiskResultResponse:
    assessment = fetch_assessment(db, assessment_id, current_user_id)
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")

    prediction = db.query(PredictionResult).filter_by(assessment_id=assessment.id).first()
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not ready")

    result = assemble_assessment_result(assessment, prediction)
    return result
