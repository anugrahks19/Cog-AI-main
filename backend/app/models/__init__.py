from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    name: Mapped[str] = mapped_column(String(120))
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[int] = mapped_column(Integer, default=0) # 0:Male, 1:Female
    education: Mapped[int] = mapped_column(Integer, default=0) # 0:None..3:Higher
    # Medical
    family_history: Mapped[int] = mapped_column(Integer, default=0)
    diabetes: Mapped[int] = mapped_column(Integer, default=0)
    hypertension: Mapped[int] = mapped_column(Integer, default=0)
    depression: Mapped[int] = mapped_column(Integer, default=0)
    head_injury: Mapped[int] = mapped_column(Integer, default=0)
    # Lifestyle
    sleep_quality: Mapped[int] = mapped_column(Integer, default=7)
    physical_activity: Mapped[int] = mapped_column(Integer, default=5)
    smoking: Mapped[int] = mapped_column(Integer, default=0)
    alcohol_consumption: Mapped[float] = mapped_column(Float, default=0.0)
    diet_quality: Mapped[float] = mapped_column(Float, default=5.0)
    height: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # cm
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # kg
    
    language: Mapped[str] = mapped_column(String(16))
    consent: Mapped[bool] = mapped_column(Boolean, default=False)

    assessments: Mapped[list["Assessment"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    language: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    memory_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    attention_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    language_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    executive_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clock_drawing: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    risk_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    user: Mapped[User] = relationship(back_populates="assessments")
    speech_samples: Mapped[list["SpeechSample"]] = relationship(
        back_populates="assessment", cascade="all, delete-orphan"
    )
    cognitive_logs: Mapped[list["CognitiveLog"]] = relationship(
        back_populates="assessment", cascade="all, delete-orphan"
    )
    prediction: Mapped[Optional["PredictionResult"]] = relationship(
        back_populates="assessment", cascade="all, delete-orphan", uselist=False
    )


class SpeechSample(Base):
    __tablename__ = "speech_samples"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    assessment_id: Mapped[str] = mapped_column(String(36), ForeignKey("assessments.id", ondelete="CASCADE"))
    task_id: Mapped[str] = mapped_column(String(64))
    file_path: Mapped[str] = mapped_column(String(512))
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    detected_language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    language_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    translation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language_mismatch: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    processing_warnings: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    assessment: Mapped[Assessment] = relationship(back_populates="speech_samples")


class CognitiveLog(Base):
    __tablename__ = "cognitive_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    assessment_id: Mapped[str] = mapped_column(String(36), ForeignKey("assessments.id", ondelete="CASCADE"))
    task_id: Mapped[str] = mapped_column(String(64))
    task_type: Mapped[str] = mapped_column(String(32))
    prompt: Mapped[str] = mapped_column(Text)
    response_time_ms: Mapped[int] = mapped_column(Integer)
    correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    errors: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    assessment: Mapped[Assessment] = relationship(back_populates="cognitive_logs")


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    assessment_id: Mapped[str] = mapped_column(String(36), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True)
    risk_level: Mapped[str] = mapped_column(String(16))
    probability: Mapped[float] = mapped_column(Float)
    feature_importances: Mapped[list[dict]] = mapped_column(JSON, default=list)
    recommendations: Mapped[list[str]] = mapped_column(JSON, default=list)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    assessment: Mapped[Assessment] = relationship(back_populates="prediction")


__all__ = [
    "Base",
    "User",
    "Assessment",
    "SpeechSample",
    "CognitiveLog",
    "PredictionResult",
]
