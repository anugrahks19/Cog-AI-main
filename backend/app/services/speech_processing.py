from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langcodes import Language

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency failures handled at runtime
    WhisperModel = None  # type: ignore

try:
    from googletrans import Translator  # type: ignore
except Exception:  # pragma: no cover
    Translator = None  # type: ignore

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SpeechProcessingResult:
    transcript: Optional[str] = None
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    translation: Optional[str] = None
    language_mismatch: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Optional[str | float | bool | list[str]]]:
        return {
            "transcript": self.transcript,
            "detected_language": self.detected_language,
            "language_confidence": self.language_confidence,
            "translation": self.translation,
            "language_mismatch": self.language_mismatch,
            "warnings": self.warnings,
        }


@lru_cache
def _select_device() -> str:
    settings = get_settings()
    device = settings.whisper_device.lower().strip()
    if device != "auto":
        return device
    try:  # pragma: no cover - optional torch dependency
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch optional
        pass
    return "cpu"


class SpeechProcessingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Optional[WhisperModel] = None
        self._translator: Optional[Translator] = None
        self._models_dir = self.settings.storage_dir / "models"
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_language(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        try:
            return Language.get(value).to_tag().lower()
        except Exception:
            return value.lower()

    def _load_model(self) -> WhisperModel:
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed; cannot run speech processing")
        if self._model is None:
            device = _select_device()
            compute_type = self.settings.whisper_compute_type
            # Ensure compute_type is compatible with CPU
            if device == "cpu" and compute_type not in {"int8", "int8_float16", "int16", "float32"}:
                compute_type = "int8"
            logger.info("Loading Whisper model '%s' on %s with compute_type=%s", self.settings.whisper_model_size, device, compute_type)
            self._model = WhisperModel(
                self.settings.whisper_model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(self._models_dir),
            )
        return self._model

    def _load_translator(self) -> Optional[Translator]:
        if not self.settings.enable_translation:
            return None
        if Translator is None:
            logger.warning("googletrans package not available; translations disabled")
            return None
        if self._translator is None:
            self._translator = Translator()
        return self._translator

    def process_file(self, audio_path: str | Path, expected_language: Optional[str] = None) -> SpeechProcessingResult:
        path = Path(audio_path)
        result = SpeechProcessingResult()

        if not path.exists():
            result.warnings.append("Audio file missing for transcription")
            return result

        try:
            model = self._load_model()
        except Exception as exc:  # pragma: no cover - hardware/env dependent
            warning = f"Speech model unavailable: {exc}"
            logger.exception(warning)
            result.warnings.append(warning)
            return result

        normalized_expected = self._normalize_language(expected_language)

        try:
            segments, info = model.transcribe(
                str(path),
                beam_size=5,
                temperature=0,
                vad_filter=True,
                compression_ratio_threshold=2.4,
            )

            pieces: list[str] = []
            for segment in segments:
                text = (segment.text or "").strip()
                if text:
                    pieces.append(text)
            transcript = " ".join(pieces).strip()

            detected_language = self._normalize_language(getattr(info, "language", None))
            language_confidence = getattr(info, "language_probability", None)

            result.transcript = transcript or None
            result.detected_language = detected_language
            result.language_confidence = language_confidence

            if language_confidence is not None and language_confidence < 0.75:
                result.warnings.append(
                    "Spoken language confidence is low; transcription accuracy may be affected."
                )

            if not result.transcript:
                result.warnings.append("No speech content detected in the recording.")

            if normalized_expected and detected_language and normalized_expected != detected_language:
                result.language_mismatch = True
                result.warnings.append(
                    f"Detected language '{detected_language}' differs from selected '{normalized_expected}'."
                )

            if result.transcript and self.settings.enable_translation:
                translator = self._load_translator()
                if translator:
                    try:
                        src = detected_language or normalized_expected or "auto"
                        translation = translator.translate(result.transcript, src=src, dest="en")
                        result.translation = translation.text
                    except Exception as exc:  # pragma: no cover - network dependent
                        logger.warning("Translation failed: %s", exc)
                        result.warnings.append("Translation unavailable; using original transcript.")
        except Exception as exc:  # pragma: no cover - audio-specific failures
            logger.exception("Transcription failed: %s", exc)
            result.warnings.append(f"Transcription failed: {exc}")

        return result


speech_processor = SpeechProcessingService()

__all__ = ["speech_processor", "SpeechProcessingService", "SpeechProcessingResult"]
