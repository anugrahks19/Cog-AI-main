from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = Field(
        default="sqlite:///./cogai.db",
        alias="DATABASE_URL",
        description="SQLAlchemy connection string.",
    )
    jwt_secret: str = Field(default="change-me", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60 * 24)
    storage_dir: Path = Field(default=Path("storage"), alias="STORAGE_DIR")
    allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    speech_languages: list[str] = Field(
        default_factory=lambda: [
            "en",
            "hi",
            "bn",
            "ta",
            "te",
            "kn",
            "ml",
            "mr",
            "gu",
            "pa",
        ],
        alias="SPEECH_LANGUAGES",
        description="Languages supported for speech recognition.",
    )
    whisper_model_size: str = Field(
        default="medium",
        alias="WHISPER_MODEL_SIZE",
        description="faster-whisper model size (e.g., tiny, base, small, medium).",
    )
    whisper_device: str = Field(
        default="auto",
        alias="WHISPER_DEVICE",
        description="Device to run Whisper on (cpu, cuda, auto).",
    )
    whisper_compute_type: str = Field(
        default="int8",
        alias="WHISPER_COMPUTE_TYPE",
        description="Compute type for Whisper inference (int8, int16, float16, float32).",
    )
    enable_translation: bool = Field(
        default=True,
        alias="ENABLE_TRANSLATION",
        description="Whether to translate transcripts to English for downstream processing.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
