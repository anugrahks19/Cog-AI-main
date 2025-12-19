from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from fastapi import UploadFile

from app.core.config import get_settings

settings = get_settings()


def get_storage_path(assessment_id: str, task_id: str) -> Path:
    directory = settings.storage_dir / assessment_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{task_id}.webm"


async def store_audio_file(file: UploadFile, assessment_id: str, task_id: str) -> str:
    destination = get_storage_path(assessment_id, task_id)

    with destination.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    return str(destination)
