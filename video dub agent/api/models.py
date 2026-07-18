"""
api/models.py
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str                    # queued | processing | completed | failed
    progress_step: str
    progress_message: str
    progress_percent: int
    video_filename: str
    target_language: str
    transcription_engine: str
    output_path: Optional[str] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]