"""
api/job_manager.py
In-memory job tracking for dubbing pipeline requests.
Each job has a unique ID, a status, progress info, and eventually an output path.
"""

import uuid
import time
from enum import Enum
from typing import Optional, Dict
from threading import Lock


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str, video_filename: str, target_language: str, transcription_engine: str):
        self.job_id = job_id
        self.video_filename = video_filename
        self.target_language = target_language
        self.transcription_engine = transcription_engine
        self.status = JobStatus.QUEUED
        self.progress_step = "queued"
        self.progress_message = "Waiting to start..."
        self.progress_percent = 0
        self.output_path: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = time.time()

    def update(self, step: str, message: str, percent: int):
        self.progress_step = step
        self.progress_message = message
        self.progress_percent = percent
        self.updated_at = time.time()

    def mark_processing(self):
        self.status = JobStatus.PROCESSING
        self.updated_at = time.time()

    def mark_completed(self, output_path: str):
        self.status = JobStatus.COMPLETED
        self.output_path = output_path
        self.progress_percent = 100
        self.progress_message = "Dubbing complete!"
        self.updated_at = time.time()

    def mark_failed(self, error: str):
        self.status = JobStatus.FAILED
        self.error = error
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress_step": self.progress_step,
            "progress_message": self.progress_message,
            "progress_percent": self.progress_percent,
            "video_filename": self.video_filename,
            "target_language": self.target_language,
            "transcription_engine": self.transcription_engine,
            "output_path": self.output_path,
            "error": self.error,
        }


class JobManager:
    """Thread-safe in-memory store for all jobs"""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = Lock()

    def create_job(self, video_filename: str, target_language: str, transcription_engine: str) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id, video_filename, target_language, transcription_engine)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list:
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]


# Single shared instance used across the whole API
job_manager = JobManager()