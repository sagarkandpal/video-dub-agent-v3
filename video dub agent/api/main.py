"""
api/main.py
FastAPI backend for the video dubbing pipeline.

Endpoints:
- POST /upload         -> accepts a video, starts dubbing in the background, returns job_id
- GET  /status/{job_id} -> returns current progress for a job
- GET  /download/{job_id} -> downloads the finished dubbed video
"""

import os
import re
import shutil
import sys
import threading
import uuid
from datetime import datetime
from typing import Optional

# Force UTF-8 output so emoji in print() don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.pipeline import DubbingPipeline

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="Video Dubbing API",
    version="1.0.0",
    description="Multi-speaker video dubbing with AI"
)

# Allow local frontend / testing tools to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = "downloads/uploads"
OUTPUT_DIR = "downloads/output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# JOB MANAGEMENT
# ============================================================

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job:
    def __init__(self, job_id: str, video_filename: str, target_language: str, transcription_engine: str):
        self.job_id = job_id
        self.video_filename = video_filename
        self.target_language = target_language
        self.transcription_engine = transcription_engine
        self.status = JobStatus.PENDING
        self.progress_percent = 0
        self.progress_step = "initialized"
        self.progress_message = "Job created"
        self.output_path = None
        self.error = None
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def to_dict(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "progress_step": self.progress_step,
            "progress_message": self.progress_message,
            "video_filename": self.video_filename,
            "target_language": self.target_language,
            "transcription_engine": self.transcription_engine,
            "output_path": self.output_path,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

class JobManager:
    def __init__(self):
        self.jobs = {}
        self._lock = threading.Lock()
    
    def create_job(self, video_filename: str, target_language: str, transcription_engine: str) -> Job:
        job_id = str(uuid.uuid4())
        with self._lock:
            job = Job(job_id, video_filename, target_language, transcription_engine)
            self.jobs[job_id] = job
            return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, step: str, message: str, percent: int):
        job = self.get_job(job_id)
        if job:
            with self._lock:
                job.progress_step = step
                job.progress_message = message
                job.progress_percent = min(percent, 99)
                job.updated_at = datetime.now().isoformat()
    
    def mark_completed(self, job_id: str, output_path: str):
        job = self.get_job(job_id)
        if job:
            with self._lock:
                job.status = JobStatus.COMPLETED
                job.progress_percent = 100
                job.progress_step = "complete"
                job.progress_message = "✅ Dubbing complete!"
                job.output_path = output_path
                job.updated_at = datetime.now().isoformat()
    
    def mark_failed(self, job_id: str, error: str):
        job = self.get_job(job_id)
        if job:
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = error
                job.progress_message = f"❌ Failed: {error}"
                job.updated_at = datetime.now().isoformat()
    
    def list_jobs(self):
        return [job.to_dict() for job in self.jobs.values()]

job_manager = JobManager()

# ============================================================
# LANGUAGES
# ============================================================

ALL_LANGUAGES = [
    "Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", 
    "Punjabi", "Kannada", "Malayalam", "Odia", "Spanish", "French", 
    "German", "Japanese", "Chinese", "Arabic", "Portuguese", "Russian"
]

# ============================================================
# PIPELINE LAZY LOADING
# ============================================================

_pipeline = None
_pipeline_lock = threading.Lock()
_job_run_lock = threading.Lock()

def get_pipeline():
    """Load the heavy dubbing pipeline lazily so the API starts immediately."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                print("[INFO] Initializing dubbing pipeline (this may take a moment)...")
                _pipeline = DubbingPipeline()
                print("[INFO] Pipeline ready")
    return _pipeline

# ============================================================
# HELPERS
# ============================================================

def _safe_upload_name(filename: str) -> str:
    """Create a unique, filesystem-safe upload filename."""
    original_name = os.path.basename(filename or "uploaded_video.mp4")
    stem, ext = os.path.splitext(original_name)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "video"
    safe_ext = ext.lower() if ext else ".mp4"
    return f"{uuid.uuid4().hex[:8]}_{safe_stem}{safe_ext}"

def _make_job_output_copy(job_id: str, output_path: str) -> str:
    """Keep each completed job download stable."""
    final_path = os.path.join(OUTPUT_DIR, f"{job_id}_{os.path.basename(output_path)}")
    if os.path.abspath(output_path) != os.path.abspath(final_path):
        shutil.copy2(output_path, final_path)
    return final_path

def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


def _is_youtube_url(url: str) -> bool:
    return bool(url and ("youtube.com" in url or "youtu.be" in url))

# ============================================================
# BACKGROUND PROCESSING
# ============================================================

MAX_VIDEO_DURATION_SECONDS = 10 * 60   # 10 minutes

def run_dubbing_job(job_id: str, video_path: str, target_language: str, transcription_engine: str):
    """Runs in a background thread. Updates job_manager as the pipeline progresses."""
    job = job_manager.get_job(job_id)
    if not job:
        return

    def progress_callback(step: str, message: str, percent: int):
        job_manager.update_job(job_id, step, message, percent)

    try:
        # Update status
        job_manager.update_job(job_id, "queued", "Preparing dubbing engine...", 5)
        
        with _job_run_lock:
            pipeline = get_pipeline()
            result = pipeline.process_video(
                video_path,
                target_language=target_language,
                transcription_engine=transcription_engine,
                progress_callback=progress_callback
            )
        
        output_path = result.get("output_path")
        if output_path and os.path.exists(output_path):
            final_output_path = _make_job_output_copy(job_id, output_path)
            job_manager.mark_completed(job_id, final_output_path)
            print(f"✅ Job {job_id} completed successfully!")
        else:
            job_manager.mark_failed(job_id, "Pipeline finished but no output video was produced.")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Job {job_id} failed: {error_msg}")
        job_manager.mark_failed(job_id, error_msg)

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "🎬 Video Dubbing API is running",
        "ui": "/ui",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload": "POST /upload",
            "upload_youtube": "POST /upload-youtube",
            "status": "GET /status/{job_id}",
            "download": "GET /download/{job_id}",
            "languages": "GET /languages",
            "jobs": "GET /jobs"
        }
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "pipeline_loaded": _pipeline is not None,
        "jobs_count": len(job_manager.jobs)
    }

@app.get("/languages")
def get_languages():
    """List all supported target languages"""
    return {"languages": ALL_LANGUAGES}

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form("Hindi"),
    transcription_engine: str = Form("whisper")
):
    """Upload a video and start the dubbing pipeline in the background"""
    
    if target_language not in ALL_LANGUAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported target_language. Choose from: {ALL_LANGUAGES}"
        )
    
    if transcription_engine not in ("whisper", "sarvam"):
        raise HTTPException(
            status_code=400, 
            detail="transcription_engine must be 'whisper' or 'sarvam'"
        )
    
    # Validate file extension
    allowed_ext = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_ext}"
        )
    
    # Save uploaded file
    safe_name = _safe_upload_name(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Check duration limit (max 10 minutes)
    duration = _get_video_duration(saved_path)
    if duration and duration > MAX_VIDEO_DURATION_SECONDS:
        os.remove(saved_path)
        raise HTTPException(
            status_code=400,
            detail=f"Video is {duration/60:.1f} minutes long. Maximum allowed is 10 minutes."
        )
    
    # Create job
    job = job_manager.create_job(
        file.filename, 
        target_language, 
        transcription_engine
    )
    
    # Start background processing
    background_tasks.add_task(
        run_dubbing_job, 
        job.job_id, 
        saved_path, 
        target_language, 
        transcription_engine
    )
    
    return {
        "job_id": job.job_id,
        "message": "Upload received. Dubbing started in background.",
        "status_url": f"/status/{job.job_id}"
    }

@app.post("/upload-youtube")
async def upload_youtube(
    background_tasks: BackgroundTasks,
    youtube_url: str = Form(...),
    target_language: str = Form("Hindi"),
    transcription_engine: str = Form("whisper")
):
    """Submit a YouTube URL and start the dubbing pipeline in the background"""
    
    if target_language not in ALL_LANGUAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported target_language. Choose from: {ALL_LANGUAGES}"
        )
    
    if transcription_engine not in ("whisper", "sarvam"):
        raise HTTPException(
            status_code=400, 
            detail="transcription_engine must be 'whisper' or 'sarvam'"
        )
    
    if not _is_youtube_url(youtube_url):
        raise HTTPException(
            status_code=400, 
            detail="Please provide a valid YouTube URL"
        )
    
    # Create job
    job = job_manager.create_job(
        youtube_url, 
        target_language, 
        transcription_engine
    )
    
    # Start background processing
    background_tasks.add_task(
        run_dubbing_job, 
        job.job_id, 
        youtube_url, 
        target_language, 
        transcription_engine
    )
    
    return {
        "job_id": job.job_id,
        "message": "YouTube link received. Dubbing started in background.",
        "status_url": f"/status/{job.job_id}"
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Check the progress of a dubbing job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.to_dict()

@app.get("/jobs")
def list_jobs():
    """List all jobs (for debugging)"""
    return {
        "jobs": job_manager.list_jobs(),
        "count": len(job_manager.jobs)
    }

@app.get("/download/{job_id}")
def download_video(job_id: str):
    """Download the finished dubbed video"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409, 
            detail=f"Job is not completed yet (status: {job.status})"
        )
    
    if not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(
            status_code=404, 
            detail="Output file not found on disk"
        )
    
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=os.path.basename(job.output_path)
    )

# ============================================================
# FRONTEND SERVING
# ============================================================

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/ui", include_in_schema=False)
def serve_ui():
    """Redirect /ui to the frontend index.html"""
    return RedirectResponse(url="/frontend/index.html")