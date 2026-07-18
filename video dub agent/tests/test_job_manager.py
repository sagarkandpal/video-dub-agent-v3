"""
tests/test_job_manager.py
Unit tests for the Job and JobManager classes in api/main.py.
No mocking needed here — these classes are pure Python logic with
no external calls (no network, no AI models, no file I/O).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.main import Job, JobManager, JobStatus


# ============================================================
# Job class tests
# ============================================================

def test_job_creation_has_correct_initial_values():
    """A newly created Job should start in 'pending' status with 0% progress."""
    job = Job(
        job_id="test-123",
        video_filename="my_video.mp4",
        target_language="Hindi",
        transcription_engine="whisper"
    )

    assert job.job_id == "test-123"
    assert job.video_filename == "my_video.mp4"
    assert job.target_language == "Hindi"
    assert job.transcription_engine == "whisper"
    assert job.status == JobStatus.PENDING
    assert job.progress_percent == 0
    assert job.output_path is None
    assert job.error is None


def test_job_to_dict_contains_all_expected_keys():
    """to_dict() should return a dictionary with every field the API needs to send back."""
    job = Job("id1", "video.mp4", "Tamil", "sarvam")
    result = job.to_dict()

    expected_keys = {
        "job_id", "status", "progress_percent", "progress_step",
        "progress_message", "video_filename", "target_language",
        "transcription_engine", "output_path", "error",
        "created_at", "updated_at"
    }
    assert expected_keys.issubset(result.keys())
    assert result["job_id"] == "id1"
    assert result["target_language"] == "Tamil"


# ============================================================
# JobManager tests
# ============================================================

def test_create_job_generates_unique_id():
    """Every job created should get its own unique job_id (a UUID)."""
    manager = JobManager()
    job1 = manager.create_job("video1.mp4", "Hindi", "whisper")
    job2 = manager.create_job("video2.mp4", "Hindi", "whisper")

    assert job1.job_id != job2.job_id


def test_get_job_returns_correct_job():
    """get_job() should return the exact job that was created with that ID."""
    manager = JobManager()
    job = manager.create_job("video.mp4", "Hindi", "whisper")

    fetched = manager.get_job(job.job_id)

    assert fetched is not None
    assert fetched.job_id == job.job_id
    assert fetched.video_filename == "video.mp4"


def test_get_job_returns_none_for_unknown_id():
    """Asking for a job_id that doesn't exist should return None, not crash."""
    manager = JobManager()
    result = manager.get_job("this-id-does-not-exist")

    assert result is None


def test_update_job_changes_progress_fields():
    """update_job() should change step, message, and percent for an existing job."""
    manager = JobManager()
    job = manager.create_job("video.mp4", "Hindi", "whisper")

    manager.update_job(job.job_id, "diarize_speakers", "Identifying speakers...", 25)

    updated = manager.get_job(job.job_id)
    assert updated.progress_step == "diarize_speakers"
    assert updated.progress_message == "Identifying speakers..."
    assert updated.progress_percent == 25


def test_update_job_caps_percent_at_99():
    """
    IMPORTANT BUSINESS RULE: update_job() should never let progress hit 100%
    on its own — only mark_completed() should set it to 100. This test
    documents and protects that rule.
    """
    manager = JobManager()
    job = manager.create_job("video.mp4", "Hindi", "whisper")

    manager.update_job(job.job_id, "sync_video", "Almost done", 100)

    updated = manager.get_job(job.job_id)
    assert updated.progress_percent == 99   # capped, not 100


def test_update_job_on_unknown_id_does_not_crash():
    """Calling update_job() with a bad job_id should silently do nothing, not raise an error."""
    manager = JobManager()
    # Should not raise any exception
    manager.update_job("fake-id", "step", "message", 50)


def test_mark_completed_sets_status_and_output_path():
    """mark_completed() should set status=completed, percent=100, and save the output path."""
    manager = JobManager()
    job = manager.create_job("video.mp4", "Hindi", "whisper")

    manager.mark_completed(job.job_id, "downloads/output/final_video.mp4")

    updated = manager.get_job(job.job_id)
    assert updated.status == JobStatus.COMPLETED
    assert updated.progress_percent == 100
    assert updated.output_path == "downloads/output/final_video.mp4"


def test_mark_failed_sets_status_and_error():
    """mark_failed() should set status=failed and store the error message."""
    manager = JobManager()
    job = manager.create_job("video.mp4", "Hindi", "whisper")

    manager.mark_failed(job.job_id, "ffmpeg not found")

    updated = manager.get_job(job.job_id)
    assert updated.status == JobStatus.FAILED
    assert updated.error == "ffmpeg not found"


def test_list_jobs_returns_all_created_jobs():
    """list_jobs() should return dictionaries for every job created so far."""
    manager = JobManager()
    manager.create_job("video1.mp4", "Hindi", "whisper")
    manager.create_job("video2.mp4", "Tamil", "sarvam")

    jobs = manager.list_jobs()

    assert len(jobs) == 2
    assert all(isinstance(j, dict) for j in jobs)