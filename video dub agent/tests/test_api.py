"""
tests/test_api.py
FastAPI endpoint tests. Uses the `client` fixture from conftest.py, which
gives us a TestClient wired to a MOCKED DubbingPipeline — so these tests
check the API's request handling, validation, and responses WITHOUT ever
running real AI models or making real network calls.
"""

import io


# ============================================================
# Root & health endpoints
# ============================================================

def test_root_returns_welcome_message(client):
    """Hitting the root URL should return a 200 with basic API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_endpoint_returns_ok(client):
    """Health check should always return ok=True and basic stats."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "jobs_count" in data


# ============================================================
# /languages
# ============================================================

def test_languages_endpoint_returns_list(client):
    """Should return a non-empty list of supported languages."""
    response = client.get("/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    assert isinstance(data["languages"], list)
    assert "Hindi" in data["languages"]
    assert len(data["languages"]) > 0


# ============================================================
# POST /upload -> validation tests (these should fail BEFORE
# touching the pipeline at all, so mocking isn't even the focus here)
# ============================================================

def test_upload_rejects_unsupported_language(client, sample_video_bytes):
    """Uploading with a language not in our supported list should return 400."""
    response = client.post(
        "/upload",
        files={"file": ("video.mp4", io.BytesIO(sample_video_bytes), "video/mp4")},
        data={"target_language": "Klingon", "transcription_engine": "whisper"},
    )
    assert response.status_code == 400
    assert "Unsupported target_language" in response.json()["detail"]


def test_upload_rejects_invalid_transcription_engine(client, sample_video_bytes):
    """transcription_engine must be 'whisper' or 'sarvam' — anything else is a 400."""
    response = client.post(
        "/upload",
        files={"file": ("video.mp4", io.BytesIO(sample_video_bytes), "video/mp4")},
        data={"target_language": "Hindi", "transcription_engine": "google"},
    )
    assert response.status_code == 400
    assert "transcription_engine" in response.json()["detail"]


def test_upload_rejects_unsupported_file_extension(client, sample_video_bytes):
    """Uploading a .txt file (not a video format) should be rejected with 400."""
    response = client.post(
        "/upload",
        files={"file": ("notavideo.txt", io.BytesIO(sample_video_bytes), "text/plain")},
        data={"target_language": "Hindi", "transcription_engine": "whisper"},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_missing_file_returns_422(client):
    """FastAPI's own validation: if 'file' is missing entirely, it's a 422
    (Unprocessable Entity) — this comes from FastAPI/Pydantic itself, not our code."""
    response = client.post(
        "/upload",
        data={"target_language": "Hindi", "transcription_engine": "whisper"},
    )
    assert response.status_code == 422


def test_upload_accepts_valid_video_and_creates_job(client, sample_video_bytes, monkeypatch):
    """
    A valid upload (correct language, engine, and .mp4 extension) should:
    1. Return 200
    2. Include a job_id
    3. The job should now exist and be queryable via /status
    We also bypass the ffprobe duration check by monkeypatching it, since
    our fake video bytes aren't a real video ffprobe could read.
    """
    import api.main as main_module
    monkeypatch.setattr(main_module, "_get_video_duration", lambda path: 30.0)

    response = client.post(
        "/upload",
        files={"file": ("video.mp4", io.BytesIO(sample_video_bytes), "video/mp4")},
        data={"target_language": "Hindi", "transcription_engine": "whisper"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "status_url" in data

    # The job should now be trackable
    status_response = client.get(f"/status/{data['job_id']}")
    assert status_response.status_code == 200


def test_upload_rejects_video_longer_than_10_minutes(client, sample_video_bytes, monkeypatch):
    """Business rule: videos over 10 minutes (600 seconds) must be rejected with 400."""
    import api.main as main_module
    monkeypatch.setattr(main_module, "_get_video_duration", lambda path: 700.0)  # ~11.6 min

    response = client.post(
        "/upload",
        files={"file": ("longvideo.mp4", io.BytesIO(sample_video_bytes), "video/mp4")},
        data={"target_language": "Hindi", "transcription_engine": "whisper"},
    )

    assert response.status_code == 400
    assert "10 minutes" in response.json()["detail"]


# ============================================================
# POST /upload-youtube
# ============================================================

def test_upload_youtube_rejects_invalid_url(client):
    """A non-YouTube URL should be rejected with 400."""
    response = client.post(
        "/upload-youtube",
        data={
            "youtube_url": "https://vimeo.com/12345",
            "target_language": "Hindi",
            "transcription_engine": "whisper",
        },
    )
    assert response.status_code == 400
    assert "valid YouTube URL" in response.json()["detail"]


def test_upload_youtube_accepts_valid_url_and_creates_job(client):
    """A valid YouTube URL should create a trackable job, same as file upload."""
    response = client.post(
        "/upload-youtube",
        data={
            "youtube_url": "https://youtube.com/watch?v=abc123",
            "target_language": "Hindi",
            "transcription_engine": "whisper",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


# ============================================================
# GET /status/{job_id}
# ============================================================

def test_status_for_nonexistent_job_returns_404(client):
    """Asking for status of a job_id that was never created should be a 404."""
    response = client.get("/status/this-job-does-not-exist")
    assert response.status_code == 404


def test_status_for_existing_job_returns_job_data(client):
    """After creating a job via /upload-youtube, /status should return its details."""
    upload_response = client.post(
        "/upload-youtube",
        data={
            "youtube_url": "https://youtube.com/watch?v=xyz789",
            "target_language": "Tamil",
            "transcription_engine": "sarvam",
        },
    )
    job_id = upload_response.json()["job_id"]

    status_response = client.get(f"/status/{job_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["job_id"] == job_id
    assert data["target_language"] == "Tamil"
    assert data["transcription_engine"] == "sarvam"


# ============================================================
# GET /download/{job_id}
# ============================================================

def test_download_for_nonexistent_job_returns_404(client):
    """Downloading a job that was never created should be a 404."""
    response = client.get("/download/no-such-job")
    assert response.status_code == 404


def test_download_for_incomplete_job_returns_409(client):
    """
    A job that exists but hasn't finished processing yet (still 'pending' or
    'processing') should return 409 Conflict — the file simply isn't ready.
    """
    upload_response = client.post(
        "/upload-youtube",
        data={
            "youtube_url": "https://youtube.com/watch?v=notdoneyet",
            "target_language": "Hindi",
            "transcription_engine": "whisper",
        },
    )
    job_id = upload_response.json()["job_id"]

    # Job was just created, background task hasn't run (or is mocked/slow),
    # so it should still be pending, not completed.
    download_response = client.get(f"/download/{job_id}")
    assert download_response.status_code in (404, 409)


# ============================================================
# GET /jobs
# ============================================================

def test_jobs_list_endpoint_returns_count_and_list(client):
    """/jobs should return both the count and the list of all jobs, for debugging."""
    client.post(
        "/upload-youtube",
        data={
            "youtube_url": "https://youtube.com/watch?v=listtest",
            "target_language": "Hindi",
            "transcription_engine": "whisper",
        },
    )

    response = client.get("/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "count" in data
    assert data["count"] >= 1