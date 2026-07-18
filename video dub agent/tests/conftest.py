"""
tests/conftest.py
Shared fixtures for all test files. Pytest automatically finds this file
and makes every fixture here available to all tests, without needing imports.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path so "from api.main import app" works
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def mock_pipeline_result():
    """
    A fake, realistic-looking result that the real DubbingPipeline.process_video()
    would return after a successful run. We use this instead of running the
    real AI pipeline (which needs API keys, ffmpeg, and takes minutes).
    """
    return {
        "video_path": "downloads/videos/test_video.mp4",
        "audio_path": "downloads/audio/test_video.wav",
        "target_language": "Hindi",
        "speakers": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "duration": 5.0},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "duration": 5.0},
        ],
        "speaker_genders": {"SPEAKER_00": "male", "SPEAKER_01": "female"},
        "dubbed_segments": [
            {"speaker": "SPEAKER_00", "audio_path": "downloads/dubbed/s0.wav",
             "start": 0.0, "end": 5.0, "text": "Hello", "duration": 5.0},
        ],
        "output_path": "downloads/output/test_video_dubbed.mp4",
        "status": "🎉 Dubbing complete!"
    }


@pytest.fixture
def client(monkeypatch, tmp_path):
    """
    Creates a FastAPI TestClient with the REAL DubbingPipeline replaced by
    a fake (mock) one. This lets us test the API's request handling,
    validation, and job tracking WITHOUT ever loading Whisper, calling
    Groq, or touching real AI models.
    """
    # Patch DubbingPipeline BEFORE importing api.main, so that when
    # api.main does `from core.pipeline import DubbingPipeline`, anything
    # that tries to construct it gets our fake version instead.
    with patch("core.pipeline.DubbingPipeline") as MockPipelineClass:
        mock_instance = MagicMock()
        MockPipelineClass.return_value = mock_instance

        # Import app AFTER patching, so api.main picks up the patched class
        from api.main import app
        import api.main as main_module

        # Redirect upload/output folders to a temporary test directory
        # so tests never touch your real downloads/ folder
        main_module.UPLOAD_DIR = str(tmp_path / "uploads")
        main_module.OUTPUT_DIR = str(tmp_path / "output")
        os.makedirs(main_module.UPLOAD_DIR, exist_ok=True)
        os.makedirs(main_module.OUTPUT_DIR, exist_ok=True)

        from fastapi.testclient import TestClient
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def sample_video_bytes():
    """A tiny fake 'video file' — just bytes, not a real playable video.
    Good enough for testing upload validation logic (file extension, etc.)
    since we mock the actual processing anyway."""
    return b"fake video file content for testing"