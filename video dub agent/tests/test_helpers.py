"""
tests/test_helpers.py
Unit tests for small pure utility/helper functions used across the project.
These have no external dependencies (no network, no AI, no file I/O that
matters for the logic itself), so they're tested directly without mocking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.main import _safe_upload_name, _is_youtube_url
from downloaders.yt_downloader import YouTubeDownloader
from core.language_config import is_indian_language, get_sarvam_code, ALL_LANGUAGES


# ============================================================
# api/main.py -> _safe_upload_name()
# ============================================================

def test_safe_upload_name_keeps_normal_filename_readable():
    """A normal, simple filename should stay mostly recognizable after sanitizing."""
    result = _safe_upload_name("myvideo.mp4")
    assert result.endswith(".mp4")
    assert "myvideo" in result


def test_safe_upload_name_removes_special_characters():
    """Filenames with spaces, emojis, or special symbols should not crash the sanitizer,
    and the unsafe characters should be stripped/replaced."""
    result = _safe_upload_name("my video 😡 #podcast!.mp4")
    # No spaces, no emoji, no special punctuation should survive
    assert " " not in result
    assert "😡" not in result
    assert "#" not in result
    assert "!" not in result
    assert result.endswith(".mp4")


def test_safe_upload_name_handles_missing_filename():
    """If filename is empty/None, it should fall back to a default name instead of crashing."""
    result = _safe_upload_name("")
    assert result.endswith(".mp4")


def test_safe_upload_name_produces_unique_names_for_same_input():
    """Calling this twice with the SAME filename should give two DIFFERENT results,
    because it adds a random prefix — this prevents one upload overwriting another."""
    result1 = _safe_upload_name("video.mp4")
    result2 = _safe_upload_name("video.mp4")
    assert result1 != result2


# ============================================================
# api/main.py -> _is_youtube_url()
# ============================================================

def test_is_youtube_url_accepts_standard_watch_link():
    assert _is_youtube_url("https://youtube.com/watch?v=abc123") is True


def test_is_youtube_url_accepts_short_link():
    assert _is_youtube_url("https://youtu.be/abc123") is True


def test_is_youtube_url_rejects_non_youtube_link():
    assert _is_youtube_url("https://vimeo.com/12345") is False


def test_is_youtube_url_rejects_empty_string():
    assert _is_youtube_url("") is False


# ============================================================
# downloaders/yt_downloader.py -> is_youtube_url()
# ============================================================

def test_downloader_recognizes_watch_url():
    downloader = YouTubeDownloader(base_dir="downloads")
    assert downloader.is_youtube_url("https://www.youtube.com/watch?v=xyz") is True


def test_downloader_recognizes_shorts_url():
    downloader = YouTubeDownloader(base_dir="downloads")
    assert downloader.is_youtube_url("https://youtube.com/shorts/xyz") is True


def test_downloader_rejects_local_file_path():
    downloader = YouTubeDownloader(base_dir="downloads")
    assert downloader.is_youtube_url("C:/Users/sagar/Videos/my_video.mp4") is False


# ============================================================
# core/language_config.py
# ============================================================

def test_hindi_is_recognized_as_indian_language():
    assert is_indian_language("Hindi") is True


def test_spanish_is_not_indian_language():
    assert is_indian_language("Spanish") is False


def test_unknown_language_is_not_indian_language():
    """A language that doesn't exist in our config should safely return False, not crash."""
    assert is_indian_language("Klingon") is False


def test_get_sarvam_code_for_known_language():
    assert get_sarvam_code("Tamil") == "ta-IN"


def test_get_sarvam_code_falls_back_to_hindi_for_unknown_language():
    """If someone asks for a Sarvam code for a language Sarvam doesn't support,
    it should default to Hindi's code instead of crashing."""
    assert get_sarvam_code("Klingon") == "hi-IN"


def test_all_languages_contains_both_indian_and_foreign():
    """ALL_LANGUAGES should be the combined list used to populate the dropdown —
    it must contain both Indian and foreign languages."""
    assert "Hindi" in ALL_LANGUAGES
    assert "Spanish" in ALL_LANGUAGES