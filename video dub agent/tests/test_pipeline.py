"""
tests/test_pipeline.py
Tests for the LangGraph pipeline structure and node-level logic in core/pipeline.py.

We mock all 6 heavy components (extractor, transcriber, diarizer, translator,
dubber, syncer) so building a DubbingPipeline here never loads Whisper,
pyannote, or makes any real API calls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_pipeline():
    """
    Builds a real DubbingPipeline object, but with every heavy internal
    component replaced by a MagicMock. This lets us test the graph
    structure and node logic without loading any real AI models.
    """
    with patch("core.pipeline.AudioExtractor") as MockExtractor, \
         patch("core.pipeline.VideoTranscriber") as MockTranscriber, \
         patch("core.pipeline.TextTranslator") as MockTranslator, \
         patch("core.pipeline.SpeakerDiarizer") as MockDiarizer, \
         patch("core.pipeline.VoiceDubber") as MockDubber, \
         patch("core.pipeline.VideoSyncer") as MockSyncer:

        from core.pipeline import DubbingPipeline
        pipeline = DubbingPipeline()
        yield pipeline


# ============================================================
# Graph structure tests
# ============================================================

def test_pipeline_builds_without_error(mock_pipeline):
    """Simply constructing the pipeline (with mocked components) should not raise."""
    assert mock_pipeline is not None
    assert mock_pipeline.app is not None


def test_graph_contains_all_six_nodes(mock_pipeline):
    """
    The compiled graph should contain exactly the 6 nodes we expect.
    This protects against someone accidentally renaming or removing a step.
    """
    graph_nodes = mock_pipeline.graph.nodes.keys()

    expected_nodes = {
        "extract_audio",
        "diarize_speakers",
        "transcribe_segments",
        "translate_text",
        "generate_dub",
        "sync_video",
    }
    assert expected_nodes.issubset(set(graph_nodes))


# ============================================================
# _report() — progress callback logic
# ============================================================

def test_report_calls_callback_with_correct_arguments(mock_pipeline):
    """
    _report() should call the progress_callback function stored in state
    with exactly the step, message, and percent we pass in.
    """
    received = []

    def fake_callback(step, message, percent):
        received.append((step, message, percent))

    state = {"progress_callback": fake_callback}

    mock_pipeline._report(state, "extract_audio", "Extracting audio...", 10)

    assert received == [("extract_audio", "Extracting audio...", 10)]


def test_report_does_nothing_when_no_callback_provided(mock_pipeline):
    """
    If state has no progress_callback (e.g. CLI usage via main.py), _report()
    should just silently do nothing — not raise an error.
    """
    state = {"progress_callback": None}
    # Should not raise any exception
    mock_pipeline._report(state, "extract_audio", "Extracting audio...", 10)


def test_report_does_not_crash_if_callback_itself_raises(mock_pipeline):
    """
    If the callback function itself has a bug and raises an exception,
    _report() should swallow that error rather than crashing the whole
    dubbing pipeline over a progress-reporting glitch.
    """
    def broken_callback(step, message, percent):
        raise ValueError("something went wrong in the UI layer")

    state = {"progress_callback": broken_callback}

    # Should not raise, even though broken_callback raises internally
    mock_pipeline._report(state, "extract_audio", "Extracting audio...", 10)


# ============================================================
# _extract_audio() node logic (mocked extractor)
# ============================================================

def test_extract_audio_node_calls_extractor_and_returns_audio_path(mock_pipeline):
    """
    _extract_audio should call extractor.extract_from_video() and return
    its result as audio_path in the state update.
    """
    mock_pipeline.extractor.youtube_downloader.is_youtube_url.return_value = False
    mock_pipeline.extractor.extract_from_video.return_value = "downloads/audio/test.wav"

    state = {
        "video_path": "downloads/videos/test.mp4",
        "progress_callback": None,
    }

    result = mock_pipeline._extract_audio(state)

    assert result["audio_path"] == "downloads/audio/test.wav"
    mock_pipeline.extractor.extract_from_video.assert_called_once()


def test_extract_audio_node_downloads_youtube_video_first(mock_pipeline):
    """
    If video_path is a YouTube URL, the node should first call
    download_youtube_video() to get a local file, THEN extract audio from it.
    """
    mock_pipeline.extractor.youtube_downloader.is_youtube_url.return_value = True
    mock_pipeline.extractor.download_youtube_video.return_value = "downloads/videos/downloaded.mp4"
    mock_pipeline.extractor.extract_from_video.return_value = "downloads/audio/downloaded.wav"

    state = {
        "video_path": "https://youtube.com/watch?v=abc123",
        "progress_callback": None,
    }

    result = mock_pipeline._extract_audio(state)

    mock_pipeline.extractor.download_youtube_video.assert_called_once_with(
        "https://youtube.com/watch?v=abc123"
    )
    assert result["video_path"] == "downloads/videos/downloaded.mp4"
    assert result["audio_path"] == "downloads/audio/downloaded.wav"


# ============================================================
# _diarize_speakers() node logic (mocked diarizer)
# ============================================================

def test_diarize_speakers_node_returns_speakers_and_genders(mock_pipeline):
    """
    _diarize_speakers should call diarizer.diarize() then diarizer.detect_speaker_gender(),
    and put both results into the state update.
    """
    fake_speakers = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "duration": 5.0}
    ]
    fake_genders = {"SPEAKER_00": "male"}

    mock_pipeline.diarizer.diarize.return_value = fake_speakers
    mock_pipeline.diarizer.detect_speaker_gender.return_value = fake_genders

    state = {"audio_path": "downloads/audio/test.wav", "progress_callback": None}

    result = mock_pipeline._diarize_speakers(state)

    assert result["speakers"] == fake_speakers
    assert result["speaker_genders"] == fake_genders


# ============================================================
# _generate_dub() node logic — data transformation test
# ============================================================

def test_generate_dub_node_flattens_translated_segments_correctly(mock_pipeline):
    """
    _generate_dub takes translated_text (a dict keyed by speaker_id) and
    should flatten it into a single list of segments with gender attached,
    before passing it to dubber.generate_segment_dubs().
    """
    mock_pipeline.dubber.generate_segment_dubs.return_value = []

    state = {
        "video_path": "test.mp4",
        "target_language": "Hindi",
        "progress_callback": None,
        "speaker_genders": {"SPEAKER_00": "male", "SPEAKER_01": "female"},
        "translated_text": {
            "SPEAKER_00": [{"text": "नमस्ते", "start": 0.0, "end": 2.0}],
            "SPEAKER_01": [{"text": "कैसे हो", "start": 2.0, "end": 4.0}],
        },
    }

    mock_pipeline._generate_dub(state)

    # Check what was actually passed to generate_segment_dubs()
    call_args = mock_pipeline.dubber.generate_segment_dubs.call_args
    flat_segments = call_args[0][0]  # first positional argument

    assert len(flat_segments) == 2
    speakers_seen = {seg["speaker"] for seg in flat_segments}
    assert speakers_seen == {"SPEAKER_00", "SPEAKER_01"}

    # Gender should have been correctly attached to each segment
    seg_00 = next(s for s in flat_segments if s["speaker"] == "SPEAKER_00")
    assert seg_00["gender"] == "male"


def test_generate_dub_node_uses_unknown_gender_when_missing(mock_pipeline):
    """
    If a speaker has no entry in speaker_genders (e.g. gender detection
    failed), the segment should default to gender="unknown" instead of crashing.
    """
    mock_pipeline.dubber.generate_segment_dubs.return_value = []

    state = {
        "video_path": "test.mp4",
        "target_language": "Hindi",
        "progress_callback": None,
        "speaker_genders": {},   # no gender info at all
        "translated_text": {
            "SPEAKER_00": [{"text": "hello", "start": 0.0, "end": 2.0}],
        },
    }

    mock_pipeline._generate_dub(state)

    call_args = mock_pipeline.dubber.generate_segment_dubs.call_args
    flat_segments = call_args[0][0]

    assert flat_segments[0]["gender"] == "unknown"