# Video Dubbing Platform

An AI-powered platform that translates and dubs videos containing multiple
speakers into a target language — preserving speaker identity, conversation
flow, and audio-video synchronization. Exposes a FastAPI backend with an
HTML/JS frontend.

## Features

- Accepts local video files (MP4, MOV, AVI, MKV, WEBM) or YouTube URLs, up to 10 minutes
- Speaker diarization — identifies and separates multiple speakers in one video
- Gender-aware voice assignment (pitch analysis on each speaker's audio)
- Transcription via two swappable engines: **Sarvam** (best for Hindi/Hinglish source audio) or **Whisper** (best for English source audio)
- Translation to 18 languages (10 Indian + 8 foreign) via Groq LLM, tuned for natural spoken dialogue rather than literal translation
- Voice dubbing via **Sarvam TTS** (Indian target languages) or **edge-tts** (foreign target languages), with a consistent voice per speaker
- Time-stretched audio sync so dubbed dialogue fits each speaker's original time slot
- Background job processing with live progress tracking via polling
- REST API for upload, status polling, and download

## Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full system diagram and
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md) for why things are built the way they are.

## Project Structure

```
video dub agent/
├── api/                    FastAPI backend
│   ├── main.py              app, routes, job orchestration
│   ├── job_manager.py       (legacy, superseded by in-file Job/JobManager)
│   └── models.py            Pydantic response schemas
├── core/                   Dubbing pipeline (LangGraph)
│   ├── pipeline.py          6-node LangGraph workflow
│   ├── extractor.py         video -> audio extraction (local + YouTube)
│   ├── diarizer.py          pyannote speaker diarization + gender detection
│   ├── transcriber.py       Whisper / Sarvam speech-to-text
│   ├── translator.py        Groq LLM translation
│   ├── dubber.py            edge-tts / Sarvam text-to-speech
│   ├── syncer.py            ffmpeg-based audio/video muxing + time-stretch
│   └── language_config.py   supported languages + engine routing
├── downloaders/
│   └── yt_downloader.py     yt-dlp wrapper
├── frontend/
│   └── index.html           upload UI, progress tracking, playback
├── tests/                  Automated test suite (pytest)
├── main.py                  CLI entry point (local testing without the API)
├── requirements.txt
└── requirements-test.txt
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
Requires **ffmpeg** installed and on PATH.

### 2. Configure environment
Create a `.env` file in the project root:
```env
HUGGINGFACE_TOKEN=your_hf_token       # for pyannote speaker diarization
GROQ_API_KEY=your_groq_key            # for translation
SARVAM_API_KEY=your_sarvam_key        # for Indian-language STT/TTS
WHISPER_MODEL=base                    # optional, defaults to "base"
```
- HuggingFace: accept the gated model licenses for `pyannote/speaker-diarization-community-1` and `pyannote/segmentation-3.0` on huggingface.co before first run.
- edge-tts (foreign-language TTS) needs no API key.

### 3. Run the backend
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
- API root: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 4. Open the frontend
Open `frontend/index.html` directly in a browser (or via `http://localhost:8000/ui`
if served through the backend's static mount). It talks to the API at
`http://localhost:8000` by default — change `API_BASE` at the top of the
`<script>` block if your backend runs elsewhere.

### 5. (Optional) Run via CLI instead of the API
```bash
python main.py
```
Prompts for a video path/YouTube URL, target language, and transcription engine,
then runs the pipeline synchronously in the terminal.

## API Usage

### `GET /health`
Returns API status and whether the AI pipeline has been loaded yet.
```json
{ "ok": true, "pipeline_loaded": false, "jobs_count": 0 }
```

### `GET /languages`
Returns all supported target languages.
```json
{ "languages": ["Hindi", "Tamil", ..., "Russian"] }
```

### `POST /upload`
Upload a local video file and start dubbing in the background.

| Field | Type | Notes |
|---|---|---|
| `file` | file | required. mp4/mov/avi/mkv/webm |
| `target_language` | form field | default `"Hindi"` |
| `transcription_engine` | form field | `"whisper"` or `"sarvam"`, default `"whisper"` |

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my_video.mp4" \
  -F "target_language=Hindi" \
  -F "transcription_engine=whisper"
```
```json
{ "job_id": "a1b2c3d4-...", "message": "Upload received...", "status_url": "/status/a1b2c3d4-..." }
```
Returns `400` for unsupported language/engine/file type, or if the video exceeds 10 minutes.

### `POST /upload-youtube`
Same as above, but for a YouTube URL instead of a file.

| Field | Type |
|---|---|
| `youtube_url` | form field, required |
| `target_language` | form field |
| `transcription_engine` | form field |

```bash
curl -X POST http://localhost:8000/upload-youtube \
  -F "youtube_url=https://youtube.com/watch?v=xxxxx" \
  -F "target_language=Tamil" \
  -F "transcription_engine=sarvam"
```

### `GET /status/{job_id}`
Poll this to track progress (recommended: every 2 seconds).
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "processing",
  "progress_step": "translate_text",
  "progress_message": "Translating to Hindi...",
  "progress_percent": 55,
  "video_filename": "my_video.mp4",
  "target_language": "Hindi",
  "transcription_engine": "whisper",
  "output_path": null,
  "error": null
}
```
`status` is one of `pending`, `processing`, `completed`, `failed`. Returns `404` if the job doesn't exist.

### `GET /download/{job_id}`
Downloads the finished MP4 once `status` is `completed`.
Returns `409` if the job isn't finished yet, `404` if the job or output file doesn't exist.

### `GET /jobs`
Lists all jobs in memory (debugging/admin use).

## Testing

```bash
pip install -r requirements-test.txt
pytest -v                                              # run all tests
pytest tests/test_api.py -v                            # run one file
pytest tests/test_api.py::test_health_endpoint_returns_ok -v   # run one test
pytest --cov=api --cov=core --cov-report=term-missing  # coverage report
```

54 tests across 4 files:
- `test_job_manager.py` — job lifecycle, pure logic, no mocking
- `test_helpers.py` — filename sanitization, URL validation, language config
- `test_api.py` — all endpoints, with the AI pipeline mocked out
- `test_pipeline.py` — LangGraph structure and node logic, with all 6 heavy AI components mocked

Real external services (Whisper, pyannote, Sarvam, Groq, edge-tts, ffmpeg) are
never called during tests — see `DESIGN_DECISIONS.md` for why.

## Known Limitations

- In-memory job store — jobs are lost on server restart (no database yet)
- No authentication/rate limiting on the API
- Sarvam TTS runs sequentially (its free tier has strict rate limits), so
  Indian-language dubbing is slower than foreign-language dubbing (parallel edge-tts)
- Gender detection defaults to alternating voices when pitch analysis is inconclusive on short clips

## Docker Hub
Pre-built image available: `docker pull sagarkandpal/videodubagent:latest`