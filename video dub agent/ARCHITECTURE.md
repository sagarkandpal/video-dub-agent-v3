# Architecture

## System overview

```mermaid
flowchart TB
    subgraph Client["Client"]
        FE["Frontend (index.html)\nUpload UI + progress polling"]
    end

    subgraph API["FastAPI Backend (api/main.py)"]
        UP["POST /upload\nPOST /upload-youtube"]
        ST["GET /status/{job_id}"]
        DL["GET /download/{job_id}"]
        JM["JobManager\n(in-memory job store)"]
        BG["BackgroundTasks\n(runs pipeline off the request thread)"]
    end

    subgraph Pipeline["core/pipeline.py — LangGraph StateGraph"]
        N1["1. extract_audio"]
        N2["2. diarize_speakers"]
        N3["3. transcribe_segments"]
        N4["4. translate_text"]
        N5["5. generate_dub"]
        N6["6. sync_video"]
        N1 --> N2 --> N3 --> N4 --> N5 --> N6
    end

    subgraph External["External services / models"]
        YT["yt-dlp\n(YouTube download)"]
        PYA["pyannote.audio\n(speaker diarization, HF)"]
        WHISP["Whisper\n(local model)"]
        SARV["Sarvam AI\n(STT translate-mode + TTS)"]
        GROQ["Groq LLM\n(Llama 3.3 — translation)"]
        EDGE["edge-tts\n(foreign-language TTS)"]
        FF["ffmpeg\n(mux, time-stretch)"]
    end

    FE -- "multipart upload / youtube url" --> UP
    UP -- "create job" --> JM
    UP -- "schedule" --> BG
    BG -- "process_video(...)" --> N1
    FE -- "poll every 2s" --> ST
    ST -- "read" --> JM
    N1 <--> YT
    N2 <--> PYA
    N3 <--> WHISP
    N3 <--> SARV
    N4 <--> GROQ
    N5 <--> SARV
    N5 <--> EDGE
    N6 <--> FF
    N6 -- "progress_callback(step,msg,%)" --> JM
    N6 -- "output_path" --> JM
    FE -- "GET download" --> DL
    DL -- "read output_path" --> JM
```

## Request lifecycle

1. **Upload**: Client sends a file or YouTube URL to `/upload` or `/upload-youtube`.
   The endpoint validates language, engine, and file type/duration *before*
   touching the pipeline, creates a `Job` in `JobManager`, and schedules
   `run_dubbing_job()` as a FastAPI `BackgroundTask`. The response (job_id)
   returns immediately — the client never waits for dubbing to finish on this request.

2. **Processing**: `run_dubbing_job()` calls `DubbingPipeline.process_video()`,
   passing a `progress_callback`. The LangGraph pipeline runs its six nodes
   sequentially, each one calling `self._report(...)` at the start, which
   invokes the callback to update the job's `progress_step` / `progress_percent`
   in `JobManager`.

3. **Polling**: The frontend polls `GET /status/{job_id}` every 2 seconds and
   updates a step-by-step UI. No websockets — simple and sufficient given
   dubbing jobs run for tens of seconds to minutes, not real-time.

4. **Download**: Once `status == completed`, `GET /download/{job_id}` streams
   the finished MP4 from disk.

## LangGraph pipeline (core/pipeline.py)

A `StateGraph` with a single `DubbingState` TypedDict flowing through six
nodes, added and wired in a strictly linear chain
(`extract_audio → diarize_speakers → transcribe_segments → translate_text → generate_dub → sync_video → END`).

| Node | Responsibility | Key external call |
|---|---|---|
| `extract_audio` | Download (if YouTube) + extract WAV from video | yt-dlp, pydub |
| `diarize_speakers` | Split audio by speaker + estimate each speaker's gender via pitch (librosa) | pyannote.audio |
| `transcribe_segments` | Convert each speaker's audio to English text | Whisper (local) or Sarvam (translate mode) |
| `translate_text` | English → target language, natural spoken style | Groq (Llama 3.3) |
| `generate_dub` | Text → speech per speaker, gender-matched voice | Sarvam TTS (Indian) or edge-tts (foreign) |
| `sync_video` | Time-stretch each dubbed clip to its original slot, overlay onto a silent base track, mux with ffmpeg | ffmpeg |

Each node reads only the state keys it needs and returns a partial dict that
LangGraph merges into the running state — this keeps nodes independently
testable (see `tests/test_pipeline.py`), since a node can be called directly
with a hand-built `state` dict and mocked components.

## Why LangGraph instead of a plain function chain?

See `DESIGN_DECISIONS.md`.