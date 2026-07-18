# Design Decisions

## 1. LangGraph for pipeline orchestration

**Decision**: Use LangGraph's `StateGraph` instead of a plain sequence of function calls.

**Why**: The dubbing process is inherently a linear pipeline with shared,
growing state (video path → audio path → speakers → transcripts →
translations → dubbed clips → final video). LangGraph gives us:
- A single typed `DubbingState` that's explicit about what each step reads/writes
- Nodes that are independently unit-testable (call a node function directly
  with a hand-built state dict — no need to run the whole graph)
- A natural place to hang progress reporting (`_report()` called at the top
  of every node) without threading callback logic through six separate function signatures
- Room to grow into non-linear flows later (retries, conditional branches
  like "skip translation if source == target language") without restructuring

A simpler alternative (six functions called in sequence in `process_video()`)
would have worked for the current linear flow, but LangGraph was chosen to
keep the pipeline structure explicit and extensible, and because the graph
nodes map directly onto the six pipeline steps in the assignment brief.

## 2. Two transcription engines, routed by source language characteristics

**Decision**: Support both Whisper (general-purpose, local) and Sarvam
(Indian-language specialist, translate-mode API), selectable per request.

**Why**: Whisper handles English (and many languages) well but is noticeably
weaker on Hindi/Hinglish code-switched speech, which is extremely common in
Indian podcast/YouTube content — the platform's primary target content.
Sarvam's `saaras:v3` model in `mode="translate"` was built specifically for
Indian-language audio and outputs English directly, skipping a separate
translation step for the transcription itself. Rather than picking one, the
API exposes `transcription_engine` as a parameter so the caller can match the
engine to their source audio.

**Downstream simplification**: Both engines are designed to always output
**English** text (Whisper transcribes English source directly; Sarvam
translates non-English source to English). This means `translate_text` only
ever has one job — English → target language — instead of needing per-engine
translation logic.

## 3. Two TTS engines, routed by target language

**Decision**: Sarvam TTS (`bulbul:v3`) for the 10 Indian target languages,
edge-tts for the 8 foreign languages.

**Why**: edge-tts is free and has no API key requirement, but its Indian-language
voices are more limited than Sarvam's, which was purpose-built for Indian
languages and speaker naturalness. Foreign languages (Spanish, French, etc.)
are well-covered by edge-tts's neural voices, so there's no need to spend
Sarvam's (rate-limited) quota on them. `is_indian_language()` in
`language_config.py` is the single source of truth for this routing decision.

**Trade-off accepted**: Sarvam TTS calls run sequentially (not in parallel
like edge-tts) because Sarvam's free tier has strict rate limits — this makes
Indian-language dubbing slower, which is documented as a known limitation
rather than hidden.

## 4. Gender-matched voice assignment via pitch analysis

**Decision**: After diarization, concatenate each speaker's audio segments
and run `librosa.pyin` pitch detection once per speaker, classifying male
(<165Hz median) vs. female, then assign a stable voice per speaker for the
whole video.

**Why**: Without this, dubbed voices were being assigned arbitrarily
(alternating or defaulting), producing a jarring gender mismatch — e.g. a male
speaker dubbed with a female voice. Concatenating all of a speaker's segments
before running pitch detection (rather than per-segment) avoids failures on
very short clips (<0.3s), where pitch detection is unreliable.

## 5. Background jobs + polling instead of synchronous requests or websockets

**Decision**: `/upload` returns immediately with a `job_id`; the frontend
polls `/status/{job_id}` every 2 seconds.

**Why**: A dubbing job can take anywhere from ~15 seconds (short clips) to
several minutes (near the 10-minute cap). Holding an HTTP request open for
that long is fragile (timeouts, proxies, browser limits). Polling is simpler
to implement and debug than websockets/SSE, and the 2-second interval is
frequent enough for a responsive-feeling progress bar without meaningfully
loading the server. Given the assignment's core requirement is "processing
status APIs," polling a REST endpoint is also the most directly testable
approach with `TestClient`.

## 6. In-memory job store (no database)

**Decision**: `JobManager` keeps jobs in a plain Python dict guarded by a
`threading.Lock`.

**Why**: Scoped for a single-process local/demo deployment as described in
the assignment (Docker deployment of one service). A database would add
setup complexity disproportionate to the assignment's scope. This is
explicitly called out in `README.md` under Known Limitations — the natural
upgrade path is swapping `JobManager`'s internals for a Redis- or
SQLite-backed store without changing its public interface (`create_job`,
`get_job`, `update_job`, `mark_completed`, `mark_failed`), which is exactly
the kind of substitution the test suite's mocking of `JobManager`-adjacent
logic is designed to tolerate.

## 7. Testing strategy: mock every external/AI call, test everything else directly

**Decision**: No test ever loads Whisper, calls pyannote, hits Groq/Sarvam,
runs edge-tts, or shells out to ffmpeg. `unittest.mock.patch` replaces
`DubbingPipeline` (for API tests) or its six components (for pipeline tests)
with `MagicMock` objects.

**Why**:
- **Speed**: the full suite (54 tests) runs in under a minute; loading
  Whisper alone can take longer than that.
- **Determinism**: AI model outputs vary; testing "did the API call the
  translator with this text" is meaningful and stable, testing "is this
  translation correct" is not something a unit test should assert.
- **No credentials required in CI**: tests never need real HUGGINGFACE_TOKEN,
  GROQ_API_KEY, or SARVAM_API_KEY to pass.
- **Correct scope**: unit/integration tests should verify *our* code's logic
  (validation, routing, state transformation, error handling) — not
  third-party services' correctness, which is out of this project's control anyway.

This intentionally leaves core AI-calling code (`diarizer.py`, `dubber.py`,
`syncer.py`, most of `transcriber.py`) with low measured coverage (13-40%).
That's coverage on lines that make real external calls — verifying those
requires either integration tests against live services (not run in CI) or
manual end-to-end testing (documented as having been done extensively during
development — see the iterative debugging history of this project). What
*is* covered at 89-100% is the code we can and should unit test:
`api/main.py`'s request handling, `language_config.py`'s pure logic, and
`pipeline.py`'s node-level data transformations and control flow.

## 8. Filename sanitization on upload

**Decision**: `_safe_upload_name()` strips special characters and prepends a
random hex prefix to every uploaded filename before saving to disk.

**Why**: Two problems needed solving — (1) filenames with emoji, `#`, spaces,
or other characters (common in YouTube-sourced content, e.g. `"Angry mode on
podcast 😡😳#hindipodcast.mp4"`) could break filesystem operations or shell
calls to ffmpeg downstream, and (2) two uploads with the same original
filename would silently overwrite each other on disk. The random prefix
solves both by guaranteeing a filesystem-safe, unique name while keeping a
recognizable stem for debugging.