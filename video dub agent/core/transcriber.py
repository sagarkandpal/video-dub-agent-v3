"""
core/transcriber.py
Two transcription engines, both producing ENGLISH text:
- Sarvam (mode="translate"): best for Hindi/Hinglish source audio,
  translates directly to English during transcription.
- Whisper: best for English source audio, transcribes as-is.
The rest of the pipeline always works with English text, then a single
Groq translation step converts English -> target language (Indian or foreign).
"""

import os
import whisper
import torch
from sarvamai import SarvamAI
from dotenv import load_dotenv

load_dotenv()


class VideoTranscriber:
    """Transcribe audio to English using Sarvam or Whisper"""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📢 Using device: {self.device} for transcription")
        self.model = whisper.load_model(model_size, device=self.device)
        self._full_transcript_cache = {}   # (audio_path, engine) -> word list

        sarvam_key = os.getenv("SARVAM_API_KEY")
        self.sarvam_client = SarvamAI(api_subscription_key=sarvam_key) if sarvam_key else None
        if not self.sarvam_client:
            print("⚠️ SARVAM_API_KEY not found — Sarvam engine will fall back to Whisper")

        os.makedirs("downloads/transcripts", exist_ok=True)

    # ── Whisper path ────────────────────────────────────────────────
    def _transcribe_with_whisper(self, audio_path: str) -> list:
        print("📝 Running Whisper (English) on full audio...")
        result = self.model.transcribe(
            audio_path,
            task="transcribe",
            word_timestamps=True,
            verbose=False
        )
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
        return words

    # ── Sarvam path (translate mode -> English) ─────────────────────
    def _transcribe_with_sarvam(self, audio_path: str) -> list:
        print("📝 Running Sarvam (translate mode -> English) on full audio...")
        with open(audio_path, "rb") as f:
            response = self.sarvam_client.speech_to_text.transcribe(
                file=(os.path.basename(audio_path), f, "audio/wav"),
                model="saaras:v3",
                mode="translate"   # Hindi/Hinglish audio -> English text
            )

        words = []
        if hasattr(response, "timestamps") and response.timestamps:
            for ts in response.timestamps:
                words.append({
                    "word": ts.get("text", "") + " ",
                    "start": ts.get("start_time_seconds", 0),
                    "end": ts.get("end_time_seconds", 0)
                })
        else:
            # No word-level timestamps returned — treat whole transcript as one block
            words.append({"word": response.transcript, "start": 0, "end": 999999})

        return words

    def _get_words(self, audio_path: str, engine: str = "sarvam") -> list:
        cache_key = f"{audio_path}::{engine}"
        if cache_key in self._full_transcript_cache:
            return self._full_transcript_cache[cache_key]

        if engine == "sarvam" and self.sarvam_client:
            words = self._transcribe_with_sarvam(audio_path)
        else:
            words = self._transcribe_with_whisper(audio_path)

        self._full_transcript_cache[cache_key] = words
        return words

    def transcribe_segment(self, audio_path: str, start: float, end: float, engine: str = "sarvam") -> str:
        """Get English text for a specific time range using the chosen engine"""
        try:
            words = self._get_words(audio_path, engine)
            segment_words = [w["word"] for w in words if w["start"] < end and w["end"] > start]
            return "".join(segment_words).strip()
        except Exception as e:
            print(f"⚠️ Segment transcription error: {e}")
            return ""