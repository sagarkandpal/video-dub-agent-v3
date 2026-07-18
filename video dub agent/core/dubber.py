"""
core/dubber.py
Voice cloning using edge-tts (free, no API key needed).
"""

import os
import base64
import asyncio
import subprocess
import wave
import numpy as np
import edge_tts
from sarvamai import SarvamAI
from core.language_config import is_indian_language, get_sarvam_code, INDIAN_LANGUAGES

# Free voice pool per target language. Two distinct voices per language
# so different speakers get different voices (identity preserved).
# Free voice pool per target language, split by gender
VOICE_POOL = {
    "Hindi":      {"male": ["hi-IN-MadhurNeural"],   "female": ["hi-IN-SwaraNeural"]},
    "Spanish":    {"male": ["es-ES-AlvaroNeural"],   "female": ["es-ES-ElviraNeural"]},
    "French":     {"male": ["fr-FR-HenriNeural"],    "female": ["fr-FR-DeniseNeural"]},
    "German":     {"male": ["de-DE-ConradNeural"],   "female": ["de-DE-KatjaNeural"]},
    "Japanese":   {"male": ["ja-JP-KeitaNeural"],    "female": ["ja-JP-NanamiNeural"]},
    "Chinese":    {"male": ["zh-CN-YunxiNeural"],    "female": ["zh-CN-XiaoxiaoNeural"]},
    "Arabic":     {"male": ["ar-SA-HamedNeural"],    "female": ["ar-SA-ZariyahNeural"]},
    "Portuguese": {"male": ["pt-BR-AntonioNeural"],  "female": ["pt-BR-FranciscaNeural"]},
    "Russian":    {"male": ["ru-RU-DmitryNeural"],   "female": ["ru-RU-SvetlanaNeural"]},
}


class VoiceDubber:
    """Generate dubbed voices using edge-tts"""

    def __init__(self, use_elevenlabs: bool = False):
        # use_elevenlabs kept as a parameter only so pipeline.py doesn't break;
        # edge-tts is always used now.
        self.speaker_voice_map = {}
        print("✅ edge-tts dubber initialized (free, no API key needed)")
        sarvam_key = os.getenv("SARVAM_API_KEY")
        self.sarvam_client = SarvamAI(api_subscription_key=sarvam_key) if sarvam_key else None


    def get_voice_for_speaker(self, speaker_id: str, target_language: str = "Hindi", gender: str = "unknown") -> str:
        """Return a voice for this speaker, matched to detected gender when known"""
        lang_pool = VOICE_POOL.get(target_language, VOICE_POOL["Hindi"])

        if speaker_id not in self.speaker_voice_map:
            if gender == "male":
                voice = lang_pool["male"][0]
            elif gender == "female":
                voice = lang_pool["female"][0]
            else:
                # unknown gender: alternate between pools so different speakers still sound different
                idx = len(self.speaker_voice_map) % 2
                voice = lang_pool["female"][0] if idx == 0 else lang_pool["male"][0]

            self.speaker_voice_map[speaker_id] = voice
            print(f"🎙️ Assigned voice {voice} to {speaker_id} (gender={gender})")

        return self.speaker_voice_map[speaker_id]



    def generate_dub(self, text: str, output_path: str, speaker_voice: str = None,target_language: str = "Hindi", gender: str = None) -> str:
        """
        Generate dubbed audio. Routes to Sarvam TTS for Indian languages,
        and edge-tts for foreign languages.
        """
        if not text or not text.strip():
            text = "..."

        # ── Route 1: Indian language -> Sarvam TTS ──────────────────────
        if is_indian_language(target_language) and self.sarvam_client:
            return self._generate_dub_sarvam(text, output_path, target_language, gender)

        # ── Route 2: Foreign language -> edge-tts ───────────────────────
        voice = speaker_voice or "hi-IN-SwaraNeural"
        temp_mp3 = output_path.replace(".wav", "_raw.mp3")

        try:
            async def _run():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(temp_mp3)

            asyncio.run(_run())

            if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
                raise Exception("edge-tts produced empty file")

            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_mp3, "-ar", "44100", "-ac", "1", output_path],
                check=True, capture_output=True
            )
            os.remove(temp_mp3)

            print(f"✅ Generated (edge-tts): {os.path.basename(output_path)} "
                f"({os.path.getsize(output_path)/1024:.1f} KB) [voice={voice}]")
            return output_path

        except Exception as e:
            print(f"⚠️ edge-tts generation failed: {e}")
            print("Using silent audio fallback...")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            return self._create_silent_audio(text, output_path, duration=5)


    def _generate_dub_sarvam(self, text: str, output_path: str, target_language: str, gender: str = None) -> str:
        """Generate dubbed audio using Sarvam bulbul:v3 (Indian languages)"""
        try:
            lang_code = get_sarvam_code(target_language)
            speaker = "amit" if gender == "male" else "priya"

            audio = self.sarvam_client.text_to_speech.convert(
                text=text,
                target_language_code=lang_code,
                model="bulbul:v3",
                speaker=speaker
            )

            temp_wav = output_path.replace(".wav", "_raw.wav")
            audio_bytes = base64.b64decode(audio.audios[0])
            with open(temp_wav, "wb") as f:
                f.write(audio_bytes)

            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_wav, "-ar", "44100", "-ac", "1", output_path],
                check=True, capture_output=True
            )
            os.remove(temp_wav)

            print(f"✅ Generated (Sarvam): {os.path.basename(output_path)} "
                f"({os.path.getsize(output_path)/1024:.1f} KB) [speaker={speaker}, lang={lang_code}]")
            return output_path

        except Exception as e:
            print(f"⚠️ Sarvam TTS failed: {e}")
            print("Using silent audio fallback...")
            return self._create_silent_audio(text, output_path, duration=5)


    def _create_silent_audio(self, text: str, output_path: str, duration: float = 5.0) -> str:
        """Create silent audio file as fallback"""
        try:
            sample_rate = 16000
            samples = int(duration * sample_rate)
            silent_audio = np.zeros(samples, dtype=np.int16)

            with wave.open(output_path, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(silent_audio.tobytes())

            print(f"🔇 Created silent audio: {os.path.basename(output_path)} ({duration}s)")
            return output_path

        except Exception as e:
            print(f"⚠️ Could not create silent audio: {e}")
            return output_path



    def generate_segment_dubs(self, segments: list, output_dir: str = "downloads/dubbed", target_language: str = "Hindi") -> list:
        """Generate dubs for multiple text segments IN PARALLEL.
        Routes each segment to Sarvam (Indian target) or edge-tts (foreign target)."""
        os.makedirs(output_dir, exist_ok=True)

        jobs = []
        for i, segment in enumerate(segments):
            speaker = segment.get("speaker", "SPEAKER_01")
            output_path = f"{output_dir}/{speaker}_{i}.wav"

            text = segment.get("text", "")
            if not text or len(text.strip()) == 0:
                text = "..."

            segment_duration = segment.get("end", 5) - segment.get("start", 0)
            if segment_duration < 3:
                segment_duration = 5

            gender = segment.get("gender", "unknown")

            jobs.append({
                "index": i,
                "speaker": speaker,
                "output_path": output_path,
                "text": text,
                "gender": gender,
                "start": segment.get("start", 0),
                "end": segment.get("end", segment.get("start", 0) + segment_duration),
                "segment_duration": segment_duration
            })

        if is_indian_language(target_language) and self.sarvam_client:
            print(f"🎙️ Using Sarvam TTS for {target_language} (sequential — Sarvam has stricter rate limits)")
            for job in jobs:
                self._generate_dub_sarvam(job["text"], job["output_path"], target_language, job["gender"])
        else:
            print("🎙️ Using edge-tts (parallel)")
            asyncio.run(self._generate_all_async_edge(jobs, target_language))

        dubbed_segments = []
        for job in jobs:
            output_path = job["output_path"]
            try:
                with wave.open(output_path, 'rb') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    actual_duration = frames / float(rate)
                if actual_duration == 0:
                    actual_duration = job["segment_duration"]
            except:
                actual_duration = job["segment_duration"]

            dubbed_segments.append({
                "speaker": job["speaker"],
                "audio_path": output_path,
                "start": job["start"],
                "end": job["end"],
                "text": job["text"],
                "duration": actual_duration
            })

        return dubbed_segments


    async def _generate_all_async_edge(self, jobs: list, target_language: str):
        """Run edge-tts generation for all jobs concurrently (foreign languages only)"""
        semaphore = asyncio.Semaphore(5)

        async def _generate_one(job):
            async with semaphore:
                voice = self.get_voice_for_speaker(job["speaker"], target_language, job["gender"])
                await self._generate_dub_async_edge(job["text"], job["output_path"], voice)

        await asyncio.gather(*[_generate_one(job) for job in jobs])


    async def _generate_dub_async_edge(self, text: str, output_path: str, voice: str):
        """Async edge-tts generation (foreign languages only)"""
        if not text or not text.strip():
            text = "..."

        temp_mp3 = output_path.replace(".wav", "_raw.mp3")

        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(temp_mp3)

            if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
                raise Exception("edge-tts produced empty file")

            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_mp3, "-ar", "44100", "-ac", "1", output_path],
                check=True, capture_output=True
            )
            os.remove(temp_mp3)
            print(f"✅ Generated (edge-tts): {os.path.basename(output_path)} [voice={voice}]")

        except Exception as e:
            print(f"⚠️ edge-tts generation failed for {output_path}: {e}")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            self._create_silent_audio(text, output_path, duration=5)


    async def _generate_all_async(self, jobs: list):
        """Run edge-tts generation for all jobs concurrently, limited batches to avoid overload"""
        semaphore = asyncio.Semaphore(5)  # max 5 concurrent TTS calls at once

        async def _generate_one(job):
            async with semaphore:
                await self._generate_dub_async(job["text"], job["output_path"], job["voice"])

        await asyncio.gather(*[_generate_one(job) for job in jobs])


    async def _generate_dub_async(self, text: str, output_path: str, voice: str):
        """Async version of generate_dub, used internally for parallel generation"""
        if not text or not text.strip():
            text = "..."

        temp_mp3 = output_path.replace(".wav", "_raw.mp3")

        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(temp_mp3)

            if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
                raise Exception("edge-tts produced empty file")

            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_mp3, "-ar", "44100", "-ac", "1", output_path],
                check=True, capture_output=True
            )
            os.remove(temp_mp3)

            print(f"✅ Generated audio: {os.path.basename(output_path)} [voice={voice}]")

        except Exception as e:
            print(f"⚠️ edge-tts generation failed for {output_path}: {e}")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            self._create_silent_audio(text, output_path, duration=5)