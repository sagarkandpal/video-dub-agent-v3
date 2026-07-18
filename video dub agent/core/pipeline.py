"""
core/pipeline.py
LangGraph-based dubbing pipeline.
All files saved in downloads/ folder.
"""

import os
import sys
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import all core components
from .extractor import AudioExtractor
from .transcriber import VideoTranscriber
from .translator import TextTranslator
from .diarizer import SpeakerDiarizer
from .dubber import VoiceDubber
from .syncer import VideoSyncer

load_dotenv()


class DubbingState(TypedDict):
    """State object that flows through the LangGraph pipeline"""
    video_path: str
    audio_path: str
    target_language: str
    transcription_engine: str
    source_language: str
    speakers: List[Dict]
    speaker_genders: Dict[str, str]   # "sarvam" or "whisper"
    transcripts: Dict[str, List[Dict]]
    translated_text: Dict[str, List[Dict]]
    dubbed_segments: List[Dict]
    output_path: str
    status: str
    progress_callback: Any   


class DubbingPipeline:
    """Main pipeline class using LangGraph for workflow orchestration"""
    
    def __init__(self):
        """Initialize all components and build the graph"""
        # Initialize all core components
        self.extractor = AudioExtractor()
        self.transcriber = VideoTranscriber()
        self.translator = TextTranslator()
        self.diarizer = SpeakerDiarizer()
        self.dubber = VoiceDubber(use_elevenlabs=True)
        self.syncer = VideoSyncer()
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self):
        """Build the LangGraph workflow with all nodes and edges"""
        graph = StateGraph(DubbingState)
        
        # Add all nodes
        graph.add_node("extract_audio", self._extract_audio)
        graph.add_node("diarize_speakers", self._diarize_speakers)
        graph.add_node("transcribe_segments", self._transcribe_segments)
        graph.add_node("translate_text", self._translate_text)
        graph.add_node("generate_dub", self._generate_dub)
        graph.add_node("sync_video", self._sync_video)
        
        # Add edges (sequential flow)
        graph.add_edge("extract_audio", "diarize_speakers")
        graph.add_edge("diarize_speakers", "transcribe_segments")
        graph.add_edge("transcribe_segments", "translate_text")
        graph.add_edge("translate_text", "generate_dub")
        graph.add_edge("generate_dub", "sync_video")
        graph.add_edge("sync_video", END)
        
        # Set the entry point
        graph.set_entry_point("extract_audio")
        
        return graph
    

    def _report(self, state: DubbingState, step: str, message: str, percent: int):
        """Call the progress callback if one was provided (used by the API layer)."""
        callback = state.get("progress_callback")
        if callback:
            try:
                callback(step, message, percent)
            except Exception:
                pass   # never let a broken callback crash the pipeline


    def _extract_audio(self, state: DubbingState) -> Dict[str, Any]:
        """Node 1: Extract audio from video"""
        print("📢 Extracting audio from video...")
        self._report(state, "extract_audio", "Extracting audio from video...", 10)
        
        video_path = state["video_path"]
        if self.extractor.youtube_downloader.is_youtube_url(video_path):
            print("🎬 YouTube URL detected. Downloading video first...")
            self._report(state, "extract_audio", "Downloading YouTube video...", 12)
            video_path = self.extractor.download_youtube_video(video_path)
            print(f"🎬 Video downloaded to: {video_path}")
            
        audio_path = self.extractor.extract_from_video(video_path)
        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "status": "Audio extracted successfully"
        }
    

    def _diarize_speakers(self, state: DubbingState) -> Dict[str, Any]:
        """Node 2: Identify speakers and detect their gender"""
        print("🎤 Identifying speakers...")
        self._report(state, "diarize_speakers", "Identifying speakers...", 25)
        speakers = self.diarizer.diarize(state["audio_path"])

        print("🧑‍🤝‍🧑 Detecting speaker gender...")
        gender_map = self.diarizer.detect_speaker_gender(state["audio_path"], speakers)

        return {
            "speakers": speakers,
            "speaker_genders": gender_map,
            "status": f"Found {len(set(s['speaker'] for s in speakers))} speakers"
        }
    

    def _transcribe_segments(self, state: DubbingState) -> Dict[str, Any]:
        print("📝 Transcribing speaker segments...")
        self._report(state, "transcribe_segments", "Transcribing speaker segments...", 40)
        transcripts = {}
        engine = state.get("transcription_engine", "sarvam")

        os.makedirs("downloads/transcripts", exist_ok=True)
        video_name = os.path.splitext(os.path.basename(state["video_path"]))[0]
        transcript_file = f"downloads/transcripts/{video_name}_original_english.txt"

        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(f"Engine used: {engine}\n\n")
            for speaker in state["speakers"]:
                speaker_id = speaker["speaker"]
                if speaker_id not in transcripts:
                    transcripts[speaker_id] = []

                text = self.transcriber.transcribe_segment(
                    state["audio_path"],
                    speaker["start"],
                    speaker["end"],
                    engine=engine
                )

                transcripts[speaker_id].append({
                    "text": text,
                    "start": speaker["start"],
                    "end": speaker["end"]
                })

                f.write(f"[{speaker_id}] ({speaker['start']:.2f}s - {speaker['end']:.2f}s) [engine={engine}]\n{text}\n\n")

        print(f"📄 English transcript saved: {transcript_file}")

        return {
            "transcripts": transcripts,
            "status": f"Transcribed {len(transcripts)} speakers using {engine}"
        }
    

    
    def _translate_text(self, state: DubbingState) -> Dict[str, Any]:
        """Node 4: Translate transcribed text and save to file"""
        target_lang = state.get('target_language', 'Hindi')
        print(f"🌍 Translating to {target_lang}...")
        self._report(state, "translate_text", f"Translating to {target_lang}...", 55)
        translated = {}

        video_name = os.path.splitext(os.path.basename(state["video_path"]))[0]
        translated_file = f"downloads/transcripts/{video_name}_translated_{target_lang}.txt"

        with open(translated_file, "w", encoding="utf-8") as f:
            for speaker_id, segments in state["transcripts"].items():
                translated[speaker_id] = []
                for segment in segments:
                    translated_text = self.translator.translate(
                        segment["text"],
                        target_lang
                    )
                    translated[speaker_id].append({
                        "text": translated_text,
                        "start": segment["start"],
                        "end": segment["end"]
                    })
                    f.write(f"[{speaker_id}] ({segment['start']:.2f}s - {segment['end']:.2f}s)\n{translated_text}\n\n")

        print(f"📄 Translated transcript saved: {translated_file}")

        return {
            "translated_text": translated,
            "status": f"Translation complete → {translated_file}"
        }
    

    
    def _generate_dub(self, state: DubbingState) -> Dict[str, Any]:
        print("🎙️ Generating dubbed voices...")
        self._report(state, "generate_dub", "Generating dubbed voices...", 70)
        os.makedirs("downloads/dubbed", exist_ok=True)

        flat_segments = []
        for speaker_id, segments in state["translated_text"].items():
            gender = state.get("speaker_genders", {}).get(speaker_id, "unknown")

            for segment in segments:
                flat_segments.append({
                    "speaker": speaker_id,
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "gender": gender          # ← ab yeh use hoga, purana speaker_voice hata do
                })

        dubbed_segments = self.dubber.generate_segment_dubs(
            flat_segments,
            output_dir="downloads/dubbed",
            target_language=state.get("target_language", "Hindi")
        )

        return {
            "dubbed_segments": dubbed_segments,
            "status": f"Generated {len(dubbed_segments)} dubbed segments"
        }
    
    
    def _sync_video(self, state: DubbingState) -> Dict[str, Any]:
        """Node 6: Sync dubbed audio with video"""
        print("🔄 Syncing audio with video...")
        self._report(state, "sync_video", "Syncing audio with video...", 90)
        
        # Create output directory in downloads/
        os.makedirs("downloads/output", exist_ok=True)
        
        # Generate output path in downloads/output/
        video_name = os.path.splitext(os.path.basename(state["video_path"]))[0]
        output_path = f"downloads/output/{video_name}_dubbed.mp4"
        
        final_path = self.syncer.sync_segments_to_video(
            state["video_path"],
            state["dubbed_segments"],
            output_path
        )
        
        return {
            "output_path": final_path,
            "status": "🎉 Dubbing complete!"
        }
    

    
    def process_video(self, video_path: str, target_language: str = "Hindi",
                   transcription_engine: str = "sarvam", progress_callback=None) -> Dict:
        """Process a video through the entire dubbing pipeline.
        progress_callback: optional function(step: str, message: str, percent: int),
        used by the API layer to report live status. CLI usage (main.py) doesn't pass
        this, so it stays None and nothing changes there."""
        # Ensure all directories exist in downloads/
        os.makedirs("downloads", exist_ok=True)
        os.makedirs("downloads/videos", exist_ok=True)
        os.makedirs("downloads/audio", exist_ok=True)
        os.makedirs("downloads/transcripts", exist_ok=True)
        os.makedirs("downloads/dubbed", exist_ok=True)
        os.makedirs("downloads/output", exist_ok=True)
        
        # Initial state
        initial_state = {
            "video_path": video_path,
            "audio_path": "",
            "target_language": target_language,
            "transcription_engine": transcription_engine,
            "source_language": "Hindi", 
            "speakers": [],
            "speaker_genders": {},
            "transcripts": {},
            "translated_text": {},
            "dubbed_segments": [],
            "output_path": "",
            "status": "Starting pipeline...",
            "progress_callback": progress_callback
        }
        
        # Run the LangGraph pipeline
        result = self.app.invoke(initial_state)
        return result