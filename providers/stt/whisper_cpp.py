# providers/stt/whisper_cpp.py
# ─────────────────────────────────────────────────────────────
# STT provider using whisper.cpp via the pywhispercpp binding.
# whisper.cpp is a pure C++ port of Whisper — it uses Apple's
# Core ML and Neural Engine on M2, making it the fastest option.
#
# Install: pip install pywhispercpp
# Model download: happens automatically on first use.
#
# NOTE: This is optional. If you just want to start, use
# whisper_faster. Come back to this when you want to experiment.
# ─────────────────────────────────────────────────────────────

import os
import tempfile
from core.base_stt import BaseSTTProvider


class WhisperCppProvider(BaseSTTProvider):
    """
    STT provider using whisper.cpp (pywhispercpp binding).
    Fastest option on M2 — uses Apple Neural Engine.
    """

    def __init__(self, cfg: dict):
        self.model_size = cfg.get("model_size", "medium")
        self.language = cfg.get("language", "hi")
        self.model = None

    def load(self) -> None:
        """
        Load whisper.cpp model.
        Checks for local file in models/stt/ first, otherwise uses default cache.
        """
        try:
            from pywhispercpp.model import Model
            
            # 1. Try to find local model file first
            local_path = os.path.join("models", "stt", f"ggml-{self.model_size}.bin")
            
            if os.path.exists(local_path):
                print(f"[WhisperCpp] Found local model file at: {local_path}")
                self.model = Model(local_path, print_realtime=False)
            else:
                # 2. Fallback to default (downloads to ~/.cache/pywhispercpp)
                print(f"[WhisperCpp] Local model not found at {local_path}")
                print(f"[WhisperCpp] Loading model from default cache: {self.model_size} ...")
                self.model = Model(self.model_size, print_realtime=False)
                
            print("[WhisperCpp] Model loaded.")
        except ImportError:
            raise ImportError(
                "pywhispercpp not installed. Run: pip install pywhispercpp"
            )

    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Transcribe audio bytes using whisper.cpp.
        Handles conversion from various formats (WebM, etc.) to 16kHz mono WAV.
        """
        import io
        import librosa
        import soundfile as sf

        # 1. Write incoming bytes to a temporary file first
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as f:
            f.write(audio_bytes)
            raw_path = f.name

        processed_path = None
        try:
            # 2. Use librosa to load and convert to 16kHz mono
            # librosa.load is robust and handles resampling/mono conversion
            y, sr = librosa.load(raw_path, sr=16000)
            
            # 3. Write as a clean RIFF WAV file (16-bit PCM)
            # whisper.cpp requires this specific format
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, y, 16000, format='WAV', subtype='PCM_16')
                processed_path = f.name

            # 4. Run transcription
            segments = self.model.transcribe(processed_path, language=self.language)
            text = " ".join(s.text for s in segments)
            return {"text": text.strip(), "language": self.language}
            
        except Exception as e:
            print(f"[WhisperCpp] Transcription error: {e}")
            raise e
        finally:
            # Clean up both temporary files
            if os.path.exists(raw_path):
                os.unlink(raw_path)
            if processed_path and os.path.exists(processed_path):
                os.unlink(processed_path)