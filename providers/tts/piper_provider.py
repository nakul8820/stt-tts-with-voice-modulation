# providers/tts/piper_provider.py
# ─────────────────────────────────────────────────────────────
# TTS provider using Piper (extremely fast, local-first TTS)
# Uses ONNX models — no complex dependencies.
# ─────────────────────────────────────────────────────────────

import io
import os
import wave
import numpy as np
from core.base_tts import BaseTTSProvider

class PiperProvider(BaseTTSProvider):
    def __init__(self, cfg: dict):
        self.device = cfg.get("device", "cpu")
        self.model_path = "models/tts/hi_IN-abid-medium.onnx"
        self.voice = None

    def load(self) -> None:
        """Load Piper model."""
        try:
            from piper import PiperVoice
        except ImportError:
            print("[Piper] Error: piper-tts not installed. Run 'pip install piper-tts'")
            return

        if not os.path.exists(self.model_path):
            print(f"[Piper] Model file not found at {self.model_path}")
            print("[Piper] Please download it from: https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/abid/medium/hi_IN-abid-medium.onnx")
            return

        self.voice = PiperVoice.load(self.model_path)
        print("[Piper] Model loaded successfully.")

    def synthesize(self, text: str, voice_id: str = "default") -> bytes:
        """Synthesize text using Piper."""
        if self.voice is None:
            return b""

        # Create a buffer for the WAV output
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            # Piper synthesize() returns an iterator of audio chunks
            self.voice.synthesize(text, wav_file)
        
        buffer.seek(0)
        return buffer.read()
