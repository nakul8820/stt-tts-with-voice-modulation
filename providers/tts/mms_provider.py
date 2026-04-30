# providers/tts/mms_provider.py
# ─────────────────────────────────────────────────────────────
# TTS provider using Meta/Google MMS (Massively Multilingual Speech)
# Fully open, non-gated, and runs 100% locally on CPU/MPS.
# ─────────────────────────────────────────────────────────────

import io
import torch
import numpy as np
import soundfile as sf
from core.base_tts import BaseTTSProvider

class MMSProvider(BaseTTSProvider):
    def __init__(self, cfg: dict):
        self.device = cfg.get("device", "cpu")
        self.model = None
        self.tokenizer = None
        # MMS uses specific repo IDs for different languages
        # 'facebook/mms-tts-hin' is the Hindi model.
        self.model_id = "facebook/mms-tts-hin"

    def load(self) -> None:
        """Load MMS model and tokenizer."""
        from transformers import VitsModel, AutoTokenizer

        print(f"[MMS] Loading {self.model_id} ...")
        
        # This will download automatically (no login required)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = VitsModel.from_pretrained(self.model_id).to(self.device)
        
        print("[MMS] Model ready.")

    def synthesize(self, text: str, voice_id: str = "default") -> bytes:
        """Convert text to audio using MMS (VITS)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**inputs).waveform

        # Convert tensor to numpy float32
        audio_np = output.cpu().numpy().squeeze().astype(np.float32)

        # Encode to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, samplerate=self.model.config.sampling_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()
