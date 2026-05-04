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
        # Force CPU for MMS as it is very fast and avoids MPS bugs
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.model_id = "facebook/mms-tts-hin"

    def load(self) -> None:
        """Load MMS model from local directory."""
        from transformers import VitsModel, AutoTokenizer
        import os

        # Point to the local folder
        local_dir = os.path.join("models", "tts", "mms-hin")
        
        print(f"[MMS] Loading model from {local_dir} ...")
        
        if not os.path.exists(local_dir):
            print(f"[MMS] ERROR: Folder {local_dir} not found!")
            return

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        # Force the model to Float32 and CPU for absolute stability
        self.model = VitsModel.from_pretrained(
            local_dir, 
            local_files_only=True,
            torch_dtype=torch.float32
        ).to(self.device).float()
        
        print("[MMS] Model ready and forced to Float32.")

    def synthesize(self, text: str, voice_id: str = "default") -> bytes:
        """Convert text to audio using MMS (VITS)."""
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # input_ids MUST be Long, other tensors (attention_mask) stay as they are
        # but all move to the CPU (self.device)
        model_inputs = {}
        for k, v in inputs.items():
            if k == "input_ids":
                model_inputs[k] = v.to(self.device).long()
            else:
                model_inputs[k] = v.to(self.device)

        with torch.inference_mode():
            # Safety check: if input_ids are empty or too short, VITS will crash
            if model_inputs["input_ids"].shape[1] < 1:
                print("[MMS] ERROR: Input text resulted in 0 tokens. Ensure you are using Hindi characters.")
                return b""
                
            try:
                output = self.model(**model_inputs).waveform
            except Exception as e:
                print(f"[MMS] Generation error: {e}")
                return b""

        # Convert tensor to numpy float32
        audio_np = output.cpu().numpy().squeeze().astype(np.float32)

        # Encode to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, samplerate=self.model.config.sampling_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()
