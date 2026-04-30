# providers/tts/coqui_xtts.py
# ─────────────────────────────────────────────────────────────
# CHANGES FROM ORIGINAL:
#   1. synthesize() now calls modulation.get_layer1_params()
#      to get temperature + speed before calling model.inference()
#   2. Uses model.inference() directly instead of tts.tts_to_file()
#      because tts_to_file() doesn't support speed/temperature params
#   3. Everything else identical to original
# ─────────────────────────────────────────────────────────────

import io
import torch
import numpy as np
import soundfile as sf

from core.base_tts import BaseTTSProvider
# Import the modulation module so we can read Layer 1 params
from modules.modulation import get_layer1_params


class CoquiXTTSProvider(BaseTTSProvider):

    def __init__(self, cfg: dict):
        self.device = cfg.get("device", "cpu")
        self.voice_id = cfg.get("voice_id", "default")
        self.language = cfg.get("language", "hi")
        self.model = None
        self.config = None

        # voice_profiles maps voice_id → path to reference .wav
        # Use the local reference.wav for the default voice
        self.voice_profiles = {
            "default": "reference.wav",
        }

    def load(self) -> None:
        """
        Load XTTS v2 using the low-level API (XttsConfig + Xtts)
        instead of the high-level TTS() wrapper.
        We need the low-level API to pass speed + temperature
        at inference time — the high-level wrapper doesn't support them.
        """
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        if self.device == "mps" and not torch.backends.mps.is_available():
            print("[CoquiXTTS] MPS not available, falling back to CPU")
            self.device = "cpu"

        print(f"[CoquiXTTS] Loading XTTS v2 on {self.device} ...")

        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        # 1. Determine local paths
        # We expect it in: models/tts/tts_models--multilingual--multi-dataset--xtts_v2
        import os
        tts_home = os.environ.get("TTS_HOME")
        if not tts_home:
            # Fallback if environment variable not set
            project_root = os.getcwd()
            tts_home = os.path.join(project_root, "models", "tts")
            
        local_model_dir = os.path.join(tts_home, model_name.replace("/", "--"))
        config_path = os.path.join(local_model_dir, "config.json")
        
        if os.path.exists(config_path):
            print(f"[CoquiXTTS] Found local model files at {local_model_dir}")
            model_path = local_model_dir
        else:
            print(f"[CoquiXTTS] Local model not found at {local_model_dir}. Attempting download...")
            try:
                manager = ModelManager()
                model_path, config_path, _ = manager.download_model(model_name)
            except Exception as e:
                print(f"[CoquiXTTS] Error: Could not download or find model: {e}")
                print("[CoquiXTTS] Please run 'python download_models.py' while online.")
                # If we fail here, the app might crash later if self.model is None.
                # But at least we show a clear error message.
                raise

        # Load config from the cached directory
        self.xtts_config = XttsConfig()
        self.xtts_config.load_json(config_path)

        # Load the model from the checkpoint
        self.model = Xtts.init_from_config(self.xtts_config)
        self.model.load_checkpoint(
            self.xtts_config,
            checkpoint_dir=model_path,
            eval=True
            # eval=True puts model in inference mode (disables dropout etc.)
        )
        self.model.to(self.device)

        # ── OPTIMIZATION: Pre-compute speaker latents ──
        # We compute the "voice identity" once during startup.
        # This saves 1-2 seconds per request.
        print(f"[CoquiXTTS] Pre-computing speaker latents for {self.voice_profiles['default']}...")
        ref_path = self.voice_profiles.get("default")
        if ref_path and os.path.exists(ref_path):
            self.default_gpt_cond, self.default_speaker_embedding = \
                self.model.get_conditioning_latents(audio_path=[ref_path])
        else:
            speaker_name = "Daisy Studious"
            self.default_gpt_cond = self.model.speaker_manager.speakers[speaker_name]["gpt_cond_latent"]
            self.default_speaker_embedding = self.model.speaker_manager.speakers[speaker_name]["speaker_embedding"]

        print("[CoquiXTTS] XTTS v2 loaded and optimized.")

    def add_voice_profile(self, voice_id: str, wav_path: str) -> None:
        """
        Register a new voice profile (reference audio file).
        Called by the admin endpoint when a new voice is uploaded.
        """
        self.voice_profiles[voice_id] = wav_path
        print(f"[CoquiXTTS] Voice profile added: {voice_id} → {wav_path}")

    def synthesize(self, text: str, voice_id: str = "default") -> bytes:
        """
        Generate speech using optimized inference.
        """
        layer1 = get_layer1_params()
        temperature = layer1["temperature"]
        speed       = layer1["speed"]

        print(f"[CoquiXTTS] Synthesizing: '{text}' (temp={temperature}, speed={speed})")

        # ── OPTIMIZATION: Use torch.inference_mode() ──
        with torch.inference_mode():
            out = self.model.inference(
                text=text,
                language=self.language,
                gpt_cond_latent=self.default_gpt_cond,
                speaker_embedding=self.default_speaker_embedding,
                temperature=temperature,
                speed=speed,
                repetition_penalty=2.0,
            )

        # out["wav"] is a numpy array of float32 audio samples
        audio_np = np.array(out["wav"], dtype=np.float32)

        # Encode to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, samplerate=24000, format="WAV")
        # XTTS v2 always outputs at 24000 Hz
        buffer.seek(0)
        return buffer.read()