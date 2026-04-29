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
        self.model = None
        self.config = None

        # voice_profiles maps voice_id → path to reference .wav
        # "default" = None means use XTTS built-in default speaker
        self.voice_profiles = {
            "default": None,
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

        # Find where Coqui cached the model after download_models.py ran
        manager = ModelManager()
        model_path, config_path, _ = manager.download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        # model_path = directory containing model checkpoint files
        # config_path = path to config.json

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

        print("[CoquiXTTS] XTTS v2 loaded.")

    def add_voice_profile(self, voice_id: str, wav_path: str) -> None:
        """
        Register a new voice profile (reference audio file).
        Called by the admin endpoint when a new voice is uploaded.
        """
        self.voice_profiles[voice_id] = wav_path
        print(f"[CoquiXTTS] Voice profile added: {voice_id} → {wav_path}")

    def synthesize(self, text: str, voice_id: str = "default") -> bytes:
        """
        Generate speech using model.inference() (low-level API).
        This is the only XTTS API that accepts speed + temperature.

        Steps:
        1. Get Layer 1 params from modulation config
        2. Get speaker conditioning latents (voice identity)
        3. Run inference with all params
        4. Return WAV bytes
        """

        # ── Step 1: Get Layer 1 params from modulation config ──
        # These are read fresh on every call so admin changes
        # take effect without restarting the server.
        layer1 = get_layer1_params()
        temperature = layer1["temperature"]   # 0.1–1.0
        speed       = layer1["speed"]         # 0.5–2.0

        # ── Step 2: Get speaker conditioning latents ───────────
        # Conditioning latents encode the voice identity from
        # the reference audio. They tell the model "sound like this."
        ref_wav = self.voice_profiles.get(voice_id)
        # ref_wav is None for "default", or a .wav path for cloned voices

        if ref_wav is None:
            # No reference audio — use a short silence as placeholder.
            # XTTS will use its built-in default speaker characteristics.
            import tempfile, os
            silence = np.zeros(int(0.5 * 22050), dtype=np.float32)
            # 0.5 seconds of silence at 22050Hz (XTTS input sample rate)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, silence, 22050)
            tmp.close()
            ref_paths = [tmp.name]
            cleanup_tmp = True
        else:
            ref_paths = [ref_wav]
            cleanup_tmp = False

        try:
            # get_conditioning_latents encodes the reference audio
            # into two tensors that capture speaker identity:
            # gpt_cond_latent: GPT conditioning (prosody/rhythm)
            # speaker_embedding: speaker identity vector
            gpt_cond_latent, speaker_embedding = \
                self.model.get_conditioning_latents(audio_path=ref_paths)

            # ── Step 3: Run inference ───────────────────────────
            out = self.model.inference(
                text=text,
                language="hi",
                # "hi" handles Hinglish — XTTS detects English
                # words within Hindi context automatically

                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,

                # ── Layer 1 modulation params ──
                temperature=temperature,
                # Low = deterministic/flat, High = varied/expressive

                speed=speed,
                # Affects token generation rate (rhythm at model level)

                repetition_penalty=2.0,
                # Prevents model from repeating phonemes/words.
                # 2.0 is a good default. Higher = stricter.
            )

        finally:
            if cleanup_tmp:
                os.unlink(ref_paths[0])

        # out["wav"] is a numpy array of float32 audio samples
        audio_np = np.array(out["wav"], dtype=np.float32)

        # Encode to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, samplerate=24000, format="WAV")
        # XTTS v2 always outputs at 24000 Hz
        buffer.seek(0)
        return buffer.read()