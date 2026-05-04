# providers/mms_hinglish_provider.py
# ─────────────────────────────────────────────────────────────────────────────
# MMS Hinglish TTS Provider
#
# Plugs into your existing factory/router architecture.
# Implements the same interface as CoquiXTTSProvider and ParlerTTSProvider
# so your factory.py can load it with zero changes to routing logic.
#
# Pipeline:
#   romanized Hinglish text
#       ↓
#   HinglishTransliterator (IndicXlit) → Devanagari text
#       ↓
#   Facebook MMS-TTS Hindi (facebook/mms-tts-hin) → audio waveform
#       ↓
#   numpy float32 array at 16kHz
#
# Integration (when you're ready):
#   1. Copy this file into your providers/ folder
#   2. Copy modules/hinglish_transliterator.py into your modules/ folder
#   3. In core/factory.py, add:
#        from providers.mms_hinglish_provider import MmsHinglishProvider
#        "mms_hinglish": MmsHinglishProvider
#   4. In config.yaml, set:
#        tts:
#          provider: mms_hinglish
#   That's it. No router changes needed.
# ─────────────────────────────────────────────────────────────────────────────

import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class MmsHinglishProvider:
    """
    TTS provider for romanized Hinglish input using Facebook MMS + IndicXlit.

    Follows the same load() / synthesize() interface as other providers
    in this project so it drops into factory.py without changes.
    """

    # Model identifier on HuggingFace Hub
    MMS_MODEL_ID = "facebook/mms-tts-hin"

    # MMS-Hindi outputs at 16kHz — confirmed from model config
    SAMPLE_RATE = 16000

    def __init__(self, model_dir: Optional[str] = None):
        """
        Args:
            model_dir: Optional local path to cached model weights.
                       If None, HuggingFace cache (~/.cache/huggingface) is used.
                       Set this to your project's models/ directory to keep
                       everything local and offline.
        """
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None
        self._transliterator = None
        self._loaded = False

        # Voice modulation parameters (matches your project's modulation concept)
        # These are post-processing adjustments applied to the raw waveform
        self.speed_factor: float = 1.0   # 0.5 = slower, 2.0 = faster
        self.pitch_shift: int = 0         # semitones, -12 to +12

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load MMS-TTS-Hindi model and IndicXlit transliterator into memory.
        Called once at startup by your router's init() function.
        """
        if self._loaded:
            return

        logger.info("Loading MMS Hinglish TTS provider...")
        total_start = time.time()

        # 1. Load transliterator
        logger.info("  [1/2] Loading IndicXlit transliterator...")
        from modules.hinglish_transliterator import HinglishTransliterator
        self._transliterator = HinglishTransliterator(model_dir=self.model_dir)
        self._transliterator.load()

        # 2. Load MMS-TTS Hindi model
        logger.info("  [2/2] Loading facebook/mms-tts-hin...")
        t = time.time()
        try:
            from transformers import VitsModel, AutoTokenizer
            import torch

            model_path = self.model_dir or self.MMS_MODEL_ID
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = VitsModel.from_pretrained(model_path)
            self._model.eval()  # inference mode — disables dropout

            # Enable MPS (Apple Silicon) if available, otherwise CPU
            if torch.backends.mps.is_available():
                self._model = self._model.to("mps")
            else:
                self._model = self._model.to("cpu")

            logger.info(f"  MMS model loaded in {time.time() - t:.2f}s")

        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch"
            )

        self._loaded = True
        logger.info(
            f"MMS Hinglish provider ready. "
            f"Total load time: {time.time() - total_start:.2f}s"
        )

    # ── Core synthesis ─────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Convert romanized Hinglish text to speech.

        This is the method your tts_router.py calls — same signature
        as other providers in the project.

        Args:
            text: Romanized Hinglish string.
                  e.g. "aaj mera meeting 3 baje hai"

        Returns:
            Tuple of (waveform, sample_rate) where:
              waveform    — numpy float32 array, shape (N,)
              sample_rate — int, always 16000 for MMS-Hindi
        """
        if not self._loaded:
            raise RuntimeError("Provider not loaded. Call load() first.")

        if not text or not text.strip():
            # Return 0.5s of silence for empty input
            return np.zeros(self.SAMPLE_RATE // 2, dtype=np.float32), self.SAMPLE_RATE

        start = time.time()

        # Step 1: Transliterate romanized Hinglish → Devanagari
        t1 = time.time()
        devanagari_text = self._transliterator.transliterate(text)
        transliterate_ms = (time.time() - t1) * 1000
        logger.debug(f"Transliterated in {transliterate_ms:.1f}ms: '{text}' → '{devanagari_text}'")

        # Step 2: Tokenize for MMS
        t2 = time.time()
        import torch
        inputs = self._tokenizer(devanagari_text, return_tensors="pt")
        
        # Move inputs to the same device as the model (MPS or CPU)
        model_inputs = {}
        for k, v in inputs.items():
            if k == "input_ids":
                model_inputs[k] = v.to(self._model.device).long()
            else:
                model_inputs[k] = v.to(self._model.device)

        # Step 3: Synthesize
        with torch.no_grad():
            output = self._model(**model_inputs)

        waveform = output.waveform.squeeze().cpu().numpy().astype(np.float32)
        synth_ms = (time.time() - t2) * 1000

        # Step 4: Optional post-processing (speed / pitch)
        waveform = self._apply_modulation(waveform)

        total_ms = (time.time() - start) * 1000
        audio_duration_ms = (len(waveform) / self.SAMPLE_RATE) * 1000
        rtf = total_ms / audio_duration_ms if audio_duration_ms > 0 else 0

        logger.info(
            f"Synthesized {len(text)} chars | "
            f"translit={transliterate_ms:.0f}ms | "
            f"synth={synth_ms:.0f}ms | "
            f"total={total_ms:.0f}ms | "
            f"RTF={rtf:.3f} | "
            f"audio={audio_duration_ms:.0f}ms"
        )

        return waveform, self.SAMPLE_RATE

    # ── Voice modulation ───────────────────────────────────────────────────────

    def _apply_modulation(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply speed and pitch adjustments to the raw waveform.
        These are the 'voice modulation' controls your project supports.

        Speed: resampling via scipy (keeps duration proportional)
        Pitch: shift via librosa if available, otherwise skipped silently
        """
        # Speed adjustment via resampling
        if abs(self.speed_factor - 1.0) > 0.01:
            try:
                from scipy.signal import resample
                target_length = int(len(waveform) / self.speed_factor)
                waveform = resample(waveform, target_length).astype(np.float32)
            except ImportError:
                logger.warning("scipy not available — speed modulation skipped")

        # Pitch shift (optional, requires librosa)
        if self.pitch_shift != 0:
            try:
                import librosa
                waveform = librosa.effects.pitch_shift(
                    waveform,
                    sr=self.SAMPLE_RATE,
                    n_steps=self.pitch_shift
                ).astype(np.float32)
            except ImportError:
                logger.warning("librosa not available — pitch modulation skipped")

        return waveform

    def set_speed(self, factor: float) -> None:
        """Set speech speed. 1.0 = normal, 0.8 = slow, 1.3 = fast."""
        self.speed_factor = max(0.5, min(2.0, factor))

    def set_pitch(self, semitones: int) -> None:
        """Set pitch shift in semitones. 0 = no change, +2 = slightly higher."""
        self.pitch_shift = max(-12, min(12, semitones))

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def get_info(self) -> dict:
        """Returns provider metadata — useful for your /api/stats endpoint."""
        return {
            "provider": "mms_hinglish",
            "tts_model": self.MMS_MODEL_ID,
            "transliterator": "ai4bharat/IndicXlit (hindi)",
            "sample_rate": self.SAMPLE_RATE,
            "language": "Hinglish (romanized → Devanagari → speech)",
            "speed_factor": self.speed_factor,
            "pitch_shift": self.pitch_shift,
            "loaded": self._loaded,
        }
