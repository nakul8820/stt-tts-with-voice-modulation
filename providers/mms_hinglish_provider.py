# providers/mms_hinglish_provider.py
# ─────────────────────────────────────────────────────────────────────────────
# MMS Hinglish TTS Provider
#
# Pipeline:
#   Romanized Hinglish text
#     → Aksharantar lexicon (optional) + IndicXlit ± phoneme English → Devanagari
#     → Facebook MMS-TTS Hindi (VITS)      → audio waveform
#
# Interface mirrors the experiment runner expectations:
#   provider.load()
#   waveform, sample_rate = provider.synthesize(text)
#   provider._transliterator.transliterate(text)  # for inspection
#   provider.MMS_MODEL_ID                          # for report
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import logging
from typing import Optional
import numpy as np
import torch

import io
import soundfile as sf
from modules.hinglish_transliterator import HinglishTransliterator
from core.base_tts import BaseTTSProvider

logger = logging.getLogger(__name__)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_project_path(rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    return os.path.normpath(os.path.join(_project_root(), rel_or_abs))


class MmsHinglishProvider(BaseTTSProvider):
    """
    Full Hinglish TTS pipeline:
      Romanized text → Devanagari (IndicXlit) → Audio (MMS-TTS Hindi)

    Usage:
        provider = MmsHinglishProvider()
        provider.load()
        waveform, sample_rate = provider.synthesize("aaj mera meeting hai")
    """

    # Public attribute used by run_experiment.py for the report
    MMS_MODEL_ID = "facebook/mms-tts-hin"

    def __init__(self, cfg: dict = None):
        """
        Args:
            cfg: Configuration dict from config.yaml.
        """
        model_dir = cfg.get("model_dir") if cfg else None
        
        # Resolve model directory
        if model_dir:
            self._model_dir = model_dir
        else:
            # Try the path where download_models.py puts it
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Support both directory names used across conversations
            for candidate in ["mms-tts-hin", "mms-hin"]:
                path = os.path.join(project_root, "models", "tts", candidate)
                if os.path.isdir(path):
                    self._model_dir = path
                    break
            else:
                # Fallback — will use HuggingFace hub cache
                self._model_dir = self.MMS_MODEL_ID

        # Internal state
        self._model = None
        self._tokenizer = None
        self._sample_rate = None
        self._device = "cpu"  # MMS is fast enough on CPU; avoids MPS bugs

        phoneme_mode = (cfg.get("english_phoneme_mode") if cfg else None) or "phoneme"

        aksh_path: Optional[str] = None
        ak_cfg = (cfg.get("aksharantar_lexicon") or {}) if cfg else {}
        if ak_cfg.get("enabled", False):
            raw_path = ak_cfg.get("path") or "data/lexicons/aksharantar_hi.tsv"
            resolved = _resolve_project_path(str(raw_path))
            if os.path.isfile(resolved):
                aksh_path = resolved
                logger.info("Aksharantar lexicon enabled: %s", resolved)
            else:
                logger.warning(
                    "aksharantar_lexicon.enabled is true but file missing: %s",
                    resolved,
                )

        self._transliterator = HinglishTransliterator()
        self._speed = 1.0

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load MMS model + IndicXlit transliterator into memory."""
        # 1. Load transliterator
        logger.info("Loading HinglishTransliterator (IndicXlit)...")
        self._transliterator.load()

        # 2. Load MMS-TTS Hindi
        logger.info(f"Loading MMS-TTS Hindi from: {self._model_dir}")
        from transformers import VitsModel, AutoTokenizer

        local_files_only = os.path.isdir(self._model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_dir,
            local_files_only=local_files_only
        )
        self._model = VitsModel.from_pretrained(
            self._model_dir,
            local_files_only=local_files_only,
            torch_dtype=torch.float32,
        ).to(self._device).float()
        self._model.eval()

        self._sample_rate = self._model.config.sampling_rate
        logger.info(f"MMS-TTS Hindi ready — sample rate: {self._sample_rate}Hz")

    def synthesize(self, text: str, voice_id: str = "default") -> bytes:
        """
        Synthesize romanized Hinglish to audio.

        Args:
            text: Romanized Hinglish string, e.g. "aaj mera meeting hai"
            voice_id: Ignored (MMS only has one voice).

        Returns:
            Raw WAV audio as bytes.
        """
        if self._model is None:
            raise RuntimeError("Call load() before synthesize()")

        # Step 1: Romanized → Devanagari
        devanagari = self._transliterator.transliterate(text)
        logger.info("[TTS Transliteration] '%s' -> '%s'", text, devanagari)
        print(f"[TTS Transliteration] input='{text}' output='{devanagari}'")

        # Step 2: Devanagari → audio
        inputs = self._tokenizer(devanagari, return_tensors="pt")
        model_inputs = {}
        for k, v in inputs.items():
            if k == "input_ids":
                model_inputs[k] = v.to(self._device).long()
            else:
                model_inputs[k] = v.to(self._device)

        if model_inputs["input_ids"].shape[1] < 1:
            logger.error("Tokenizer produced 0 tokens — check Devanagari output")
            buf = io.BytesIO()
            sf.write(
                buf,
                np.zeros(self._sample_rate, dtype=np.float32),
                samplerate=self._sample_rate,
                format="WAV",
            )
            buf.seek(0)
            return buf.read()

        with torch.inference_mode():
            output = self._model(**model_inputs).waveform

        waveform = output.cpu().numpy().squeeze().astype(np.float32)

        # Optional speed adjustment via simple resampling
        if abs(self._speed - 1.0) > 0.01:
            waveform = self._apply_speed(waveform, self._speed)

        # Encode to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, waveform, samplerate=self._sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def set_speed(self, speed: float) -> None:
        """Set playback speed (0.5 = slow, 1.0 = normal, 1.5 = fast)."""
        self._speed = max(0.25, min(3.0, speed))

    def set_pitch(self, semitones: int) -> None:
        """
        Set pitch shift in semitones. Requires librosa.
        Stored for use in synthesize() — not yet applied by default.
        """
        self._pitch_semitones = semitones

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_speed(self, waveform: np.ndarray, speed: float) -> np.ndarray:
        """Naive speed change via linear resampling (no pitch change)."""
        try:
            import scipy.signal as signal
            target_len = int(len(waveform) / speed)
            resampled = signal.resample(waveform, target_len)
            return resampled.astype(np.float32)
        except ImportError:
            logger.warning("scipy not installed — speed adjustment skipped")
            return waveform
