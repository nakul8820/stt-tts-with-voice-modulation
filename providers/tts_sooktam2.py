# providers/tts_sooktam2.py
# ─────────────────────────────────────────────────────────────────────────────
# Sooktam 2 TTS Provider  (BharatGen / bharatgenai/sooktam2 on HuggingFace)
#
# Sooktam 2 is a reference-guided, multilingual Indic TTS model.
# It needs a short reference WAV + its transcript to clone a voice, then
# synthesises any target text in that voice.
#
# Supported languages (cls_language values):
#   hindi, marathi, gujarati, tamil, telugu, kannada,
#   bengali, malayalam, odia, urdu, punjabi, english
#
# Drop this file in your existing  providers/  folder.
# Then wire it up in  core/factory.py  (see factory patch below).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Language aliases ─────────────────────────────────────────────────────────
# Maps the shorthand values you might put in config.yaml → Sooktam 2's
# internal cls_language token.
LANGUAGE_MAP: dict[str, str] = {
    "hi":      "hindi",
    "hindi":   "hindi",
    "mr":      "marathi",
    "marathi": "marathi",
    "gu":      "gujarati",
    "gujarati":"gujarati",
    "ta":      "tamil",
    "tamil":   "tamil",
    "te":      "telugu",
    "telugu":  "telugu",
    "kn":      "kannada",
    "kannada": "kannada",
    "bn":      "bengali",
    "bengali": "bengali",
    "ml":      "malayalam",
    "malayalam":"malayalam",
    "or":      "odia",
    "odia":    "odia",
    "ur":      "urdu",
    "urdu":    "urdu",
    "pa":      "punjabi",
    "punjabi": "punjabi",
    "en":      "english",
    "english": "english",
    # Hinglish — Sooktam 2 has no dedicated Hinglish token;
    # "hindi" gives the most natural result for code-mixed speech.
    "hinglish":"hindi",
}

# Default reference audio bundled with the project.
# Place a 5-10 second, 16 kHz mono WAV here — any clean Hindi speaker works.
DEFAULT_REF_WAV  = "data/ref_voices/default_hi.wav"
DEFAULT_REF_TEXT = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?"   # "Hello, how can I help you?"


class Sooktam2Provider:
    """
    Wraps  bharatgenai/sooktam2  to match the provider interface used
    throughout this project:

        provider = Sooktam2Provider(config)
        provider.load()
        wav_array, sample_rate = provider.synthesize(text, ...)
    """

    MODEL_ID = "models/sooktam2"

    # ── Constructor ──────────────────────────────────────────────────────────
    def __init__(self, config: dict):
        """
        Args:
            config: the  tts:  block from config.yaml, e.g.
                {
                  "provider":    "sooktam2",
                  "language":    "hi",          # or "hindi", "hinglish", …
                  "device":      "cpu",          # "cpu" | "cuda" | "mps"
                  "ref_wav":     "data/ref_voices/default_hi.wav",
                  "ref_text":    "नमस्ते …",
                }
        """
        self.cfg      = config
        self.model    = None
        self._loaded  = False

        # Language
        raw_lang        = config.get("language", "hi")
        self.cls_language = LANGUAGE_MAP.get(raw_lang.lower(), "hindi")

        # Device — fall back to CPU if MPS/CUDA not available
        self.device = config.get("device", "cpu")

        # Reference voice (voice-cloning anchor)
        self.ref_wav  = config.get("ref_wav",  DEFAULT_REF_WAV)
        self.ref_text = config.get("ref_text", DEFAULT_REF_TEXT)

    # ── load() ───────────────────────────────────────────────────────────────
    def load(self) -> None:
        """
        Downloads (first run) and loads the Sooktam 2 model into memory.
        Called once at app startup by  main.py → lifespan().
        """
        if self._loaded:
            logger.info("Sooktam2Provider: already loaded, skipping.")
            return

        logger.info("Sooktam2Provider: loading %s …", self.MODEL_ID)
        t0 = time.perf_counter()

        try:
            from transformers import AutoModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Run:  pip install transformers"
            ) from exc

        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,   # required — model ships custom code
        )

        # Move to requested device when the model exposes .to()
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        self._loaded = True
        elapsed = time.perf_counter() - t0
        logger.info(
            "Sooktam2Provider: ready in %.2f s  (device=%s, lang=%s)",
            elapsed, self.device, self.cls_language,
        )

    # ── synthesize() ─────────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        *,
        ref_wav:  Optional[str] = None,
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Convert  text  to a NumPy float32 audio array.

        Args:
            text:     The text you want spoken (Hindi / Hinglish / etc.)
            ref_wav:  Path to a reference WAV for voice cloning.
                      Falls back to  config.yaml → tts.ref_wav  if omitted.
            ref_text: Transcript of the reference WAV.
                      Falls back to  config.yaml → tts.ref_text  if omitted.
            language: Override the language for this request only.
                      Falls back to  config.yaml → tts.language  if omitted.

        Returns:
            (wav_array, sample_rate)
            wav_array  — np.ndarray, dtype float32, values in [-1, 1]
            sample_rate — int (typically 22050 or 24000 Hz)
        """
        if not self._loaded:
            raise RuntimeError(
                "Sooktam2Provider.load() must be called before synthesize()."
            )

        if not text or not text.strip():
            raise ValueError("synthesize() received empty text.")

        # Resolve per-call overrides → provider defaults
        _ref_wav  = ref_wav  or self.ref_wav
        _ref_text = ref_text or self.ref_text
        _lang_raw = language or self.cfg.get("language", "hi")
        _lang     = LANGUAGE_MAP.get(_lang_raw.lower(), "hindi")

        # Validate reference WAV path
        _ref_wav_path = Path(_ref_wav)
        if not _ref_wav_path.exists():
            raise FileNotFoundError(
                f"Reference WAV not found: {_ref_wav_path.resolve()}\n"
                "Create  data/ref_voices/default_hi.wav  (5-10 s, 16 kHz mono),\n"
                "or set  tts.ref_wav  in config.yaml to an existing file."
            )

        logger.debug(
            "Sooktam2Provider.synthesize | lang=%s | ref=%s | text=%.60s…",
            _lang, _ref_wav, text,
        )

        t0 = time.perf_counter()

        wav, sr, _ = self.model.infer(
            ref_file  = str(_ref_wav_path),
            ref_text  = _ref_text,
            gen_text  = text,
            tokenizer = "cls",          # character-level tokenizer used by Sooktam 2
            cls_language = _lang,
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Sooktam2Provider: synthesised %.2f s of audio in %.2f s  (RTF=%.2f)",
            len(wav) / sr, elapsed, elapsed / (len(wav) / sr),
        )

        # Normalise to float32 in [-1, 1] — the modulation pipeline expects this
        wav = np.asarray(wav, dtype=np.float32)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = wav / peak

        return wav, int(sr)

    # ── Convenience ──────────────────────────────────────────────────────────
    @property
    def sample_rate(self) -> int:
        """Return model sample rate (available after load())."""
        if self.model is None:
            raise RuntimeError("Call load() first.")
        # Sooktam 2 emits 24 kHz; fall back gracefully if attr missing
        return getattr(self.model, "sample_rate", 24000)

    def __repr__(self) -> str:
        return (
            f"Sooktam2Provider("
            f"loaded={self._loaded}, "
            f"lang={self.cls_language}, "
            f"device={self.device})"
        )
