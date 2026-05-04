# modules/audio_stitcher.py
# ─────────────────────────────────────────────────────────────────────────────
# Audio Stitcher — 3 strategies
#
# strategy="raw"
#   Trim silence edges only. No volume matching. You hear the raw
#   voice difference between models.
#
# strategy="normalized"
#   Trim silence + RMS volume normalization. Ensures both models
#   output at the same perceived loudness. Reduces jarring volume
#   jumps at language boundaries.
#
# strategy="tuned"
#   Used in conjunction with VITS parameter tuning in mms_engine.py.
#   Here we additionally apply spectral centroid matching — shifts the
#   timbral brightness of each chunk toward a common target so the
#   two voices blend better perceptually.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

TARGET_RMS        = 0.15
SILENCE_THRESHOLD = 0.01
INTER_WORD_GAP_S  = 0.08    # gap between same-language segments
LANG_SWITCH_GAP_S = 0.10    # slightly longer at Hindi↔English boundaries


# ── Primitives ─────────────────────────────────────────────────────────────────

def _make_silence(duration_s: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _trim_silence(wav: np.ndarray, sr: int = 16000, pad_ms: int = 10) -> np.ndarray:
    if len(wav) == 0:
        return wav
    pad = int(pad_ms / 1000 * sr)
    above = np.where(np.abs(wav) > SILENCE_THRESHOLD)[0]
    if len(above) == 0:
        return wav
    return wav[max(0, above[0] - pad): min(len(wav), above[-1] + pad + 1)]


def _rms_normalize(wav: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    if len(wav) == 0:
        return wav
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-6:
        return wav
    return np.clip(wav * (target / rms), -1.0, 1.0).astype(np.float32)


def _spectral_centroid(wav: np.ndarray, sr: int = 16000) -> float:
    """Estimate spectral centroid (brightness) of a waveform."""
    if len(wav) < 512:
        return 0.0
    spectrum = np.abs(np.fft.rfft(wav[:4096]))
    freqs = np.fft.rfftfreq(min(len(wav), 4096), d=1.0/sr)
    total = np.sum(spectrum)
    if total < 1e-6:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def _match_brightness(wav: np.ndarray, source_centroid: float, target_centroid: float) -> np.ndarray:
    """
    Nudge the spectral brightness of wav toward target_centroid.
    Uses a simple shelving filter approximation via FFT.
    Only applies a subtle correction — not full voice matching.
    """
    if len(wav) < 512 or abs(source_centroid - target_centroid) < 50:
        return wav  # negligible difference, skip

    ratio = target_centroid / max(source_centroid, 1.0)
    # Clamp correction to avoid over-processing
    ratio = np.clip(ratio, 0.7, 1.4)

    # Apply in frequency domain
    fft = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(len(wav))
    # Simple high-shelf: scale frequencies above midpoint
    mid = 0.3
    scale = np.where(freqs > mid, ratio, 1.0)
    corrected = np.fft.irfft(fft * scale, n=len(wav))
    return np.clip(corrected, -1.0, 1.0).astype(np.float32)


# ── Main stitch function ───────────────────────────────────────────────────────

def stitch(
    chunks: List[Tuple[np.ndarray, str]],
    strategy: str = "normalized",
    sr: int = 16000,
) -> np.ndarray:
    """
    Stitch audio chunks from multiple MMS model outputs.

    Args:
        chunks:   List of (waveform, lang) — lang is "hi" or "en"
        strategy: One of:
                    "raw"        — trim silence only
                    "normalized" — trim + RMS normalize
                    "tuned"      — trim + RMS normalize + brightness matching
        sr:       Sample rate (must be same for all chunks)

    Returns:
        Single float32 waveform
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    # Step 1: Trim silence from all chunks
    processed = []
    for wav, lang in chunks:
        if wav is None or len(wav) == 0:
            continue
        wav = _trim_silence(wav, sr=sr)
        if len(wav) > 0:
            processed.append((wav, lang))

    if not processed:
        return np.zeros(0, dtype=np.float32)

    if len(processed) == 1:
        wav, _ = processed[0]
        if strategy in ("normalized", "tuned"):
            wav = _rms_normalize(wav)
        return wav

    # Step 2: Strategy-specific processing
    if strategy == "raw":
        final_chunks = processed

    elif strategy == "normalized":
        final_chunks = [(_rms_normalize(wav), lang) for wav, lang in processed]

    elif strategy == "tuned":
        # RMS normalize first
        normed = [(_rms_normalize(wav), lang) for wav, lang in processed]

        # Compute centroids per language — average across all chunks of that lang
        centroids = {"hi": [], "en": []}
        for wav, lang in normed:
            c = _spectral_centroid(wav, sr)
            if c > 0:
                centroids[lang].append(c)

        avg_hi = float(np.mean(centroids["hi"])) if centroids["hi"] else 0
        avg_en = float(np.mean(centroids["en"])) if centroids["en"] else 0

        # Target centroid: weighted average of both languages
        all_cents = centroids["hi"] + centroids["en"]
        target_centroid = float(np.mean(all_cents)) if all_cents else 0

        if target_centroid > 0 and avg_hi > 0 and avg_en > 0:
            final_chunks = []
            for wav, lang in normed:
                src = avg_hi if lang == "hi" else avg_en
                wav = _match_brightness(wav, src, target_centroid)
                final_chunks.append((wav, lang))
        else:
            final_chunks = normed

    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use raw/normalized/tuned")

    # Step 3: Concatenate with natural gaps
    parts = []
    for i, (wav, lang) in enumerate(final_chunks):
        parts.append(wav)
        if i < len(final_chunks) - 1:
            next_lang = final_chunks[i + 1][1]
            gap_s = LANG_SWITCH_GAP_S if lang != next_lang else INTER_WORD_GAP_S
            parts.append(_make_silence(gap_s, sr))

    return np.concatenate(parts).astype(np.float32)
