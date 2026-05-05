#!/usr/bin/env python3
# scripts/make_ref_voice.py
# ─────────────────────────────────────────────────────────────────────────────
# ONE-TIME SETUP SCRIPT
#
# Generates  data/ref_voices/default_hi.wav  — the default reference voice
# that Sooktam 2 uses to clone a speaker when no ref_wav is sent via the API.
#
# This script uses gTTS (Google Text-to-Speech) to produce a short Hindi
# utterance, resamples it to 16 kHz mono, and saves it as a WAV.
# The result is a plain, neutral Hindi voice — good enough as a default.
#
# For BEST RESULTS: record your own 5-10 second WAV and replace the file.
#
# Usage:
#   pip install gtts soundfile scipy
#   python scripts/make_ref_voice.py
# ─────────────────────────────────────────────────────────────────────────────

import io
import sys
from pathlib import Path

TARGET_PATH = Path("data/ref_voices/default_hi.wav")
TARGET_SR   = 16_000   # Sooktam 2 works best with 16 kHz reference audio
REF_TEXT    = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?"


def main() -> None:
    try:
        from gtts import gTTS
    except ImportError:
        sys.exit("Install gTTS first:  pip install gtts")

    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
    except ImportError:
        sys.exit("Install deps first:  pip install numpy soundfile scipy")

    print(f"Generating reference voice → {TARGET_PATH}")

    # ── Step 1: synthesise Hindi speech with gTTS ─────────────────────────
    tts = gTTS(text=REF_TEXT, lang="hi", slow=False)
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)

    # ── Step 2: decode MP3 → numpy array ─────────────────────────────────
    # soundfile can't read MP3 directly; use pydub if available, else scipy
    try:
        from pydub import AudioSegment  # type: ignore
        seg = AudioSegment.from_mp3(mp3_buf)
        seg = seg.set_channels(1)           # mono
        raw = np.array(seg.get_array_of_samples(), dtype=np.float32)
        raw /= 32768.0                      # int16 → float32 [-1, 1]
        src_sr = seg.frame_rate
    except ImportError:
        # Fallback: write MP3 to disk, use ffmpeg via subprocess
        import subprocess, tempfile, os
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_mp3.write(mp3_buf.read())
        tmp_mp3.close()
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3.name,
             "-ac", "1", "-ar", str(TARGET_SR), tmp_wav.name],
            check=True, capture_output=True,
        )
        raw, src_sr = sf.read(tmp_wav.name, dtype="float32")
        os.unlink(tmp_mp3.name)
        os.unlink(tmp_wav.name)
        # Already resampled by ffmpeg — skip scipy step
        TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(TARGET_PATH), raw, TARGET_SR)
        print(f"✅  Saved: {TARGET_PATH}  ({len(raw)/TARGET_SR:.1f} s, {TARGET_SR} Hz mono)")
        return

    # ── Step 3: resample to 16 kHz ────────────────────────────────────────
    if src_sr != TARGET_SR:
        g   = gcd(TARGET_SR, src_sr)
        up  = TARGET_SR // g
        dn  = src_sr   // g
        raw = resample_poly(raw, up, dn).astype(np.float32)

    # ── Step 4: save as WAV ───────────────────────────────────────────────
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(TARGET_PATH), raw, TARGET_SR)

    duration = len(raw) / TARGET_SR
    print(f"✅  Saved: {TARGET_PATH}  ({duration:.1f} s, {TARGET_SR} Hz mono)")
    print()
    print("TIP: For better voice cloning quality, replace this file with")
    print("     a 5-10 s recording of your own voice (16 kHz, mono WAV).")
    print(f"     Keep the transcript in config.yaml → tts.ref_text in sync.")


if __name__ == "__main__":
    main()
