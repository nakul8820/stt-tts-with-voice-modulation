# Modulatin Pipeline
# modules/modulation.py
# ─────────────────────────────────────────────────────────────
# The modulation pipeline.
# Takes raw WAV bytes from any TTS provider and applies
# post-processing effects based on the admin's config.
#
# Also exposes get_layer1_params() so the TTS provider can
# pull temperature and speed values before generation.
#
# This is Layer 2. Layer 1 happens inside the provider.
# Both read from the same modulation: section of config.yaml.
#
# Dependencies:
#   pip install librosa soundfile numpy scipy noisereduce
# ─────────────────────────────────────────────────────────────

import io
import numpy as np
import soundfile as sf
import yaml

# We import librosa and scipy lazily (inside functions) so that
# if modulation is disabled, we don't pay the import cost.


def _load_mod_cfg() -> dict:
    """
    Load only the modulation: section from config.yaml.
    Called on every request so config changes take effect
    without restarting the server.
    """
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("modulation", {})


def get_layer1_params() -> dict:
    """
    Called by the TTS provider BEFORE generating audio.
    Returns the parameters that must be passed INTO the model
    at inference time (Layer 1 — neural-level control).

    Returns dict with:
        temperature: float  (XTTS expressiveness)
        speed:       float  (XTTS native speed)
    """
    cfg = _load_mod_cfg()

    # Map admin's expressiveness slider (1–10) → XTTS temperature (0.1–1.0)
    # Formula: temperature = expressiveness / 10
    # 1  → 0.1 (very flat/robotic)
    # 5  → 0.5 (neutral, natural)
    # 10 → 1.0 (very expressive/emotional)
    expressiveness = cfg.get("expressiveness", 5)
    temperature = round(expressiveness / 10, 2)

    speed = cfg.get("speed_model", 1.0)

    return {
        "temperature": temperature,
        "speed": speed
    }


def _wav_bytes_to_numpy(wav_bytes: bytes) -> tuple:
    """
    Convert raw WAV bytes → (numpy float32 array, sample_rate).
    Used at the start of the pipeline before any processing.
    """
    buffer = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buffer, dtype="float32")
    # sf.read returns (samples, samplerate)
    # dtype="float32" gives us values in range [-1.0, 1.0]
    return audio, sr


def _numpy_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """
    Convert (numpy float32 array, sample_rate) → raw WAV bytes.
    Used at the end of the pipeline to package the result.
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()


def _apply_pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Shift pitch up or down by N semitones without changing duration.
    Uses librosa's phase vocoder — high quality, no chipmunk effect.

    semitones: float
        0  = no change
        +4 = higher (e.g. female-ish from male)
        -4 = deeper (e.g. male-ish from female)
        ±12 = one full octave

    Internally uses STFT + phase vocoder to shift frequency bins
    while keeping the time envelope the same.
    """
    if semitones == 0:
        return audio   # skip processing entirely if no change needed

    import librosa
    return librosa.effects.pitch_shift(
        audio,
        sr=sr,
        n_steps=semitones,
        # bins_per_octave=12 is default — 1 step = 1 semitone
    )


def _apply_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Speed up or slow down audio WITHOUT changing pitch.
    This is different from XTTS speed= which affects generation rhythm.

    rate: float
        1.0 = no change
        1.5 = 50% faster (shorter duration, same pitch)
        0.7 = 30% slower (longer duration, same pitch)

    Uses librosa phase vocoder (same algorithm as pitch_shift internally).
    """
    if rate == 1.0:
        return audio

    import librosa
    return librosa.effects.time_stretch(audio, rate=rate)


def _apply_volume(audio: np.ndarray, gain: float) -> np.ndarray:
    """
    Multiply audio amplitude by gain factor.
    gain 1.0 = no change, 1.5 = 50% louder, 0.5 = 50% quieter.

    We clip to [-1, 1] after to prevent distortion if gain > 1.0
    would push samples beyond the float32 valid range.
    """
    if gain == 1.0:
        return audio

    amplified = audio * gain
    return np.clip(amplified, -1.0, 1.0)
    # np.clip(array, min, max) — any value outside range is clamped


def _apply_eq(audio: np.ndarray, sr: int, bass: int, treble: int) -> np.ndarray:
    """
    Simple 2-band EQ: bass (low shelf) and treble (high shelf).

    bass:   int -5 to +5  → warmth/fullness of voice
    treble: int -5 to +5  → clarity/brightness of voice

    Implemented using scipy Butterworth IIR filters.
    A "shelf" filter boosts/cuts all frequencies below (low shelf)
    or above (high shelf) a cutoff frequency.
    """
    if bass == 0 and treble == 0:
        return audio   # skip if no EQ needed

    from scipy import signal

    # --- Low shelf (bass) at 200Hz ---
    if bass != 0:
        # Map admin value -5→+5 to gain multiplier
        # +5 = 2.0x (6dB boost), -5 = 0.5x (6dB cut)
        bass_gain = 1.0 + (bass * 0.2)
        # 0.2 per step means ±1.0 range = ±5 steps

        # Butterworth lowpass filter as the "shelf" basis
        # nyquist = half the sample rate (Nyquist theorem)
        nyquist = sr / 2.0
        cutoff = 200.0 / nyquist   # normalize cutoff to 0-1 range

        # butter() designs a Butterworth IIR filter
        # N=2 = 2nd order (gentle slope), btype="low" = lowpass
        b, a = signal.butter(N=2, Wn=cutoff, btype="low")

        # lfilter applies the filter to the signal
        low_band = signal.lfilter(b, a, audio)

        # Blend: boosted low band + original high band
        audio = audio + low_band * (bass_gain - 1.0)
        audio = np.clip(audio, -1.0, 1.0)

    # --- High shelf (treble) at 4000Hz ---
    if treble != 0:
        treble_gain = 1.0 + (treble * 0.2)
        nyquist = sr / 2.0
        cutoff = 4000.0 / nyquist
        cutoff = min(cutoff, 0.99)
        # Clamp to 0.99 — scipy requires Wn < 1.0 for lowpass

        b, a = signal.butter(N=2, Wn=cutoff, btype="high")
        high_band = signal.lfilter(b, a, audio)

        audio = audio + high_band * (treble_gain - 1.0)
        audio = np.clip(audio, -1.0, 1.0)

    return audio


def _apply_reverb(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Add a simple room reverb effect using convolution.
    We generate a synthetic impulse response (IR) — no external
    IR file needed. The IR simulates a small room reflection.

    Convolution reverb works by simulating how sound bounces
    around a room. The IR captures that bounce pattern.
    fftconvolve does this efficiently in the frequency domain.
    """
    from scipy.signal import fftconvolve

    # Build a simple exponential decay IR (simulates a small room)
    # Duration: 0.3 seconds of decay
    ir_duration = int(0.3 * sr)
    t = np.linspace(0, 0.3, ir_duration)

    # Exponential decay: strong at start, fades to silence
    # decay rate 15 = fairly short room, increase for bigger room
    ir = np.exp(-15 * t)

    # Add a small delay (20ms) to simulate first reflection
    delay_samples = int(0.02 * sr)
    ir[:delay_samples] = 0   # silence before first reflection

    # Normalize the IR so it doesn't change overall volume
    ir = ir / np.sum(np.abs(ir))

    # Convolve audio with IR using FFT (fast convolution)
    reverb_signal = fftconvolve(audio, ir, mode="full")

    # Trim to original length (fftconvolve output is longer)
    reverb_signal = reverb_signal[:len(audio)]

    # Blend: 70% dry (original) + 30% wet (reverb)
    # Pure reverb would sound like a cave — blend is more natural
    blended = 0.7 * audio + 0.3 * reverb_signal
    return np.clip(blended, -1.0, 1.0)


def _apply_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Remove background hiss and TTS synthesis artifacts.
    Uses the noisereduce library which applies spectral gating —
    it estimates the noise floor from quieter parts of the audio
    and subtracts it from the whole signal.

    Install: pip install noisereduce
    """
    try:
        import noisereduce as nr
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            # stationary=True = assume noise is consistent throughout
            # Good for TTS output which has consistent background hiss
            prop_decrease=0.75
            # 0.75 = reduce noise by 75%. 1.0 would be too aggressive.
        )
    except ImportError:
        # If noisereduce isn't installed, skip silently
        print("[Modulation] noisereduce not installed — skipping denoise")
        return audio


def _apply_normalize(audio: np.ndarray) -> np.ndarray:
    """
    Peak normalize audio to -3dB (0.707 amplitude).
    Ensures every TTS output has consistent loudness regardless
    of what text was synthesized or which voice was used.

    Peak normalization finds the loudest sample, then scales
    all samples so that peak = target level.
    """
    peak = np.max(np.abs(audio))
    # np.abs() converts negative values to positive (we want the
    # magnitude of the peak, not its sign)

    if peak == 0:
        return audio   # silence — avoid division by zero

    target = 0.707
    # 0.707 = -3dB. (20 * log10(0.707) ≈ -3)
    # -3dB headroom prevents clipping on playback.

    return audio * (target / peak)


def process(wav_bytes: bytes) -> bytes:
    """
    Main entry point — called by tts_router after synthesis.

    Reads current modulation config from config.yaml,
    applies each enabled effect in order, returns modulated WAV bytes.

    The order of operations matters:
      1. Pitch shift   (frequency domain — do before time stretch)
      2. Time stretch  (time domain)
      3. EQ            (frequency shaping)
      4. Volume gain   (amplitude)
      5. Reverb        (convolution — do after EQ)
      6. Denoise       (spectral — do after reverb)
      7. Normalize     (final level — always last)
    """
    cfg = _load_mod_cfg()

    # Read all params from config (with safe defaults if missing)
    pitch     = cfg.get("pitch_semitones", 0)
    speed_p   = cfg.get("speed_post", 1.0)
    volume    = cfg.get("volume_gain", 1.0)
    bass      = cfg.get("bass", 0)
    treble    = cfg.get("treble", 0)
    reverb    = cfg.get("reverb", False)
    denoise   = cfg.get("denoise", True)
    normalize = cfg.get("normalize", True)

    # If everything is at default/off, skip all processing
    # This is a fast path that avoids importing heavy libraries
    no_pitch   = pitch == 0
    no_stretch = speed_p == 1.0
    no_volume  = volume == 1.0
    no_eq      = bass == 0 and treble == 0
    no_effects = not reverb and not denoise and not normalize

    if no_pitch and no_stretch and no_volume and no_eq and no_effects:
        return wav_bytes   # return original unchanged

    # Decode WAV bytes → numpy array
    audio, sr = _wav_bytes_to_numpy(wav_bytes)

    # Apply effects in order
    if pitch != 0:
        audio = _apply_pitch_shift(audio, sr, pitch)

    if speed_p != 1.0:
        audio = _apply_time_stretch(audio, speed_p)

    if bass != 0 or treble != 0:
        audio = _apply_eq(audio, sr, bass, treble)

    if volume != 1.0:
        audio = _apply_volume(audio, volume)

    if reverb:
        audio = _apply_reverb(audio, sr)

    if denoise:
        audio = _apply_denoise(audio, sr)

    if normalize:
        audio = _apply_normalize(audio)

    # Encode back to WAV bytes and return
    return _numpy_to_wav_bytes(audio, sr)