# core/factory.py
# ─────────────────────────────────────────────────────────────
# The factory reads config.yaml and returns the correct
# provider instance. This is the only place in the codebase
# that knows about specific provider class names.
#
# Pattern: Factory Method
# Why: main.py calls get_stt_provider() and gets back a
# BaseSTTProvider. It never imports WhisperFasterProvider
# directly. So swapping models = changing config.yaml only.
# ─────────────────────────────────────────────────────────────

import yaml
# yaml is the library that parses .yaml files into Python dicts

from core.base_stt import BaseSTTProvider
from core.base_tts import BaseTTSProvider


def _load_config() -> dict:
    """
    Read config.yaml from the project root and return it
    as a Python dictionary. Called internally by the two
    factory functions below.
    """
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
        # safe_load parses YAML without executing any code.
        # Never use yaml.load() — it can execute arbitrary code.


def get_stt_provider() -> BaseSTTProvider:
    """
    Read the stt.provider key from config.yaml and return
    an instance of the matching STT provider class.

    Returns a BaseSTTProvider — the caller only sees the
    abstract interface, not the concrete class.
    """
    cfg = _load_config()
    stt_cfg = cfg["stt"]               # the stt: section of config
    provider_name = stt_cfg["provider"] # e.g. "whisper_faster"

    # We import inside the match block (lazy import).
    # This means if you're only using whisper_faster, the vosk
    # and whisper_cpp packages are never imported — no errors
    # even if those packages aren't installed.

    if provider_name == "whisper_faster":
        from providers.stt.whisper_faster import WhisperFasterProvider
        return WhisperFasterProvider(stt_cfg)

    elif provider_name == "whisper_cpp":
        from providers.stt.whisper_cpp import WhisperCppProvider
        return WhisperCppProvider(stt_cfg)

    elif provider_name == "vosk":
        from providers.stt.vosk_provider import VoskProvider
        return VoskProvider(stt_cfg)

    else:
        # matches anything not above
        raise ValueError(
            f"Unknown STT provider: '{provider_name}'. "
            f"Check config.yaml. Valid options: "
            f"whisper_faster, whisper_cpp, vosk"
        )


def get_tts_provider() -> BaseTTSProvider:
    """
    Read the tts.provider key from config.yaml and return
    an instance of the matching TTS provider class.
    """
    cfg = _load_config()
    tts_cfg = cfg["tts"]
    provider_name = tts_cfg["provider"]  # e.g. "coqui_xtts"

    if provider_name == "coqui_xtts":
        from providers.tts.coqui_xtts import CoquiXTTSProvider
        return CoquiXTTSProvider(tts_cfg)

    elif provider_name == "indic_parler":
        from providers.tts.indic_parler import IndicParlerProvider
        return IndicParlerProvider(tts_cfg)

    elif provider_name == "mms":
        from providers.tts.mms_provider import MMSProvider
        return MMSProvider(tts_cfg)

    elif provider_name == "mms_hinglish":
        from providers.mms_hinglish_provider import MmsHinglishProvider
        return MmsHinglishProvider(tts_cfg)

    elif provider_name == "piper":
        from providers.tts.piper_provider import PiperProvider
        return PiperProvider(tts_cfg)

    else:
        raise ValueError(
            f"Unknown TTS provider: '{provider_name}'. "
            f"Valid options: coqui_xtts, indic_parler, mms, mms_hinglish, piper"
        )