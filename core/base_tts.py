# core/base_tts.py
# ─────────────────────────────────────────────────────────────
# Same idea as base_stt.py but for Text-to-Speech.
# Every TTS model must implement load() and synthesize().
# The router calls synthesize() and gets back audio bytes —
# it doesn't know or care which model produced them.
# ─────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod


class BaseTTSProvider(ABC):
    """
    Abstract base class for all Text-to-Speech providers.
    Every TTS provider (Coqui, Parler, Piper, etc.) must
    inherit from this and implement both methods.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load the TTS model into memory.
        Called once at startup. Load weights, set device,
        load speaker embeddings, etc.
        """
        ...

    @abstractmethod
    def synthesize(self, text: str, voice_id: str = "default") -> bytes:
        """
        Convert text to speech audio.

        Args:
            text:     the input text (Hinglish, Hindi, English)
            voice_id: which voice profile to use.
                      "default" = the model's built-in voice.
                      Admin can create named profiles later.

        Returns:
            Raw WAV audio as bytes.
            WAV is chosen because it's universally playable
            in the browser without any decoding library.
        """
        ...