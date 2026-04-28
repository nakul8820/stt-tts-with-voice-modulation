# file defines the CONTRACT that every STT provider must follow
# Defines what STT models needs to do. ( job description)

from abc import ABC, abstractmethod
# abc - abstract base class , 
# every subclass needs to implement this base class
# if any subclass forgets this it raises an error 

class BaseSTTProvider(ABC):
    """
    base class for all speech-to-text providers
    """
    @abstractmethod
    def load(self):
        """
        Load model into memory.
        Called once at app start up,( where we initialise the model)
        load weights from disks , move to device 
        """
        ... 
        # this is python's way of saying 
        # this method has no body - subclasses must fill it in

    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Convert audio bytes to text

        Args:
            audio_bytes: raw audio file bytes (WAV, WebM, etc.)

        Returns:
            A dict with at minimum:
            {
                "text": "the transcribed text here",
                "language": "hi"   # detected language code
            }
        Every provider must return this same shape so the
        router doesn't care which model did the transcription.
        """
        ...
    