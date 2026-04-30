# STT provider using faster-whisper 
# faster-whisper is reimplementation of OpenAO whisper
# Using CTranslate2 - it's 4X faster and uses less memory than the 
# original while producing identical transcriptions

import os
import tempfile

# tempfile lets us write audio bytes to temporary file on diskk
# faster-whisper needs file path, not raw bytes - so we write
# the bytes to temp file, transcribe it, then delete it 

from faster_whisper import WhisperModel
from core.base_stt import BaseSTTProvider

class WhisperFasterProvider(BaseSTTProvider):
    """
    STT provider wrapping faster-whisper.
    Handles Higlish well - set langauge="hi and 
    Whisper naturally hanndles code-mixed Hindi+English speech
    """
    def __init__(self, cfg: dict):
        """
        Store config values . model is not loaded here.
        Loading is done at start up
        """
        self.model_size = cfg.get("model_size", "medium")
        # .get(key, default) - use default if key is missing 

        self.language = cfg.get("language", "hi")

        self.compute_type = cfg.get("compute_type", "int8")
        # 8-bit integer quantization 
        
        self.model = None
        # will hold the loaded whisper model once load runs()

    def load(self) -> None:
        """
        Load the whisper model from disk into memory.
        faster-whisper auto-downloads the models on first call
        and catches it in `/.catche/huggingface/hub
        After first download, it loads from cache-no internet
        """
        print(f"[WhisperFaster] Loading Model: {self.model_size}...")

        try:
            # Try loading from local cache first
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                local_files_only=True
            )
            print(f"[WhisperFaster] Model loaded from local cache.")
        except Exception:
            print(f"[WhisperFaster] Local model not found or internet check required. Attempting download...")
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type
            )
            print(f"[WhisperFaster] Model downloaded and loaded successfuly")

    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Transcribe audio bytes to text.
        Steps:
        1. write bytes to temp .wav file
        2. run transcription
        3. join all segments into one string
        4. delete temp file
        5. return dict with text amd detected language
        """
        # step 1 , temp .wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # step 2 Run transcription
            # transcribe() returns generator of segments + info 
            # Each segment is chunk transcibed text with
            # start/ end timestamps.
            segments, info = self.model.transcribe(
                tmp_path,
                language = self.language,
                task = "transcribe",
                # transcibe = outputs original language
                # "translate" = outputs in English # Not needed
                beam_size = 5,
                # beam_search width, higher = more accurate but slower
                # 5 is a good defualt balace
                
                vad_filter = True,
                # VAD = Voice Activity detection
                # skips silent parts of audio automatically
                # speeds up transcription and reduces hallucinations
            )

            # step 3 : join segments
            # segments is a generator 
            # each segment has a .text attribute with transcribed text

            text = " ".join(segment.text for segment in segments)

            return {
                "text": text.strip(),
                "language": info.language
            }

        finally:
            # step 4 : always delete the temp file, even if error
            # occured above. 
            os.unlink(tmp_path)