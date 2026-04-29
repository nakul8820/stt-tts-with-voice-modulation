# providers/tts/indic_parler.py
# ─────────────────────────────────────────────────────────────
# TTS provider using Indic Parler-TTS (by AI4Bharat + HuggingFace)
# Supports 21 Indian languages + English. Auto-detects language
# from the script of the input text — no language flag needed.
# Uses a text "description" caption to control voice style.
#
# Install: pip install parler-tts transformers
# Model: ai4bharat/indic-parler-tts-mini (~1.6GB)
# ─────────────────────────────────────────────────────────────

import io
import torch
import numpy as np
import soundfile as sf
from core.base_tts import BaseTTSProvider


class IndicParlerProvider(BaseTTSProvider):
    """
    TTS provider using Indic Parler-TTS mini.
    Voice style is controlled via a natural language description
    instead of selecting a speaker by ID — very flexible.
    """

    def __init__(self, cfg: dict):
        self.device = cfg.get("device", "cpu")
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None

        # This caption controls HOW the voice sounds.
        # You can change this to experiment with different styles.
        # Examples:
        # "A male speaker delivers clear speech at moderate speed."
        # "A female speaker with expressive, fast-paced Hindi speech."
        self.voice_description = cfg.get(
            "voice_description",
            "A female speaker delivers clear, natural speech "
            "at a moderate pace with a neutral tone."
        )

    def load(self) -> None:
        """Load Indic Parler-TTS model and tokenizers."""
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        if self.device == "mps" and not torch.backends.mps.is_available():
            print("[IndicParler] MPS not available, using CPU")
            self.device = "cpu"

        print("[IndicParler] Loading model ...")
        model_id = "ai4bharat/indic-parler-tts-mini"

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id
        ).to(self.device)

        # Parler-TTS uses TWO tokenizers:
        # 1. tokenizer → for the input text (what to say)
        # 2. description_tokenizer → for the style caption
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        print("[IndicParler] Model loaded.")

    def synthesize(self, text: str, voice_id: str = "default") -> bytes:
        """
        Convert text to audio using Parler-TTS.
        Language is detected automatically from the text script.
        Hindi Devanagari → Hindi voice, Latin script → English voice.
        """
        # Tokenize the style description
        desc_inputs = self.description_tokenizer(
            self.voice_description,
            return_tensors="pt"  # return PyTorch tensors
        ).to(self.device)

        # Tokenize the actual text to speak
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        # Generate audio
        with torch.no_grad():
            # torch.no_grad() disables gradient tracking.
            # We're doing inference (not training), so we don't
            # need gradients. This saves memory and speeds things up.
            generation = self.model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=text_inputs.input_ids,
                prompt_attention_mask=text_inputs.attention_mask,
            )

        # generation is a tensor — convert to numpy float32
        audio_np = generation.cpu().numpy().squeeze().astype(np.float32)
        # .cpu() moves tensor from MPS/GPU back to CPU
        # .numpy() converts to numpy array
        # .squeeze() removes dimensions of size 1
        # e.g. shape (1, 1, 44100) → (44100,)

        # Encode numpy array → WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, samplerate=44100, format="WAV")
        # Parler-TTS outputs at 44100 Hz (CD quality)
        buffer.seek(0)
        return buffer.read()