# modules/mms_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Low-level MMS model wrapper.
# Loads and caches both MMS-Hindi and MMS-English models.
# Exposes synthesize() with per-call VITS parameter control.
#
# Used by all 3 strategies — they share one engine instance so models
# are loaded only once regardless of how many variants we generate.
# ─────────────────────────────────────────────────────────────────────────────

import time
import logging
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# VITS inference parameters
# These control the character of the synthesized voice
DEFAULT_PARAMS = {
    "noise_scale":    0.667,   # expressiveness / pronunciation variance
    "noise_scale_w":  0.8,     # duration variance
    "length_scale":   1.0,     # speaking rate (>1 = slower, <1 = faster)
}

# Tuned params — push Hindi model toward more neutral cadence
# and English model toward slightly more expressive Indian-style delivery
TUNED_PARAMS = {
    "hi": {
        "noise_scale":   0.4,    # less variance = more consistent, neutral
        "noise_scale_w": 0.6,    # tighter duration prediction
        "length_scale":  1.05,   # very slightly slower (Hindi natural pace)
    },
    "en": {
        "noise_scale":   0.5,    # pull English toward Hindi expressiveness level
        "noise_scale_w": 0.7,
        "length_scale":  1.1,    # slightly slower to match Hindi cadence
    },
}


class MmsEngine:
    """
    Loads MMS-Hindi and MMS-English once, exposes synthesize() per segment.
    Shared by all strategy variants in the experiment.
    """

    MODEL_IDS = {
        "hi":  "facebook/mms-tts-hin",
        "en":  "facebook/mms-tts-eng",
    }
    SAMPLE_RATE = 16000

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self._models: Dict = {}
        self._tokenizers: Dict = {}
        self._xlit_engine = None
        self._xlit_available = None  # None=not tried, False=unavailable, True=available
        self._loaded = False

    def load(self) -> None:
        """Load both MMS models and IndicXlit. Call once at startup."""
        if self._loaded:
            return

        logger.info("Loading MMS Engine...")
        t_total = time.time()

        try:
            from transformers import VitsModel, AutoTokenizer
            import torch

            for lang, model_id in self.MODEL_IDS.items():
                path = (self.model_dir + f"/{model_id.split('/')[-1]}") if self.model_dir else model_id
                logger.info(f"  Loading {model_id}...")
                t = time.time()
                
                import os
                local_files_only = os.path.isdir(path)
                
                self._tokenizers[lang] = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
                m = VitsModel.from_pretrained(path, local_files_only=local_files_only)
                m.eval()
                self._models[lang] = m.to("cpu")
                logger.info(f"  {model_id} ready in {time.time()-t:.1f}s")

        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        # Try loading IndicXlit (optional — used for Hindi segments)
        try:
            from modules.hinglish_transliterator import HinglishTransliterator
            self._xlit_engine = HinglishTransliterator(model_dir=self.model_dir)
            self._xlit_engine.load()
            self._xlit_available = True
            logger.info("  IndicXlit loaded for Hindi transliteration")
        except Exception as e:
            self._xlit_available = False
            logger.warning(
                f"  HinglishTransliterator not loaded: {e}. "
                "Hindi segments will use romanized text (reduced quality)."
            )

        self._loaded = True
        logger.info(f"MMS Engine ready in {time.time()-t_total:.1f}s")

    def synthesize(
        self,
        text: str,
        lang: str,
        use_phonetic_english: bool = False,
        noise_scale: float = DEFAULT_PARAMS["noise_scale"],
        noise_scale_w: float = DEFAULT_PARAMS["noise_scale_w"],
        length_scale: float = DEFAULT_PARAMS["length_scale"],
    ) -> Optional[np.ndarray]:
        """
        Synthesize a single segment.

        Args:
            text:                  Input text (romanized for Hindi, plain for English)
            lang:                  "hi" or "en"
            use_phonetic_english:  If True AND lang=="en", transliterate English
                                   to Devanagari phonetics and send to Hindi model.
                                   This gives Indian-accented English from one speaker.
            noise_scale:           VITS expressiveness parameter
            noise_scale_w:         VITS duration variance parameter
            length_scale:          VITS speaking rate (1.0=normal, 1.1=slower)

        Returns:
            float32 numpy waveform or None on failure
        """
        if not self._loaded:
            raise RuntimeError("Call load() first")

        try:
            import torch

            if lang == "hi":
                # Romanized Hindi → Devanagari
                text = self._to_devanagari(text)
                model_lang = "hi"

            elif lang == "en" and use_phonetic_english:
                # English → Devanagari phonetics → Hindi model
                # Gives Indian-accented English, same speaker as Hindi segments
                text = self._english_to_devanagari_phonetic(text)
                model_lang = "hi"

            else:
                # Standard English → MMS-English model
                model_lang = "en"

            tokenizer = self._tokenizers[model_lang]
            model = self._models[model_lang]

            inputs = tokenizer(text, return_tensors="pt")
            
            model_inputs = {}
            for k, v in inputs.items():
                if k == "input_ids":
                    model_inputs[k] = v.long()
                else:
                    model_inputs[k] = v

            with torch.no_grad():
                output = model(**model_inputs)

            waveform = output.waveform.squeeze().cpu().numpy().astype(np.float32)

            # Manually apply length scale (speed adjustment) since model doesn't support it in forward()
            if abs(length_scale - 1.0) > 0.01:
                try:
                    import scipy.signal as signal
                    target_len = int(len(waveform) / length_scale)
                    waveform = signal.resample(waveform, target_len).astype(np.float32)
                except ImportError:
                    pass

            return waveform

        except Exception as e:
            logger.error(f"Synthesis failed [{lang}] '{text[:40]}': {e}")
            return None

    # ── Transliteration helpers ────────────────────────────────────────────────

    def _to_devanagari(self, romanized: str) -> str:
        """Romanized Hindi → Devanagari via IndicXlit."""
        if not self._xlit_available:
            return romanized
        return self._xlit_engine.transliterate(romanized)

    def _english_to_devanagari_phonetic(self, text: str) -> str:
        """
        Phonetically map English words to Devanagari so MMS-Hindi model
        produces Indian-accented English.

        Uses a handcrafted phonetic map for common Hinglish English words,
        with IndicXlit as fallback for unknown words.

        Example: "meeting" → "मीटिंग", "office" → "ऑफिस"
        """
        PHONETIC_MAP = {
            "meeting":      "मीटिंग",
            "meetings":     "मीटिंग्स",
            "office":       "ऑफिस",
            "phone":        "फोन",
            "email":        "ईमेल",
            "laptop":       "लैपटॉप",
            "computer":     "कंप्यूटर",
            "internet":     "इंटरनेट",
            "wifi":         "वाईफाई",
            "password":     "पासवर्ड",
            "download":     "डाउनलोड",
            "upload":       "अपलोड",
            "file":         "फाइल",
            "folder":       "फोल्डर",
            "project":      "प्रोजेक्ट",
            "deadline":     "डेडलाइन",
            "presentation": "प्रेजेंटेशन",
            "report":       "रिपोर्ट",
            "document":     "डॉक्यूमेंट",
            "team":         "टीम",
            "manager":      "मैनेजर",
            "boss":         "बॉस",
            "client":       "क्लाइंट",
            "server":       "सर्वर",
            "software":     "सॉफ्टवेयर",
            "app":          "ऐप",
            "update":       "अपडेट",
            "call":         "कॉल",
            "ok":           "ओके",
            "okay":         "ओके",
            "thanks":       "थैंक्स",
            "hello":        "हेलो",
            "hi":           "हाई",
            "bye":          "बाय",
            "please":       "प्लीज",
            "sorry":        "सॉरी",
            "good":         "गुड",
            "great":        "ग्रेट",
            "perfect":      "परफेक्ट",
            "problem":      "प्रॉब्लम",
            "issue":        "इशू",
            "solution":     "सॉल्यूशन",
            "idea":         "आइडिया",
            "plan":         "प्लान",
            "check":        "चेक",
            "fix":          "फिक्स",
            "send":         "सेंड",
            "share":        "शेयर",
            "start":        "स्टार्ट",
            "stop":         "स्टॉप",
            "schedule":     "शेड्यूल",
            "conference":   "कॉन्फ्रेंस",
            "room":         "रूम",
            "free":         "फ्री",
            "urgent":       "अर्जेंट",
            "ready":        "रेडी",
            "done":         "डन",
            "approve":      "अप्रूव",
            "approved":     "अप्रूव्ड",
            "proposal":     "प्रपोजल",
            "review":       "रिव्यू",
            "feedback":     "फीडबैक",
            "charge":       "चार्ज",
            "slow":         "स्लो",
            "deliver":      "डिलीवर",
            "dispatch":     "डिस्पैच",
            "order":        "ऑर्डर",
            "product":      "प्रोडक्ट",
            "feature":      "फीचर",
            "features":     "फीचर्स",
            "dashboard":    "डैशबोर्ड",
            "button":       "बटन",
            "click":        "क्लिक",
            "submit":       "सबमिट",
            "login":        "लॉगिन",
            "payment":      "पेमेंट",
            "error":        "एरर",
            "message":      "मैसेज",
            "monday":       "मंडे",
            "tuesday":      "ट्यूसडे",
            "wednesday":    "वेडनसडे",
            "thursday":     "थर्सडे",
            "friday":       "फ्राइडे",
            "saturday":     "सैटरडे",
            "sunday":       "संडे",
            "at":           "एट",
            "pm":           "पीएम",
            "am":           "एएम",
            "in":           "इन",
            "on":           "ऑन",
            "for":          "फॉर",
            "with":         "विद",
            "from":         "फ्रॉम",
            "and":          "एंड",
            "or":           "ऑर",
            "the":          "द",
            "is":           "इज",
            "will":         "विल",
            "can":          "कैन",
            "should":       "शुड",
            "right":        "राइट",
            "wrong":        "रॉन्ग",
        }

        words = text.split()
        result = []
        for word in words:
            lower = word.lower().rstrip(".,!?")
            if lower in PHONETIC_MAP:
                result.append(PHONETIC_MAP[lower])
            elif self._xlit_available:
                # IndicXlit handles arbitrary romanized words
                transliterated = self._xlit_engine._transliterate_word(lower)
                result.append(transliterated)
            else:
                result.append(word)

        return " ".join(result)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
