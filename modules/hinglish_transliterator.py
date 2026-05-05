# modules/hinglish_transliterator.py
# ─────────────────────────────────────────────────────────────────────────────
# Hinglish Transliterator
#
# Converts romanized Hinglish text → Devanagari script
# so Facebook MMS-TTS Hindi model can process it.
#
# Strategy:
#   1. Tokenize input into words
#   2. Each word: check if it's a pure English word (from a small known-English
#      vocabulary) or a Hindi/ambiguous romanized word
#   3. Hindi/ambiguous words → IndicXlit (romanized → Devanagari)
#   4. English words → also passed through IndicXlit for phonetic
#      Devanagari mapping (e.g., "meeting" → "मीटिंग")
#      This keeps everything in one script for MMS-Hindi
#   5. Numbers, punctuation → handled separately
#
# Why everything through IndicXlit?
#   We want ONE consistent voice (MMS-Hindi). For that, MMS needs pure
#   Devanagari input. English words get phonetically transliterated so
#   they sound like how an Indian speaker would naturally say them.
#
# Usage:
#   from modules.hinglish_transliterator import HinglishTransliterator
#   t = HinglishTransliterator()
#   result = t.transliterate("aaj mera meeting hai at 3 PM")
#   # → "आज मेरा मीटिंग है एट 3 पीएम"
# ─────────────────────────────────────────────────────────────────────────────

import re
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Number-to-Hindi-words mapping ────────────────────────────────────────────
# MMS-Hindi handles Devanagari digits fine, but spelled-out words sound
# more natural. We convert common numbers to Hindi words.

ONES = ["", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
        "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह",
        "अठारह", "उन्नीस"]
TENS = ["", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"]


def _number_to_hindi(n: int) -> str:
    """Convert integer 0-999 to Hindi words."""
    if n == 0:
        return "शून्य"
    if n < 20:
        return ONES[n]
    if n < 100:
        ten, one = divmod(n, 10)
        return TENS[ten] + (" " + ONES[one] if one else "")
    if n < 1000:
        hun, rest = divmod(n, 100)
        return ONES[hun] + " सौ" + (" " + _number_to_hindi(rest) if rest else "")
    return str(n)  # fallback for large numbers


def _convert_numbers_in_text(text: str) -> str:
    """Replace all digit sequences in text with Hindi words."""
    def replace_num(match):
        num = int(match.group())
        return _number_to_hindi(num)
    return re.sub(r'\b\d+\b', replace_num, text)


# ── Common English abbreviations → Hindi phonetic equivalents ────────────────
# These are handled before IndicXlit because IndicXlit may struggle with
# all-caps abbreviations.

ABBREVIATIONS = {
    "PM":  "पीएम",
    "AM":  "एएम",
    "OK":  "ओके",
    "ok":  "ओके",
    "AI":  "एआई",
    "HR":  "एचआर",
    "MR":  "मिस्टर",
    "MRS": "मिसेज",
    "MS":  "मिस",
    "DR":  "डॉक्टर",
    "vs":  "वर्सेस",
    "etc": "एट सेटेरा",
    "kg":  "किलोग्राम",
    "km":  "किलोमीटर",
    "cm":  "सेंटीमीटर",
    "ml":  "मिलीलीटर",
    "GB":  "जीबी",
    "MB":  "एमबी",
    "KB":  "केबी",
}


class HinglishTransliterator:
    """
    Converts romanized Hinglish text to Devanagari using AI4Bharat IndicXlit.

    Designed to be instantiated once and reused (model stays in memory).
    Thread-safe for read operations after load().
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Args:
            model_dir: Optional path to pre-downloaded IndicXlit models.
                       If None, uses default HuggingFace cache location.
        """
        self.model_dir = model_dir
        self._engine = None
        self._loaded = False

    def load(self) -> None:
        """
        Load IndicXlit model into memory.
        """
        if self._loaded:
            return

        logger.info("Loading Hinglish Transliterator (Mode: ai4bharat)...")
        start = time.time()

        try:
            # SHIELD: Allow PyTorch to load Namespace objects (required for PyTorch 2.6+)
            import torch
            import argparse
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([argparse.Namespace])
                logger.debug("Added argparse.Namespace to PyTorch safe globals.")

            # SHIELD: Mock urduhack to prevent it from loading broken Tensorflow/Keras deps
            # We only need Hindi, so we don't care about Urdu logic.
            import sys
            from unittest.mock import MagicMock
            if 'urduhack' not in sys.modules:
                mock_urdu = MagicMock()
                mock_urdu.normalize = lambda x: x
                sys.modules['urduhack'] = mock_urdu
                logger.debug("Mocked urduhack to bypass Tensorflow conflicts.")

            from ai4bharat.transliteration import XlitEngine
            # IndicXlit handles Hindi via 'hi' code
            self._engine = XlitEngine(src_script_type="roman", beam_width=15, rescore=True)
            self._mode = "ai4bharat"
            logger.info("Successfully loaded AI4Bharat XlitEngine.")
        except Exception as e:
            logger.error(f"FATAL: Failed to load AI4Bharat engine: {e}")
            raise RuntimeError(f"IndicXlit engine failed to load: {e}")

        self._loaded = True
        elapsed = time.time() - start
        logger.info(f"Transliterator ready in {elapsed:.2f}s (Mode: {self._mode})")

    def _transliterate_word(self, word: str) -> str:
        """
        Transliterate a single romanized word to Devanagari using neural XlitEngine.
        """
        if not word.strip():
            return word
        
        try:
            # AI4Bharat XlitEngine.translit_word(word, lang_code) 
            # returns a LIST of strings (top-k results).
            results = self._engine.translit_word(word.lower(), lang_code="hi")
            
            if results and isinstance(results, list) and len(results) > 0:
                # Use the top result
                return results[0]
            
            return word
        except Exception as e:
            logger.debug(f"Neural translit failed for '{word}': {e}")
            return word

    def transliterate(self, text: str) -> str:
        """
        Convert romanized Hinglish text to Devanagari.

        Pipeline:
          raw text
            → expand abbreviations
            → convert numbers to Hindi words
            → tokenize into words + punctuation
            → transliterate each word via IndicXlit
            → rejoin

        Args:
            text: Romanized Hinglish string, e.g. "aaj mera meeting hai"

        Returns:
            Devanagari string, e.g. "आज मेरा मीटिंग है"
        """
        if not self._loaded:
            raise RuntimeError("Call load() before transliterate()")

        if not text or not text.strip():
            return text

        # Step 1: Expand known abbreviations (case-sensitive check)
        for abbr, hindi in ABBREVIATIONS.items():
            # Word-boundary match so "AM" doesn't match inside "SAME"
            text = re.sub(rf'\b{re.escape(abbr)}\b', hindi, text)

        # Step 2: Convert digit sequences to Hindi words
        text = _convert_numbers_in_text(text)

        # Step 3: Tokenize — split on whitespace but keep punctuation attached
        # to words (MMS handles punctuation in Devanagari naturally)
        tokens = text.split()
        devanagari_tokens = []

        for token in tokens:
            # Strip leading/trailing punctuation for transliteration,
            # then reattach
            match = re.match(r'^([^\w]*)(\w[\w\'-]*)([^\w]*)$', token)
            if match:
                prefix, word, suffix = match.groups()
                # Skip if already Devanagari (pass-through)
                if re.search(r'[\u0900-\u097F]', word):
                    devanagari_tokens.append(token)
                else:
                    transliterated = self._transliterate_word(word)
                    devanagari_tokens.append(prefix + transliterated + suffix)
            else:
                # Pure punctuation or whitespace token
                devanagari_tokens.append(token)

        result = " ".join(devanagari_tokens)
        logger.debug(f"Transliteration: '{text}' → '{result}'")
        return result

    def transliterate_batch(self, texts: list) -> list:
        """
        Transliterate a list of texts. More efficient for bulk processing.
        """
        return [self.transliterate(t) for t in texts]

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class RuleBasedHindiTransliterator:
    """
    Improved rule-based transliterator for Hindi.
    Uses a dictionary for common English/Hinglish words and refined phonetic rules.
    """
    def __init__(self):
        # 1. Common English -> Devanagari Dictionary
        # These are high-frequency words in Hinglish that rules might mess up.
        self.dictionary = {
            "aaj": "आज", "mera": "मेरा", "hai": "है", "kal": "कल", "mujhe": "मुझे",
            "meeting": "मीटिंग", "office": "ऑफिस", "kya": "क्या", "free": "फ्री",
            "phone": "फोन", "charge": "चार्ज", "email": "ईमेल", "check": "चेक",
            "project": "प्रोजेक्ट", "deadline": "डेडलाइन", "baje": "बजे",
            "conference": "कॉन्फ्रेंस", "room": "रूम", "ghante": "घंटे",
            "chahiye": "चाहिए", "kaam": "काम", "khatam": "खत्म", "karne": "करने",
            "liye": "लिए", "bhai": "भाई", "team": "टीम", "call": "कॉल",
            "presentation": "प्रेजेंटेशन", "ready": "रेडी", "karni": "करनी",
            "help": "हेल्प", "kar": "कर", "sakte": "सकते", "problem": "प्रॉब्लम",
            "solve": "सॉल्व", "mein": "में", "internet": "इंटरनेट", "slow": "स्लो",
            "chal": "चल", "raha": "रहा", "file": "फाइल", "download": "डाउनलोड",
            "gayi": "गयी", "haan": "हाँ", "boss": "बॉस", "approve": "अप्रूव",
            "diya": "दिया", "proposal": "प्रपोजल", "aapka": "आपका", "order": "ऑर्डर",
            "dispatch": "डिस्पैच", "gaya": "गया", "din": "दिन", "deliver": "डिलीवर",
            "jayega": "जायेगा", "issue": "इश्यू", "fix": "फिक्स", "karna": "करना",
            "bahut": "बहुत", "urgent": "अर्जेंट", "how":""
        }

        # 2. Phonetic rules (Longer patterns first)
        self.rules = [
            # Vowels (Initial/Standalone)
            (r'^aa', 'आ'), (r'^a', 'अ'), (r'^i', 'इ'), (r'^ee', 'ई'), 
            (r'^u', 'उ'), (r'^oo', 'ऊ'), (r'^e', 'ए'), (r'^ai', 'ऐ'), 
            (r'^o', 'ओ'), (r'^au', 'औ'),

            # Consonant clusters / Aspirated
            ('kh', 'ख'), ('gh', 'घ'), ('ch', 'च'), ('chh', 'छ'), ('jh', 'झ'),
            ('th', 'थ'), ('dh', 'ध'), ('ph', 'फ'), ('bh', 'भ'), ('sh', 'श'),
            ('gy', 'ज्ञ'), ('tr', 'त्र'), ('ks', 'क्ष'),

            # Vowel Matras (When not at start)
            ('aa', 'ा'), ('ee', 'ी'), ('oo', 'ू'), ('ai', 'ै'), ('au', 'ौ'),
            ('i', 'ि'), ('u', 'ु'), ('e', 'े'), ('o', 'ो'), ('a', ''), # Implicit schwa

            # Basic Consonants
            ('k', 'क'), ('g', 'ग'), ('j', 'ज'), ('t', 'त'), ('d', 'द'),
            ('n', 'न'), ('p', 'प'), ('b', 'ब'), ('m', 'म'), ('y', 'य'),
            ('r', 'र'), ('l', 'ल'), ('v', 'व'), ('w', 'व'), ('s', 'स'), ('h', 'ह'),
            ('z', 'ज़'), ('f', 'फ़'), ('c', 'क'), ('x', 'क्स')
        ]

    def transliterate(self, word: str) -> str:
        word = word.lower().strip()
        if not word: return ""

        # Step 1: Dictionary lookup
        if word in self.dictionary:
            return self.dictionary[word]

        # Step 2: Apply regex-based rules for initials
        res = word
        for pattern, replacement in self.rules:
            if pattern.startswith('^'):
                res = re.sub(pattern, replacement, res)
        
        # Step 3: Apply remaining literal rules
        for pattern, replacement in self.rules:
            if not pattern.startswith('^'):
                res = res.replace(pattern, replacement)
        
        return res
