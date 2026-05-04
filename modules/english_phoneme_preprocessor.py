# modules/english_phoneme_preprocessor.py
# ─────────────────────────────────────────────────────────────────────────────
# English Phoneme Preprocessor for Hinglish TTS Pipeline
#
# Purpose:
#   Current pipeline: English word → IndicXlit (romanized) → Devanagari
#   Problem: IndicXlit treats English words as romanized Hindi, so "email"
#            becomes "एमाइल" (wrong vowels) instead of "ईमेल" (correct).
#
#   New pipeline: English word → gruut G2P → IPA → IPA→Devanagari map → Devanagari
#   Result: "email" → ['i','m','ˈeɪ','l'] → "इमेल" (accurate phonetics)
#
# Strategy:
#   1. Detect English words in the Hinglish input
#   2. For each English word, get IPA phonemes from gruut (CMU dict-based)
#   3. Map each IPA symbol to the closest Devanagari equivalent
#   4. Fall back to IndicXlit for words gruut doesn't know
#
# Usage (standalone):
#   from modules.english_phoneme_preprocessor import EnglishPhonemePreprocessor
#   pp = EnglishPhonemePreprocessor()
#   result = pp.english_word_to_devanagari("meeting")
#   # → "मीटिंग"
#
# Usage (inside HinglishTransliterator):
#   pp = EnglishPhonemePreprocessor()
#   if is_english_word(word):
#       devanagari = pp.english_word_to_devanagari(word)
#   else:
#       devanagari = xlit_engine.translit_word(word)
# ─────────────────────────────────────────────────────────────────────────────

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── IPA → Devanagari mapping ──────────────────────────────────────────────────
# Built from the IPA chart mapped to how Indian speakers naturally pronounce
# English words in Hindi. Ordered from longest to shortest to avoid partial
# matches (e.g. 'eɪ' must come before 'e').

IPA_TO_DEVANAGARI: List[tuple] = [
    # ── Diphthongs (must come before their component vowels) ─────────────────
    ("eɪ", "े"),       # 'a' in "face", "email" → े  (e matra)
    ("aɪ", "ाइ"),     # 'i' in "price", "deadline" → ाइ
    ("ɔɪ", "ॉइ"),     # 'oy' in "boy"
    ("aʊ", "ाउ"),     # 'ow' in "mouth"
    ("oʊ", "ो"),      # 'o' in "goat" → ो
    ("ɪə", "िअ"),     # 'ear' sound
    ("eə", "ेअ"),     # 'air' sound
    ("ʊə", "ुअ"),     # 'tour' sound

    # ── Affricates ────────────────────────────────────────────────────────────
    ("t͡ʃ", "च"),     # 'ch' in "check", "chat"
    ("d͡ʒ", "ज"),     # 'j' in "judge"
    ("tʃ",  "च"),     # alternate encoding for 'ch'
    ("dʒ",  "ज"),     # alternate encoding for 'j'

    # ── Long/stressed vowels ──────────────────────────────────────────────────
    # gruut prefixes stressed syllables with ˈ or ˌ — we strip these inline
    ("ˈi",  "ी"),     # stressed 'ee' → ी
    ("ˌi",  "ि"),     # unstressed 'ee'
    ("ˈɪ",  "ि"),     # stressed short 'i'
    ("ˌɪ",  "ि"),     # unstressed short 'i'
    ("ˈeɪ", "े"),     # stressed 'ay' → े
    ("ˌeɪ", "े"),
    ("ˈæ",  "ै"),     # stressed 'a' in "cat" → ै (near-front)
    ("ˌæ",  "ै"),
    ("ˈɑ",  "ा"),     # stressed 'ah' sound
    ("ˌɑ",  "ा"),
    ("ˈɔ",  "ॉ"),     # stressed 'aw' sound → ॉ
    ("ˌɔ",  "ॉ"),
    ("ˈu",  "ू"),     # stressed 'oo' → ू
    ("ˌu",  "ू"),
    ("ˈʊ",  "ु"),     # stressed short 'oo'
    ("ˌʊ",  "ु"),
    ("ˈʌ",  ""),      # stressed 'uh' → schwa (implicit in Hindi)
    ("ˌʌ",  ""),
    ("ˈɛ",  "े"),     # stressed 'e' in "bed" → े
    ("ˌɛ",  "े"),
    ("ˈoʊ", "ो"),
    ("ˌoʊ", "ो"),
    ("ˈaɪ", "ाइ"),
    ("ˌaɪ", "ाइ"),
    ("ˈaʊ", "ाउ"),
    ("ˌaʊ", "ाउ"),
    ("ˈɔɪ", "ॉइ"),
    ("ˌɔɪ", "ॉइ"),

    # ── Short/unstressed vowels ───────────────────────────────────────────────
    ("i",   "ि"),     # short 'i'
    ("ɪ",   "ि"),     # near-close front vowel
    ("e",   "े"),     # front mid vowel
    ("æ",   "ै"),     # near-open front vowel
    ("ɑ",   "ा"),     # open back vowel ('ah')
    ("ɒ",   "ॉ"),     # rounded open back (British)
    ("ɔ",   "ॉ"),     # open-mid back rounded
    ("ʌ",   ""),      # open-mid back unrounded (schwa-like, silent in Hindi)
    ("ə",   ""),      # schwa — silent in Hindi
    ("ɜ",   "र"),     # r-colored vowel → र approximation
    ("ɚ",   "र"),     # r-colored schwa
    ("u",   "ू"),     # back close rounded
    ("ʊ",   "ु"),     # near-close back rounded
    ("ɛ",   "े"),     # open-mid front
    ("o",   "ो"),     # back mid rounded

    # ── Consonants ────────────────────────────────────────────────────────────
    ("p",   "प"),
    ("b",   "ब"),
    ("t",   "ट"),     # dental T → retroflex ट sounds closer in Indian English
    ("d",   "ड"),     # dental D → retroflex ड
    ("k",   "क"),
    ("g",   "ग"),
    ("f",   "फ़"),
    ("v",   "व"),
    ("s",   "स"),
    ("z",   "ज़"),
    ("ʃ",   "श"),     # 'sh' sound
    ("ʒ",   "ज"),     # 'zh' sound (measure)
    ("h",   "ह"),
    ("m",   "म"),
    ("n",   "न"),
    ("ŋ",   "ं"),     # 'ng' sound → anusvara (nasal dot)
    ("l",   "ल"),
    ("r",   "र"),
    ("ɹ",   "र"),     # English rhotic 'r'
    ("w",   "व"),
    ("j",   "य"),     # English 'y' consonant
    ("θ",   "थ"),     # voiceless 'th' (three) → थ
    ("ð",   "द"),     # voiced 'th' (the) → द

    # ── Stress/length markers (strip them) ───────────────────────────────────
    ("ˈ",   ""),
    ("ˌ",   ""),
    ("ː",   ""),      # length mark
]


class EnglishPhonemePreprocessor:
    """
    Converts English words to Devanagari phonetics using:
      English text → gruut G2P → IPA phonemes → Devanagari

    This is more accurate than IndicXlit for English because:
    - gruut uses the CMU Pronouncing Dictionary (109k English words)
    - IPA gives exact phoneme sequences
    - Our IPA→Devanagari map reflects how Indian speakers say English words

    Falls back to a curated hand-coded dictionary for common words
    where the phoneme mapping produces sub-optimal results.
    """

    # Hand-curated override dictionary.
    # These are words where gruut IPA → Devanagari produces something awkward,
    # so we hard-code the ideal phonetic Devanagari representation.
    OVERRIDES = {
        "email":        "ईमेल",
        "laptop":       "लैपटॉप",
        "meeting":      "मीटिंग",
        "meetings":     "मीटिंग्स",
        "office":       "ऑफिस",
        "presentation": "प्रेजेंटेशन",
        "deadline":     "डेडलाइन",
        "project":      "प्रोजेक्ट",
        "schedule":     "शेड्यूल",
        "password":     "पासवर्ड",
        "internet":     "इंटरनेट",
        "download":     "डाउनलोड",
        "upload":       "अपलोड",
        "software":     "सॉफ्टवेयर",
        "computer":     "कंप्यूटर",
        "dashboard":    "डैशबोर्ड",
        "feedback":     "फीडबैक",
        "please":       "प्लीज",
        "check":        "चेक",
        "approve":      "अप्रूव",
        "approved":     "अप्रूव्ड",
        "conference":   "कॉन्फ्रेंस",
        "payment":      "पेमेंट",
        "message":      "मैसेज",
        "feature":      "फीचर",
        "button":       "बटन",
        "login":        "लॉगिन",
        "error":        "एरर",
        "server":       "सर्वर",
        "client":       "क्लाइंट",
        "review":       "रिव्यू",
        "update":       "अपडेट",
        "report":       "रिपोर्ट",
        "document":     "डॉक्यूमेंट",
        "urgent":       "अर्जेंट",
        "team":         "टीम",
        "call":         "कॉल",
        "done":         "डन",
        "ready":        "रेडी",
        "fix":          "फिक्स",
        "send":         "सेंड",
        "share":        "शेयर",
        "app":          "ऐप",
        "wifi":         "वाईफाई",
        "phone":        "फोन",
        "charge":       "चार्ज",
        "slow":         "स्लो",
        "ok":           "ओके",
        "okay":         "ओके",
        "thanks":       "थैंक्स",
        "sorry":        "सॉरी",
        "good":         "गुड",
        "great":        "ग्रेट",
        "perfect":      "परफेक्ट",
        "problem":      "प्रॉब्लम",
        "solution":     "सॉल्यूशन",
        "idea":         "आइडिया",
        "plan":         "प्लान",
    }

    def __init__(self):
        self._gruut_available = False
        self._try_load_gruut()

    def _try_load_gruut(self) -> None:
        """Check if gruut is available for G2P."""
        try:
            import gruut  # noqa
            self._gruut_available = True
            logger.info("[EphoneticPreprocessor] gruut available for English G2P.")
        except ImportError:
            logger.warning(
                "[EphoneticPreprocessor] gruut not installed. "
                "Falling back to override dictionary only. "
                "Install with: pip install gruut"
            )

    def ipa_to_devanagari(self, phonemes: List[str]) -> str:
        """
        Convert a list of IPA phoneme strings to a Devanagari string.

        The mapping handles:
          - Diphthongs (eɪ, aɪ, etc.)
          - Stress markers (ˈ, ˌ) embedded in phoneme strings
          - Affricate ligatures (t͡ʃ, d͡ʒ)

        Args:
            phonemes: List of IPA symbols from gruut, e.g. ['m', 'ˈi', 't', 'ɪ', 'ŋ']

        Returns:
            Devanagari string, e.g. "मीटिंग"
        """
        result = []
        for phoneme in phonemes:
            matched = False
            for ipa, dev in IPA_TO_DEVANAGARI:
                if phoneme == ipa:
                    result.append(dev)
                    matched = True
                    break
            if not matched:
                # Try substring match for composite phonemes
                remaining = phoneme
                while remaining:
                    found = False
                    for ipa, dev in IPA_TO_DEVANAGARI:
                        if remaining.startswith(ipa):
                            result.append(dev)
                            remaining = remaining[len(ipa):]
                            found = True
                            break
                    if not found:
                        # Unknown IPA symbol, skip
                        remaining = remaining[1:]

        devanagari = "".join(result)

        # Post-processing: add initial vowel carrier अ if word starts with a matra
        # (matras cannot stand alone in Devanagari — they need a base consonant)
        matras = {"ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "ा", "ॉ", "ाइ", "ाउ", "ॉइ"}
        if devanagari and any(devanagari.startswith(m) for m in matras):
            devanagari = "अ" + devanagari

        return devanagari

    def english_word_to_devanagari(self, word: str) -> Optional[str]:
        """
        Convert a single English word to its Devanagari phonetic equivalent.

        Lookup order:
          1. Override dictionary (hand-curated, most accurate)
          2. gruut G2P → IPA → Devanagari (automatic, CMU dict-based)
          3. None (caller falls back to IndicXlit)

        Args:
            word: English word in lowercase

        Returns:
            Devanagari string, or None if conversion failed
        """
        lower = word.lower().strip(".,!?;:'\"")

        # 1. Check override dictionary first
        if lower in self.OVERRIDES:
            return self.OVERRIDES[lower]

        # 2. Try gruut G2P
        if self._gruut_available:
            try:
                import gruut
                phoneme_list = []
                for sent in gruut.sentences(lower, lang="en-us"):
                    for w in sent:
                        if w.phonemes:
                            phoneme_list.extend(w.phonemes)

                if phoneme_list:
                    devanagari = self.ipa_to_devanagari(phoneme_list)
                    if devanagari.strip():
                        return devanagari
            except Exception as e:
                logger.debug(f"gruut G2P failed for '{word}': {e}")

        # 3. Give up — caller should fall back to IndicXlit
        return None

    def is_english_word(self, word: str) -> bool:
        """
        Heuristic: is this word likely English (not Hinglish/Hindi romanized)?

        English indicators:
          - In our override dictionary
          - Contains English vowel clusters not common in Hindi romanization
          - Starts with patterns common in English but rare in Hindi

        This is intentionally conservative — we only classify a word as
        English if we're fairly sure. Uncertain words go through IndicXlit.
        """
        lower = word.lower().strip(".,!?;:'\"")

        # Known English words (our dictionary)
        if lower in self.OVERRIDES:
            return True

        # Pure Hindi function words — never treat as English
        HINDI_WORDS = {
            "aaj", "mera", "meri", "hai", "hain", "kal", "aur", "par",
            "kya", "karo", "karna", "karte", "karke", "mein", "pe",
            "bhai", "yaar", "haan", "nahi", "tum", "aap", "main",
            "baje", "abhi", "ab", "phir", "lekin", "toh", "kyun",
            "kab", "kahan", "yeh", "woh", "iska", "uska", "apna",
            "apni", "please", "naya", "purana", "bahut", "thoda",
        }
        if lower in HINDI_WORDS:
            return False

        # Words with typical English letter combos that don't appear in Hindi romanization
        english_patterns = [
            r"tion$", r"ment$", r"ness$", r"ble$", r"ough", r"ight",
            r"ph[aeiou]", r"[aeiou]{3}", r"^ex", r"^pre", r"^pro",
            r"ck$", r"ss$", r"ff$", r"ll$",
        ]
        for pattern in english_patterns:
            if re.search(pattern, lower):
                return True

        return False
