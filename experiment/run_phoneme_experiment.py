# experiment/run_phoneme_experiment.py
# ─────────────────────────────────────────────────────────────────────────────
# Phoneme Preprocessing Experiment
#
# Goal: Compare whether gruut G2P → IPA → Devanagari gives better English
#       word pronunciation than IndicXlit's romanized-to-Devanagari approach.
#
# Generates 3 audio variants per sentence:
#
#   baseline   — current pipeline (everything through IndicXlit)
#   phoneme    — English words through gruut G2P → IPA → Devanagari,
#                Hindi words still through IndicXlit
#   hybrid     — phoneme strategy + falls back to IndicXlit for unknown words
#
# Each variant produces a WAV. Listen and compare especially on:
#   - English words with 'ai', 'ea', 'oo' vowels (email, deadline, laptop)
#   - Words with silent letters (check, please)
#   - Multi-syllable English words (presentation, conference)
#
# Usage:
#   .venv/bin/python experiment/run_phoneme_experiment.py
#   .venv/bin/python experiment/run_phoneme_experiment.py --all
#   .venv/bin/python experiment/run_phoneme_experiment.py --word "meeting"
#   .venv/bin/python experiment/run_phoneme_experiment.py --sentence "aaj mera meeting hai"
#
# Output:
#   experiment/outputs/phoneme_compare/
#     word_analysis.txt          — IPA breakdown per English word
#     sentence_01_.../
#       baseline.wav             — current pipeline
#       phoneme.wav              — G2P preprocessing
#       hybrid.wav               — G2P with IndicXlit fallback
#     summary.txt
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import io
import re
import wave
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR",       str(PROJECT_ROOT / ".cache" / "matplotlib"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phoneme_experiment")


# ── Test sentences ─────────────────────────────────────────────────────────────
# Selected to stress-test English phoneme coverage.
# Each sentence has English words with tricky pronunciations.

TEST_SENTENCES = [
    "aaj mera email check karna hai",            # email (silent 'ai' vowel)
    "please apna laptop charge karo",             # laptop (short 'a', diphthong)
    "presentation ready hai deadline se pehle",   # two complex English words
    "aaj mera meeting hai at 3 PM",              # meeting + abbreviation
    "yeh project ka schedule fix karo",           # schedule (tricky 'ch')
    "mera phone download nahi ho raha",           # download (ow diphthong)
    "office conference call hai abhi",            # conference (nasal + 'f')
    "please feedback bhejo review ke baad",       # feedback, review
    "client ko message bhej do abhi",             # message (silent 'e')
    "payment error aa raha hai dashboard mein",   # payment, error, dashboard
]

# Isolated English words for word-level analysis
TEST_WORDS = [
    "email", "laptop", "meeting", "presentation", "deadline",
    "schedule", "download", "conference", "feedback", "review",
    "message", "payment", "error", "dashboard", "check",
    "please", "approve", "update", "feature", "urgent",
]


# ── Audio helpers ──────────────────────────────────────────────────────────────

def save_wav(waveform: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(path), waveform, sample_rate)
    except ImportError:
        pcm = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())


def numpy_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    try:
        import soundfile as sf
        sf.write(buf, waveform, sample_rate, format="WAV")
    except ImportError:
        pcm = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.read()


# ── Pipeline implementations ───────────────────────────────────────────────────
# All three transliterators share ONE loaded HinglishTransliterator instance
# to avoid paying the 90-second IndicXlit load time multiple times.

def load_shared_xlit():
    """Load IndicXlit once, reuse across all transliterator variants."""
    from modules.hinglish_transliterator import HinglishTransliterator
    xlit = HinglishTransliterator()
    xlit.load()
    return xlit


def baseline_transliterate(text: str, xlit) -> str:
    """Current pipeline: everything through IndicXlit."""
    return xlit.transliterate(text)


def phoneme_transliterate(text: str, xlit, phoneme_preprocessor) -> str:
    """New pipeline: English words via gruut G2P → IPA → Devanagari."""
    import re
    from modules.hinglish_transliterator import ABBREVIATIONS, _convert_numbers_in_text

    for abbr, hindi in ABBREVIATIONS.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', hindi, text)
    text = _convert_numbers_in_text(text)

    tokens = text.split()
    result_tokens = []
    for token in tokens:
        match = re.match(r'^([^\w]*)([\w][\w\'-]*)([^\w]*)$', token)
        if match:
            prefix, word, suffix = match.groups()
            if re.search(r'[\u0900-\u097F]', word):
                result_tokens.append(token)
                continue
            if phoneme_preprocessor.is_english_word(word):
                devanagari = phoneme_preprocessor.english_word_to_devanagari(word)
                if devanagari:
                    result_tokens.append(prefix + devanagari + suffix)
                    continue
            devanagari = xlit._transliterate_word(word)
            result_tokens.append(prefix + devanagari + suffix)
        else:
            result_tokens.append(token)
    return " ".join(result_tokens)


def hybrid_transliterate(text: str, xlit, phoneme_preprocessor) -> str:
    """Conservative: only override-dict words use G2P, rest via IndicXlit."""
    import re
    from modules.hinglish_transliterator import ABBREVIATIONS, _convert_numbers_in_text

    for abbr, hindi in ABBREVIATIONS.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', hindi, text)
    text = _convert_numbers_in_text(text)

    tokens = text.split()
    result_tokens = []
    for token in tokens:
        match = re.match(r'^([^\w]*)([\w][\w\'-]*)([^\w]*)$', token)
        if match:
            prefix, word, suffix = match.groups()
            if re.search(r'[\u0900-\u097F]', word):
                result_tokens.append(token)
                continue
            lower = word.lower()
            # Only use phoneme when word is in hand-curated override dict
            if lower in phoneme_preprocessor.OVERRIDES:
                result_tokens.append(prefix + phoneme_preprocessor.OVERRIDES[lower] + suffix)
            else:
                dev = xlit._transliterate_word(word)
                result_tokens.append(prefix + dev + suffix)
        else:
            result_tokens.append(token)
    return " ".join(result_tokens)


# ── MMS synthesis ──────────────────────────────────────────────────────────────

def synthesize(devanagari_text: str, model, tokenizer) -> Tuple[Optional[np.ndarray], float]:
    """Synthesize Devanagari text using MMS-Hindi. Returns (waveform, duration_ms)."""
    import torch
    t = time.time()
    try:
        if not devanagari_text.strip():
            return None, 0.0
        inputs = tokenizer(devanagari_text, return_tensors="pt")
        model_inputs = {}
        for k, v in inputs.items():
            if k == "input_ids":
                model_inputs[k] = v.to(model.device).long()
            else:
                model_inputs[k] = v.to(model.device)
        with torch.no_grad():
            output = model(**model_inputs)
        waveform = output.waveform.squeeze().cpu().numpy().astype(np.float32)
        return waveform, (time.time() - t) * 1000
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return None, 0.0


# ── Word-level analysis ────────────────────────────────────────────────────────

def analyze_words(preprocessor, words: List[str]) -> Dict:
    """
    For each English word, show the IPA breakdown and resulting Devanagari.
    Useful to debug the IPA→Devanagari mapping quality.
    """
    results = {}
    for word in words:
        entry = {"word": word}

        # IPA from gruut
        try:
            import gruut
            phonemes = []
            for sent in gruut.sentences(word, lang="en-us"):
                for w in sent:
                    if w.phonemes:
                        phonemes.extend(w.phonemes)
            entry["ipa_phonemes"] = phonemes
            entry["ipa_string"] = "".join(phonemes)
        except Exception as e:
            entry["ipa_phonemes"] = []
            entry["ipa_string"] = f"ERROR: {e}"

        # Devanagari from phoneme preprocessor
        devanagari = preprocessor.english_word_to_devanagari(word)
        entry["devanagari_phoneme"] = devanagari or "(fallback to IndicXlit)"
        entry["source"] = "override_dict" if word in preprocessor.OVERRIDES else "gruut_g2p"

        results[word] = entry

    return results


# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment(sentences: List[str], output_dir: Path, analyze: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MMS model once
    logger.info("Loading facebook/mms-tts-hin...")
    t_load = time.time()
    try:
        import torch
        from transformers import VitsModel, AutoTokenizer

        model_path = str(PROJECT_ROOT / "models" / "tts" / "mms-hin")
        if not Path(model_path).exists():
            model_path = "facebook/mms-tts-hin"

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=Path(model_path).is_dir()
        )
        model = VitsModel.from_pretrained(
            model_path,
            local_files_only=Path(model_path).is_dir()
        ).eval()

        if torch.backends.mps.is_available():
            model = model.to("mps")
        else:
            model = model.to("cpu")

    except Exception as e:
        logger.error(f"Failed to load MMS model: {e}")
        raise

    load_time = time.time() - t_load
    logger.info(f"Model ready in {load_time:.1f}s")

    # Load transliterators — share ONE XlitEngine instance to avoid 3x 90s load
    logger.info("Loading transliterators (single shared XlitEngine)...")
    from modules.english_phoneme_preprocessor import EnglishPhonemePreprocessor
    xlit = load_shared_xlit()
    phoneme_pp = EnglishPhonemePreprocessor()

    SAMPLE_RATE = 16000

    VARIANTS = [
        ("baseline", lambda s: baseline_transliterate(s, xlit),
         "Current: everything through IndicXlit"),
        ("phoneme",  lambda s: phoneme_transliterate(s, xlit, phoneme_pp),
         "New: English words via gruut G2P → IPA → Devanagari"),
        ("hybrid",   lambda s: hybrid_transliterate(s, xlit, phoneme_pp),
         "Conservative: only override-dict words use G2P"),
    ]

    all_results = []

    for idx, sentence in enumerate(sentences, 1):
        logger.info(f"\n[{idx}/{len(sentences)}] '{sentence}'")

        safe_name = sentence[:40].replace(" ", "_").replace("/", "-")
        sentence_dir = output_dir / f"sentence_{idx:02d}_{safe_name}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        sentence_result = {"index": idx, "input": sentence, "variants": {}}

        print(f"\n{'─'*68}")
        print(f"  [{idx}] \"{sentence}\"")
        print(f"{'─'*68}")

        for vname, translit_fn, description in VARIANTS:
            t_start = time.time()

            # Transliterate
            try:
                devanagari = translit_fn(sentence)
            except Exception as e:
                logger.error(f"  [{vname}] Transliteration failed: {e}")
                sentence_result["variants"][vname] = {"success": False, "error": str(e)}
                print(f"  {vname:<12} → TRANSLIT FAILED: {e}")
                continue

            # Synthesize
            waveform, synth_ms = synthesize(devanagari, model, tokenizer)
            total_ms = (time.time() - t_start) * 1000

            if waveform is None or len(waveform) == 0:
                sentence_result["variants"][vname] = {
                    "success": False,
                    "devanagari": devanagari,
                    "error": "empty waveform"
                }
                print(f"  {vname:<12} → SYNTH FAILED  (Devanagari: {devanagari})")
                continue

            audio_dur_ms = (len(waveform) / SAMPLE_RATE) * 1000
            rtf = total_ms / audio_dur_ms if audio_dur_ms > 0 else 0

            # Save WAV
            out_path = sentence_dir / f"{vname}.wav"
            save_wav(waveform, SAMPLE_RATE, out_path)

            sentence_result["variants"][vname] = {
                "success":      True,
                "description":  description,
                "devanagari":   devanagari,
                "total_ms":     round(total_ms, 1),
                "audio_dur_ms": round(audio_dur_ms, 1),
                "rtf":          round(rtf, 3),
                "output_file":  str(out_path),
            }

            print(f"  {vname:<12} → {total_ms:>5.0f}ms  RTF={rtf:.3f}  ✓")
            print(f"               Devanagari: {devanagari}")

        all_results.append(sentence_result)

    # ── Word-level analysis ────────────────────────────────────────────────────
    if analyze:
        from modules.english_phoneme_preprocessor import EnglishPhonemePreprocessor
        pp = EnglishPhonemePreprocessor()

        logger.info("\nRunning word-level IPA analysis...")
        word_analysis = analyze_words(pp, TEST_WORDS)

        analysis_path = output_dir / "word_analysis.txt"
        lines = ["=" * 68, "  ENGLISH WORD PHONEME ANALYSIS", "=" * 68, ""]
        lines.append(f"  {'Word':<16} {'IPA':<30} {'Devanagari':<20} Source")
        lines.append(f"  {'─'*16} {'─'*30} {'─'*20} {'─'*12}")
        for word, data in word_analysis.items():
            ipa = data.get("ipa_string", "")[:28]
            dev = data.get("devanagari_phoneme", "")[:18]
            src = data.get("source", "")
            lines.append(f"  {word:<16} {ipa:<30} {dev:<20} {src}")
        lines.append("")

        analysis_text = "\n".join(lines)
        analysis_path.write_text(analysis_text, encoding="utf-8")
        print("\n" + analysis_text)
        logger.info(f"Word analysis saved → {analysis_path}")

    # ── Summary report ─────────────────────────────────────────────────────────
    report = {
        "experiment": "english_phoneme_preprocessing",
        "model": "facebook/mms-tts-hin",
        "variants": [
            {"name": v[0], "description": v[2]} for v in VARIANTS
        ],
        "model_load_time_s": round(load_time, 2),
        "sentences": all_results,
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Report saved → {report_path}")

    # ── Summary text ───────────────────────────────────────────────────────────
    summary_lines = [
        "=" * 68,
        "  PHONEME PREPROCESSING EXPERIMENT SUMMARY",
        "=" * 68,
        "",
        "VARIANTS:",
        "",
    ]
    for vname, _, desc in VARIANTS:
        summary_lines.append(f"  {vname}")
        summary_lines.append(f"     {desc}")
        summary_lines.append("")

    summary_lines += [
        "─" * 68,
        "WHAT TO LISTEN FOR:",
        "",
        "  baseline vs phoneme:",
        "    → English vowels: does 'email' sound like 'ee-mail' or 'em-ail'?",
        "    → Diphthongs: does 'laptop' sound like 'laep-top' or 'lap-top'?",
        "    → Fricatives: does 'schedule' start with 'sh' or 'sk'?",
        "",
        "  phoneme vs hybrid:",
        "    → Does the pure phoneme strategy handle all English words?",
        "    → Which words does hybrid fall back to IndicXlit for?",
        "",
        "─" * 68,
        "KEY QUESTION:",
        "  Do English words sound MORE natural with gruut G2P preprocessing?",
        "  If YES → integrate EnglishPhonemePreprocessor into main pipeline.",
        "  If NO  → investigate which IPA→Devanagari mappings need tuning.",
        "=" * 68,
    ]

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "\n".join(summary_lines))
    logger.info(f"Summary saved → {summary_path}")
    logger.info(f"\nOutput directory: {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phoneme preprocessing experiment — compare G2P vs IndicXlit for English"
    )
    parser.add_argument("--sentence", type=str, default=None,
                        help="Single Hinglish sentence to test")
    parser.add_argument("--word", type=str, default=None,
                        help="Single English word — shows IPA breakdown and Devanagari output")
    parser.add_argument("--all", action="store_true",
                        help="Run all test sentences")
    parser.add_argument("--no-analyze", action="store_true",
                        help="Skip word-level IPA analysis")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "outputs" / "phoneme_compare"))

    args = parser.parse_args()

    if args.word:
        # Quick word-level demo
        from modules.english_phoneme_preprocessor import EnglishPhonemePreprocessor
        pp = EnglishPhonemePreprocessor()
        try:
            import gruut
            for sent in gruut.sentences(args.word, lang="en-us"):
                for w in sent:
                    print(f"Word:      {args.word}")
                    print(f"IPA:       {''.join(w.phonemes)}")
                    print(f"Phonemes:  {w.phonemes}")
                    dev = pp.english_word_to_devanagari(args.word)
                    print(f"Devanagari: {dev}")
        except Exception as e:
            print(f"Error: {e}")
        return

    if args.sentence:
        sentences = [args.sentence]
    elif args.all:
        sentences = TEST_SENTENCES
    else:
        print("No --sentence or --all specified. Running quick test (3 sentences).")
        print("Use --all for the full suite.\n")
        sentences = TEST_SENTENCES[:3]

    run_experiment(
        sentences=sentences,
        output_dir=Path(args.output_dir),
        analyze=not args.no_analyze,
    )


if __name__ == "__main__":
    main()
