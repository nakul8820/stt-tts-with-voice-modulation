# experiment/run_experiment.py
# ─────────────────────────────────────────────────────────────────────────────
# Standalone Experiment Runner
#
# Tests the Hinglish MMS TTS pipeline WITHOUT needing FastAPI or your full
# project running. Run this directly to hear results and measure latency.
#
# Usage:
#   cd <project_root>
#   python experiment/run_experiment.py
#
#   # Test a single sentence:
#   python experiment/run_experiment.py --text "aaj mera meeting hai at 3 PM"
#
#   # Test all sentences from the test file:
#   python experiment/run_experiment.py --all
#
#   # Play audio after generating (requires sounddevice):
#   python experiment/run_experiment.py --all --play
#
# Output:
#   experiment/outputs/<sentence_index>.wav   — one .wav per test sentence
#   experiment/outputs/report.json            — latency + RTF for each sentence
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

# ── Path setup so imports work from project root ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("experiment")


# ── Test sentences ─────────────────────────────────────────────────────────────
# A mix of real-world Hinglish patterns:
#   • Pure Hindi romanized
#   • Pure English words embedded
#   • Mixed mid-sentence switching
#   • Numbers, times, abbreviations
#   • Longer conversational sentences

DEFAULT_TEST_SENTENCES = [
    # Basic Hinglish
    "aaj mera meeting hai at 3 PM",
    "kal office nahi jaana mujhe",
    "kya tum free ho abhi",

    # English words mid-sentence
    "mera phone charge nahi ho raha",
    "please apna email check karo",
    "yeh project deadline kal hai",

    # Numbers and times
    "meeting 5 baje hai conference room mein",
    "mujhe 2 ghante aur chahiye kaam khatam karne ke liye",

    # Longer conversational
    "bhai kal team call hai aur mujhe presentation ready karni hai",
    "kya tum mujhe help kar sakte ho is problem ko solve karne mein",

    # More natural code-switching
    "mera internet slow chal raha hai aaj",
    "yeh file download ho gayi kya",
    "haan boss ne approve kar diya proposal",

    # Slightly formal
    "aapka order dispatch ho gaya hai aur 2 din mein deliver ho jayega",
    "iss issue ko fix karna bahut urgent hai",
]


def save_wav(waveform: np.ndarray, sample_rate: int, filepath: Path) -> None:
    """Save numpy float32 waveform to .wav file."""
    try:
        import soundfile as sf
        sf.write(str(filepath), waveform, sample_rate)
    except ImportError:
        # Fallback: write raw PCM wav manually
        import struct
        import wave
        # Convert float32 to int16
        pcm = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())


def play_audio(waveform: np.ndarray, sample_rate: int) -> None:
    """Play audio through speakers (requires sounddevice)."""
    try:
        import sounddevice as sd
        logger.info("  Playing audio...")
        sd.play(waveform, samplerate=sample_rate)
        sd.wait()
    except ImportError:
        logger.warning("  sounddevice not installed — skipping playback")
        logger.warning("  Install with: pip install sounddevice")
    except Exception as e:
        logger.warning(f"  Playback failed: {e}")


def print_banner():
    print("\n" + "═" * 65)
    print("  Hinglish MMS TTS Experiment")
    print("  Pipeline: Romanized Hinglish → IndicXlit → MMS-Hindi → Audio")
    print("═" * 65 + "\n")


def print_report(results: list) -> None:
    """Print a formatted latency report to console."""
    print("\n" + "─" * 65)
    print(f"  {'#':<4} {'Text (truncated)':<35} {'Total':>8} {'RTF':>6}")
    print("─" * 65)

    total_times = []
    for r in results:
        if r.get("success"):
            total_ms = r["total_ms"]
            rtf = r["rtf"]
            total_times.append(total_ms)
            text_preview = r["input"][:33] + ".." if len(r["input"]) > 35 else r["input"]
            status = f"{total_ms:>7.0f}ms  {rtf:>5.3f}"
        else:
            text_preview = r["input"][:33] + ".." if len(r["input"]) > 35 else r["input"]
            status = "  FAILED"
        print(f"  {r['index']:<4} {text_preview:<35} {status}")

    if total_times:
        print("─" * 65)
        print(f"  {'Average':<40} {sum(total_times)/len(total_times):>7.0f}ms")
        print(f"  {'Min':<40} {min(total_times):>7.0f}ms")
        print(f"  {'Max':<40} {max(total_times):>7.0f}ms")
        print(f"  {'Successful':<40} {len(total_times)}/{len(results)}")
    print("─" * 65 + "\n")


def run_experiment(sentences: list, play: bool = False, output_dir: Path = None):
    """
    Run the full Hinglish TTS pipeline on a list of sentences.
    Saves .wav files and returns a latency report.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load provider ──────────────────────────────────────────────────────────
    logger.info("Initializing provider...")
    from providers.mms_hinglish_provider import MmsHinglishProvider

    provider = MmsHinglishProvider()

    logger.info("Loading models (this takes ~30s on first run, faster after caching)...")
    load_start = time.time()
    provider.load()
    load_time = time.time() - load_start
    logger.info(f"Models loaded in {load_time:.1f}s\n")

    # ── Run sentences ──────────────────────────────────────────────────────────
    results = []
    print_banner()

    for i, sentence in enumerate(sentences):
        print(f"[{i+1}/{len(sentences)}] Input:  \"{sentence}\"")

        result = {
            "index": i + 1,
            "input": sentence,
            "success": False
        }

        try:
            t_start = time.time()
            waveform, sample_rate = provider.synthesize(sentence)
            total_ms = (time.time() - t_start) * 1000

            audio_duration_ms = (len(waveform) / sample_rate) * 1000
            rtf = total_ms / audio_duration_ms if audio_duration_ms > 0 else 0

            # Save .wav
            out_path = output_dir / f"{i+1:02d}_{sentence[:30].replace(' ', '_')}.wav"
            save_wav(waveform, sample_rate, out_path)

            # Transliterated form (for inspection)
            devanagari = provider._transliterator.transliterate(sentence)

            result.update({
                "success": True,
                "devanagari": devanagari,
                "total_ms": total_ms,
                "audio_duration_ms": audio_duration_ms,
                "rtf": rtf,
                "output_file": str(out_path),
                "sample_rate": sample_rate,
                "samples": len(waveform),
            })

            print(f"         Devanagari: \"{devanagari}\"")
            print(f"         Output:    {out_path.name}")
            print(f"         Latency:   {total_ms:.0f}ms | Audio: {audio_duration_ms:.0f}ms | RTF: {rtf:.3f}")

            if play:
                play_audio(waveform, sample_rate)

        except Exception as e:
            logger.error(f"  Failed: {e}", exc_info=True)
            result["error"] = str(e)

        print()
        results.append(result)

    # ── Save report ────────────────────────────────────────────────────────────
    report = {
        "model": provider.MMS_MODEL_ID,
        "transliterator": "ai4bharat/IndicXlit",
        "model_load_time_s": round(load_time, 2),
        "results": results,
    }
    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(results)
    logger.info(f"Report saved to: {report_path}")
    logger.info(f"WAV files saved to: {output_dir}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test the Hinglish MMS TTS pipeline"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Single Hinglish sentence to synthesize"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all default test sentences"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a text file with one sentence per line"
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play each audio output after synthesis (requires sounddevice)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent / "outputs"),
        help="Directory to save .wav files and report.json"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speech speed factor (0.5 = slow, 1.0 = normal, 1.5 = fast)"
    )

    args = parser.parse_args()

    # Determine which sentences to run
    if args.text:
        sentences = [args.text]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            sentences = [l.strip() for l in f if l.strip()]
    elif args.all:
        sentences = DEFAULT_TEST_SENTENCES
    else:
        # Default: run first 3 sentences as a quick smoke test
        logger.info("No --text or --all specified. Running quick test (3 sentences).")
        logger.info("Use --all to run all test sentences.\n")
        sentences = DEFAULT_TEST_SENTENCES[:3]

    run_experiment(
        sentences=sentences,
        play=args.play,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
