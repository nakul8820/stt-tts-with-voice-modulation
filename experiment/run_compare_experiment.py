# experiment/run_compare_experiment.py
# ─────────────────────────────────────────────────────────────────────────────
# Hinglish MMS Stitch — Full Comparison Experiment
#
# Generates 6 audio variants per sentence across all combinations of:
#   3 voice consistency strategies × 2 English model approaches
#
# Variants:
#   v1_raw_eng            raw stitching   + MMS-English for EN words
#   v2_raw_phonetic       raw stitching   + Hindi phonetic for EN words
#   v3_normalized_eng     RMS normalize   + MMS-English for EN words
#   v4_normalized_phonetic RMS normalize  + Hindi phonetic for EN words
#   v5_tuned_eng          tuned params    + MMS-English for EN words
#   v6_tuned_phonetic     tuned params    + Hindi phonetic for EN words
#
# Output structure:
#   experiment/outputs/stitch_compare/
#     sentence_01_aaj_mera_meeting/
#       v1_raw_eng.wav
#       v2_raw_phonetic.wav
#       v3_normalized_eng.wav
#       v4_normalized_phonetic.wav
#       v5_tuned_eng.wav
#       v6_tuned_phonetic.wav
#     sentence_02_.../
#       ...
#     report.json
#     summary.txt
#
# Usage:
#   python experiment/run_compare_experiment.py
#   python experiment/run_compare_experiment.py --all
#   python experiment/run_compare_experiment.py --text "aaj meeting hai at 3 PM"
#   python experiment/run_compare_experiment.py --all --play
#   python experiment/run_compare_experiment.py --sentence 2   (run one sentence)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import time
import wave
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_experiment")


# ── Test sentences ─────────────────────────────────────────────────────────────
# Chosen to cover different Hinglish patterns:
#   - simple embedding, heavy embedding, mostly Hindi, mostly English, numbers

TEST_SENTENCES = [
    "aaj mera meeting hai at 3 PM",
    "please apna email check karo",
    "kya tum free ho abhi",
    "bhai kal team call hai aur mujhe presentation ready karni hai",
    "mera phone charge nahi ho raha",
    "yeh project deadline kal hai",
    "haan boss ne approve kar diya proposal",
    "laptop slow chal raha hai aaj",
    "iss issue ko fix karna bahut urgent hai",
    "meeting 5 baje conference room mein hai",
    "mera order dispatch ho gaya kya",
    "dashboard mein login button click karo",
    "payment error aa raha hai system mein",
    "yeh naya feature bahut useful hai",
    "client ko message bhej do abhi",
]


# ── Variant definitions ────────────────────────────────────────────────────────

VARIANTS = [
    {
        "id":       "v1",
        "name":     "raw_eng",
        "label":    "Raw stitching + MMS-English",
        "strategy": "raw",
        "phonetic": False,
        "tuned":    False,
        "description": (
            "No normalization. English words go to MMS-English model. "
            "You'll hear two distinct voices but natural English pronunciation."
        ),
    },
    {
        "id":       "v2",
        "name":     "raw_phonetic",
        "label":    "Raw stitching + Hindi phonetic English",
        "strategy": "raw",
        "phonetic": True,
        "tuned":    False,
        "description": (
            "No normalization. English words phonetically mapped to Devanagari "
            "and spoken by MMS-Hindi. One consistent voice, Indian-accented English."
        ),
    },
    {
        "id":       "v3",
        "name":     "normalized_eng",
        "label":    "RMS normalized + MMS-English",
        "strategy": "normalized",
        "phonetic": False,
        "tuned":    False,
        "description": (
            "Volume-matched between models. English words go to MMS-English. "
            "Reduces loudness jumps. Two voices but at same volume level."
        ),
    },
    {
        "id":       "v4",
        "name":     "normalized_phonetic",
        "label":    "RMS normalized + Hindi phonetic English",
        "strategy": "normalized",
        "phonetic": True,
        "tuned":    False,
        "description": (
            "Volume-matched. English words phonetically spoken by MMS-Hindi. "
            "One voice, Indian accent, volume consistent throughout."
        ),
    },
    {
        "id":       "v5",
        "name":     "tuned_eng",
        "label":    "VITS tuned + MMS-English",
        "strategy": "tuned",
        "phonetic": False,
        "tuned":    True,
        "description": (
            "VITS noise/length parameters tuned per model + brightness matching. "
            "Tries to blend two model voices toward common speaking style. "
            "English words from MMS-English."
        ),
    },
    {
        "id":       "v6",
        "name":     "tuned_phonetic",
        "label":    "VITS tuned + Hindi phonetic English",
        "strategy": "tuned",
        "phonetic": True,
        "tuned":    True,
        "description": (
            "VITS tuned parameters + brightness matching + phonetic English. "
            "One Hindi speaker throughout, tuned for consistency. "
            "Best shot at a natural unified voice."
        ),
    },
]

# VITS params per model — used for tuned variants
TUNED_PARAMS = {
    "hi": {"noise_scale": 0.4, "noise_scale_w": 0.6, "length_scale": 1.05},
    "en": {"noise_scale": 0.5, "noise_scale_w": 0.7, "length_scale": 1.1},
}
DEFAULT_PARAMS = {"noise_scale": 0.667, "noise_scale_w": 0.8, "length_scale": 1.0}


# ── I/O helpers ───────────────────────────────────────────────────────────────

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


def play_audio(waveform: np.ndarray, sample_rate: int, label: str = "") -> None:
    try:
        import sounddevice as sd
        if label:
            print(f"    ▶  Playing: {label}")
        sd.play(waveform, samplerate=sample_rate)
        sd.wait()
    except ImportError:
        print("    sounddevice not installed — pip install sounddevice")
    except Exception as e:
        print(f"    Playback failed: {e}")


# ── Core synthesis for one variant ────────────────────────────────────────────

def synthesize_variant(
    segments: List[Dict],
    variant: Dict,
    engine,
    stitcher_fn,
) -> Tuple[Optional[np.ndarray], float, Dict]:
    """
    Synthesize one variant of a sentence.

    Returns:
        (waveform, total_ms, timing_info)
    """
    use_phonetic = variant["phonetic"]
    use_tuned    = variant["tuned"]
    strategy     = variant["strategy"]

    chunks = []
    timing = {"segments": []}
    t_start = time.time()

    for seg in segments:
        lang     = seg["lang"]
        seg_text = seg["text"]

        # Choose VITS params
        if use_tuned:
            p = TUNED_PARAMS.get(lang, DEFAULT_PARAMS)
        else:
            p = DEFAULT_PARAMS

        t_seg = time.time()
        wav = engine.synthesize(
            text=seg_text,
            lang=lang,
            use_phonetic_english=use_phonetic,
            noise_scale=p["noise_scale"],
            noise_scale_w=p["noise_scale_w"],
            length_scale=p["length_scale"],
        )
        seg_ms = (time.time() - t_seg) * 1000

        timing["segments"].append({
            "lang": lang,
            "text": seg_text,
            "synth_ms": round(seg_ms, 1),
            "model_used": "mms-hin" if (lang == "hi" or use_phonetic) else "mms-eng",
        })

        if wav is not None and len(wav) > 0:
            chunks.append((wav, lang))

    if not chunks:
        return None, 0.0, timing

    waveform = stitcher_fn(chunks, strategy=strategy, sr=engine.SAMPLE_RATE)
    total_ms = (time.time() - t_start) * 1000
    return waveform, total_ms, timing


# ── Per-sentence experiment ────────────────────────────────────────────────────

def run_sentence(
    idx: int,
    text: str,
    engine,
    stitcher_fn,
    segment_fn,
    output_dir: Path,
    play: bool = False,
) -> Dict:
    """Generate all 6 variants for one sentence."""

    # Create per-sentence subfolder
    safe = text[:35].replace(" ", "_").replace("/", "-")
    sentence_dir = output_dir / f"sentence_{idx:02d}_{safe}"
    sentence_dir.mkdir(parents=True, exist_ok=True)

    # Segment once — shared by all variants
    segments = segment_fn(text)

    print(f"\n{'─'*68}")
    print(f"  [{idx}] \"{text}\"")
    print(f"       Segments: " + " | ".join(
        f"[{'HI' if s['lang']=='hi' else 'EN'}:{s['text']}]" for s in segments
    ))
    print(f"{'─'*68}")

    sentence_result = {
        "index": idx,
        "input": text,
        "segments": segments,
        "variants": {},
    }

    sr = engine.SAMPLE_RATE

    for variant in VARIANTS:
        vname = f"{variant['id']}_{variant['name']}"

        try:
            waveform, total_ms, timing = synthesize_variant(
                segments, variant, engine, stitcher_fn
            )

            if waveform is None or len(waveform) == 0:
                print(f"  {vname:<28} → EMPTY OUTPUT")
                sentence_result["variants"][vname] = {"success": False, "error": "empty output"}
                continue

            audio_dur_ms = (len(waveform) / sr) * 1000
            rtf = total_ms / audio_dur_ms if audio_dur_ms > 0 else 0

            # Save wav
            out_path = sentence_dir / f"{vname}.wav"
            save_wav(waveform, sr, out_path)

            sentence_result["variants"][vname] = {
                "success":        True,
                "label":          variant["label"],
                "total_ms":       round(total_ms, 1),
                "audio_dur_ms":   round(audio_dur_ms, 1),
                "rtf":            round(rtf, 3),
                "output_file":    str(out_path),
                "timing":         timing,
            }

            status = f"{total_ms:>6.0f}ms  RTF={rtf:.3f}"
            print(f"  {vname:<30} → {status}  ✓")

            if play:
                play_audio(waveform, sr, variant["label"])

        except Exception as e:
            logger.error(f"  {vname} failed: {e}", exc_info=True)
            sentence_result["variants"][vname] = {"success": False, "error": str(e)}
            print(f"  {vname:<30} → FAILED: {e}")

    return sentence_result


# ── Summary text ───────────────────────────────────────────────────────────────

def write_summary(results: List[Dict], output_dir: Path, load_time: float) -> None:
    """Write a human-readable summary so you know exactly what to listen for."""
    lines = []
    lines.append("=" * 68)
    lines.append("  HINGLISH MMS STITCH — COMPARISON EXPERIMENT SUMMARY")
    lines.append("=" * 68)
    lines.append("")
    lines.append("WHAT TO LISTEN FOR PER VARIANT:")
    lines.append("")
    for v in VARIANTS:
        lines.append(f"  {v['id']}_{v['name']}")
        lines.append(f"     {v['description']}")
        lines.append("")

    lines.append("─" * 68)
    lines.append("KEY QUESTIONS TO ANSWER WHILE LISTENING:")
    lines.append("")
    lines.append("  1. v1 vs v3 vs v5 (all use MMS-English)")
    lines.append("     → Does normalization/tuning reduce the voice-switch jarring?")
    lines.append("")
    lines.append("  2. v2 vs v4 vs v6 (all use Hindi phonetic for English)")
    lines.append("     → Does consistent voice outweigh slightly accented English?")
    lines.append("")
    lines.append("  3. Odd (eng) vs Even (phonetic) — same strategy, different model")
    lines.append("     v1 vs v2, v3 vs v4, v5 vs v6")
    lines.append("     → Which English approach sounds more natural?")
    lines.append("")
    lines.append("─" * 68)
    lines.append("LATENCY SUMMARY (ms, total synthesis time):")
    lines.append("")

    # Compute per-variant averages
    variant_times = {f"{v['id']}_{v['name']}": [] for v in VARIANTS}
    for r in results:
        for vname, vdata in r.get("variants", {}).items():
            if vdata.get("success"):
                variant_times[vname].append(vdata["total_ms"])

    lines.append(f"  {'Variant':<32} {'Avg':>8} {'Min':>8} {'Max':>8}")
    lines.append(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8}")
    for vname, times in variant_times.items():
        if times:
            lines.append(
                f"  {vname:<32} {sum(times)/len(times):>7.0f}ms "
                f"{min(times):>7.0f}ms {max(times):>7.0f}ms"
            )

    lines.append("")
    lines.append(f"  Model load time: {load_time:.1f}s")
    lines.append("")
    lines.append("─" * 68)
    lines.append("OUTPUT FILES:")
    lines.append(f"  {output_dir}")
    lines.append("")
    lines.append("  Each sentence folder contains 6 .wav files:")
    lines.append("  v1_raw_eng.wav, v2_raw_phonetic.wav, v3_normalized_eng.wav,")
    lines.append("  v4_normalized_phonetic.wav, v5_tuned_eng.wav, v6_tuned_phonetic.wav")
    lines.append("=" * 68)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    logger.info(f"Summary saved → {summary_path}")


# ── Main runner ────────────────────────────────────────────────────────────────

def run_experiment(
    sentences: List[str],
    output_dir: Path,
    play: bool = False,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load engine once — shared across all sentences and variants
    from modules.mms_engine import MmsEngine
    from modules.hinglish_segmenter import segment_hinglish
    from modules.audio_stitcher import stitch

    engine = MmsEngine()
    logger.info("Loading MMS models (first run ~30s, cached after)...")
    t_load = time.time()
    engine.load()
    load_time = time.time() - t_load

    print(f"\n{'═'*68}")
    print(f"  Models loaded in {load_time:.1f}s")
    print(f"  Generating {len(VARIANTS)} variants × {len(sentences)} sentences "
          f"= {len(VARIANTS)*len(sentences)} audio files")
    print(f"  Output → {output_dir}")
    print(f"{'═'*68}")

    all_results = []

    for i, sentence in enumerate(sentences, start=1):
        result = run_sentence(
            idx=i,
            text=sentence,
            engine=engine,
            stitcher_fn=stitch,
            segment_fn=segment_hinglish,
            output_dir=output_dir,
            play=play,
        )
        all_results.append(result)

    # Save full JSON report
    report = {
        "experiment": "hinglish_mms_stitch_compare",
        "models": {
            "hindi":   "facebook/mms-tts-hin",
            "english": "facebook/mms-tts-eng",
        },
        "variants": [{
            "id": v["id"],
            "name": v["name"],
            "label": v["label"],
            "description": v["description"],
        } for v in VARIANTS],
        "model_load_time_s": round(load_time, 2),
        "sentences": all_results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Full report → {report_path}")

    # Write human-readable summary
    write_summary(all_results, output_dir, load_time)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hinglish MMS stitch comparison — generates all 6 variants per sentence"
    )
    parser.add_argument("--text", type=str, default=None,
                        help="Single Hinglish sentence to test")
    parser.add_argument("--all", action="store_true",
                        help="Run all default test sentences")
    parser.add_argument("--sentence", type=int, default=None,
                        help="Run one sentence by index (1-based) from default list")
    parser.add_argument("--file", type=str, default=None,
                        help="Text file with one sentence per line")
    parser.add_argument("--play", action="store_true",
                        help="Play each audio variant after synthesis (requires sounddevice)")
    parser.add_argument("--output-dir", type=str,
                        default=str(
                            Path(__file__).parent / "outputs" / "stitch_compare"
                        ),
                        help="Output directory for .wav files and report")

    args = parser.parse_args()

    if args.text:
        sentences = [args.text]
    elif args.sentence:
        idx = args.sentence - 1
        if not (0 <= idx < len(TEST_SENTENCES)):
            print(f"Error: --sentence must be 1–{len(TEST_SENTENCES)}")
            sys.exit(1)
        sentences = [TEST_SENTENCES[idx]]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            sentences = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    elif args.all:
        sentences = TEST_SENTENCES
    else:
        # Default: 2 sentences as a quick smoke test
        print("No --text or --all specified. Running quick test (2 sentences).")
        print("Use --all for the full suite.\n")
        sentences = TEST_SENTENCES[:2]

    run_experiment(
        sentences=sentences,
        output_dir=Path(args.output_dir),
        play=args.play,
    )


if __name__ == "__main__":
    main()
