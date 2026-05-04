# download_mms_models.py
# ─────────────────────────────────────────────────────────────────────────────
# Download MMS-TTS Hindi model and IndicXlit transliteration model
# to a local models/ directory so the system runs fully offline.
#
# Mirrors the pattern of your existing download_models.py.
#
# Usage:
#   python download_mms_models.py
#   python download_mms_models.py --models-dir /path/to/custom/dir
#
# What gets downloaded:
#   models/tts/mms-tts-hin/     ← Facebook MMS TTS Hindi (~80MB)
#   models/transliteration/     ← AI4Bharat IndicXlit Hindi (~136MB)
#
# Total: ~216MB (one-time download, runs offline after)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("download")


def sizeof_fmt(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def download_mms_hindi(models_dir: Path) -> Path:
    """
    Download facebook/mms-tts-hin from HuggingFace Hub to local directory.
    Returns the local model path.
    """
    model_id = "facebook/mms-tts-hin"
    local_path = models_dir / "tts" / "mms-tts-hin"

    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"MMS-TTS Hindi already downloaded at: {local_path}")
        return local_path

    logger.info(f"Downloading {model_id}...")
    logger.info(f"  Destination: {local_path}")

    try:
        from huggingface_hub import snapshot_download
        start = time.time()
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
            # Only download what we need for inference
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        elapsed = time.time() - start
        logger.info(f"  Downloaded in {elapsed:.1f}s → {local_path}")
        return local_path

    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )


def download_indicxlit(models_dir: Path) -> None:
    """
    Install ai4bharat-transliteration package which auto-downloads
    its models on first use, or trigger the download explicitly.

    IndicXlit manages its own model cache via the package itself.
    We just verify the package is installed and the model is accessible.
    """
    translit_dir = models_dir / "transliteration"
    translit_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Verifying ai4bharat-transliteration installation...")

    try:
        from ai4bharat.transliteration import XlitEngine
        logger.info("  Package found. Triggering Hindi model download/cache...")
        start = time.time()

        # Instantiating XlitEngine downloads the model if not cached.
        # We use a try block because first download can take a moment.
        engine = XlitEngine(lang_code="hi", beam_width=4)
        elapsed = time.time() - start

        # Warm-up inference to confirm it works
        test = engine.translit_word("test", lang_code="hi")
        logger.info(f"  IndicXlit ready in {elapsed:.1f}s (warm-up: '{test}')")

    except ImportError:
        logger.error("ai4bharat-transliteration not installed.")
        logger.error("Run: pip install ai4bharat-transliteration")
        logger.error("  or: pip install ai4bharat-transliteration[gpu]  (for GPU)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"IndicXlit setup failed: {e}")
        sys.exit(1)


def verify_mms_model(local_path: Path) -> None:
    """Quick verification that the downloaded MMS model loads correctly."""
    logger.info("Verifying MMS-TTS Hindi model...")
    try:
        from transformers import VitsModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(local_path))
        model = VitsModel.from_pretrained(str(local_path))
        sr = model.config.sampling_rate
        logger.info(f"  Model OK — sample rate: {sr}Hz, params: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        logger.error(f"  Model verification failed: {e}")
        sys.exit(1)


def print_summary(models_dir: Path) -> None:
    """Print final summary of what was downloaded."""
    print("\n" + "═" * 60)
    print("  Download Complete")
    print("═" * 60)

    mms_path = models_dir / "tts" / "mms-tts-hin"
    if mms_path.exists():
        size = sum(f.stat().st_size for f in mms_path.rglob("*") if f.is_file())
        print(f"  MMS-TTS Hindi:   {mms_path}")
        print(f"                   {sizeof_fmt(size)}")

    print(f"  IndicXlit:       managed by ai4bharat package")
    print()
    print("  To run the experiment:")
    print("    python experiment/run_experiment.py --all")
    print()
    print("  To run a single sentence:")
    print('    python experiment/run_experiment.py --text "aaj meeting hai"')
    print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download MMS-TTS Hindi and IndicXlit models for offline use"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Local directory to store models (default: ./models)"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-download verification step"
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Models directory: {models_dir}")
    logger.info("Starting downloads...\n")

    # 1. Download MMS-TTS Hindi
    mms_path = download_mms_hindi(models_dir)

    # 2. Download / verify IndicXlit
    download_indicxlit(models_dir)

    # 3. Verify MMS model loads
    if not args.skip_verify:
        verify_mms_model(mms_path)

    # 4. Print summary
    print_summary(models_dir)


if __name__ == "__main__":
    main()
