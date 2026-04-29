# download_models.py
# ─────────────────────────────────────────────────────────────
# Run this ONCE while connected to the internet.
# It reads config.yaml and downloads only the models you need.
# After this, everything runs 100% offline.
#
# Usage:
#   python download_models.py
#
# To pre-download ALL models for experimentation:
#   python download_models.py --all
# ─────────────────────────────────────────────────────────────

import sys
import yaml
import os

# Set custom paths for models to avoid permission errors in system directories
# We put them inside the project folder so they are easy to manage.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TTS_HOME = os.path.join(MODELS_DIR, "tts")

os.environ["TTS_HOME"] = TTS_HOME
# XDG_DATA_HOME is also used by some versions of Coqui
os.environ["XDG_DATA_HOME"] = MODELS_DIR

# Ensure directories exist
os.makedirs(TTS_HOME, exist_ok=True)


def load_config():
    """Load config.yaml and return as dict."""
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def download_whisper(model_size: str):
    """Download a faster-whisper model. Cached to ~/.cache/huggingface"""
    print(f"\n[Download] faster-whisper: {model_size}")
    print("  This may take a few minutes on first run...")
    try:
        from faster_whisper import WhisperModel
        # Constructing the model triggers the download if not cached.
        # After this it loads from ~/.cache/huggingface/hub — no internet.
        WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"  Done: faster-whisper {model_size}")
    except ImportError:
        print(f"  [Error] faster-whisper not installed. Skipping download for {model_size}.")
        print("  If you intend to use whisper_faster, run: pip install faster-whisper")


def download_coqui_xtts():
    """Download Coqui XTTS v2. Cached to models/tts/ (~1.8GB)"""
    print(f"\n[Download] Coqui XTTS v2 (~1.8GB) ...")
    print(f"  Target directory: {TTS_HOME}")
    from TTS.api import TTS
    # Setting use_gpu=False for download to avoid CUDA/MPS init issues during download
    TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)
    print("  Done: Coqui XTTS v2")


def download_indic_parler():
    """Download Indic Parler-TTS mini. Cached to ~/.cache/huggingface (~1.6GB)"""
    print("\n[Download] Indic Parler-TTS mini (~1.6GB) ...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="ai4bharat/indic-parler-tts-mini")
    print("  Done: Indic Parler-TTS mini")


def download_piper(voice_name: str = "hi_IN-hindi_tdil-medium"):
    """
    Download Piper Hindi voice files (.onnx + .onnx.json).
    Saved to models/piper/ in the project directory.
    """
    import os
    import requests

    print(f"\n[Download] Piper voice: {voice_name} (~60MB) ...")
    os.makedirs("models/piper", exist_ok=True)
    # exist_ok=True means don't error if the folder already exists

    base_url = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main"
        "/hi/IN/hindi_tdil/medium"
    )

    for filename in [f"{voice_name}.onnx", f"{voice_name}.onnx.json"]:
        dest = f"models/piper/{filename}"
        if os.path.exists(dest):
            print(f"  Already exists: {filename} — skipping")
            continue

        print(f"  Downloading {filename} ...")
        response = requests.get(f"{base_url}/{filename}", stream=True)
        response.raise_for_status()
        # raise_for_status() raises an exception if HTTP error (4xx/5xx)

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                # stream=True + iter_content means we write in chunks
                # instead of loading the whole file into RAM first.
                f.write(chunk)

    print(f"  Done: Piper {voice_name}")


def main():
    download_all = "--all" in sys.argv
    # sys.argv is the list of command-line arguments.
    # sys.argv[0] is the script name, sys.argv[1:] are the flags.

    if download_all:
        # Download every model so you can freely switch in config.yaml
        print("Downloading ALL models...")
        download_whisper("medium")
        download_whisper("large-v3")
        download_coqui_xtts()
        download_indic_parler()
        download_piper()
    else:
        # Download only what config.yaml currently needs
        cfg = load_config()
        stt_cfg = cfg["stt"]
        tts_cfg = cfg["tts"]

        stt_provider = stt_cfg["provider"]
        tts_provider = tts_cfg["provider"]

        print(f"Downloading models for config: STT={stt_provider}, TTS={tts_provider}")

        if stt_provider == "whisper_faster":
            download_whisper(stt_cfg.get("model_size", "medium"))
        elif stt_provider == "whisper_cpp":
            # whisper.cpp (pywhispercpp) downloads its own models 
            # automatically on first run, so we can skip this.
            print(f"\n[Info] whisper_cpp: Model '{stt_cfg.get('model_size', 'medium')}' will download automatically on first use.")
        elif stt_provider == "vosk":
            print("\n[Download] Vosk: please manually download the Hindi model")
            print("  from https://alphacephei.com/vosk/models")
            print("  and place it in models/vosk/")

        if tts_provider == "coqui_xtts":
            download_coqui_xtts()
        elif tts_provider == "indic_parler":
            download_indic_parler()
        elif tts_provider == "piper":
            download_piper()

    print("\nAll downloads complete. You can now go fully offline.")
    print("Run the app with: uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()