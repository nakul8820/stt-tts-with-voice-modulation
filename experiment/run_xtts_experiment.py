# experiment/run_xtts_experiment.py
import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xtts_experiment")

def save_wav(audio_bytes, filepath):
    with open(filepath, "wb") as f:
        f.write(audio_bytes)

def run_experiment(sentences, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from providers.tts.coqui_xtts import CoquiXTTSProvider
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    tts_cfg = config["tts"]
    logger.info(f"Initializing CoquiXTTSProvider with config: {tts_cfg}")
    
    # Environment variables for offline use
    os.environ["TTS_HOME"] = os.path.join(PROJECT_ROOT, "models", "tts")
    os.environ["XDG_DATA_HOME"] = os.path.join(PROJECT_ROOT, "models")

    provider = CoquiXTTSProvider(tts_cfg)
    provider.load()

    results = []
    for i, sentence in enumerate(sentences):
        logger.info(f"[{i+1}/{len(sentences)}] Input: {sentence}")
        
        start_time = time.time()
        try:
            audio_bytes = provider.synthesize(sentence)
            latency = (time.time() - start_time) * 1000
            
            out_path = output_dir / f"{i+1:02d}_xtts.wav"
            save_wav(audio_bytes, out_path)
            
            results.append({
                "index": i + 1,
                "input": sentence,
                "latency_ms": latency,
                "output": str(out_path)
            })
            logger.info(f"  Latency: {latency:.2f}ms")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Save report
    with open(output_dir / "xtts_report.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Experiment complete. Report saved to {output_dir / 'xtts_report.json'}")

if __name__ == "__main__":
    test_sentences = [
        "aaj mera meeting hai at 3 PM",
        "kal office nahi jaana mujhe",
        "please apna email check karo",
        "mera phone charge nahi ho raha"
    ]
    run_experiment(test_sentences, Path("experiment/outputs_xtts"))
