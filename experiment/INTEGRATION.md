# INTEGRATION.md
# ─────────────────────────────────────────────────────────────────────────────
# How to integrate the Hinglish MMS TTS provider into your existing project
# once you're happy with the experiment results.
# ─────────────────────────────────────────────────────────────────────────────

## What was built

```
providers/
  mms_hinglish_provider.py    ← Drop-in TTS provider (same interface as CoquiXTTS)

modules/
  hinglish_transliterator.py  ← IndicXlit wrapper (reusable standalone module)

experiment/
  run_experiment.py           ← Standalone test runner (no FastAPI needed)
  test_sentences.txt          ← Hinglish test inputs

download_mms_models.py        ← One-time model downloader (mirrors download_models.py)
```

---

## Step 1 — Run the experiment first

```bash
# Download models once (~216MB total)
python download_mms_models.py

# Quick smoke test (3 sentences)
python experiment/run_experiment.py

# Full test suite
python experiment/run_experiment.py --all

# With audio playback
python experiment/run_experiment.py --all --play

# Single sentence
python experiment/run_experiment.py --text "aaj mera meeting hai at 3 PM"
```

Check `experiment/outputs/` for the generated `.wav` files and `report.json`
for latency numbers.

---

## Step 2 — If results are good, integrate into your project

### 2a. Copy files

```bash
# From this experiment folder into your main project
cp providers/mms_hinglish_provider.py  <your_project>/providers/
cp modules/hinglish_transliterator.py  <your_project>/modules/
```

### 2b. Register in factory.py

In your `core/factory.py`, add the new provider to the provider map:

```python
# core/factory.py — add this import at the top
from providers.mms_hinglish_provider import MmsHinglishProvider

# Add to your provider registry dict (wherever you map names to classes)
TTS_PROVIDERS = {
    "coqui_xtts":   CoquiXTTSProvider,
    "parler":       ParlerTTSProvider,
    "mms_hinglish": MmsHinglishProvider,   # ← add this line
}
```

### 2c. Switch in config.yaml

```yaml
# config.yaml
tts:
  provider: mms_hinglish   # ← change from coqui_xtts or parler
```

That's it. No changes to `tts_router.py`, `main.py`, or `monitor/stats.py`.

---

## Step 3 — Optional: offline model path

To point the provider at your locally downloaded models (for full offline use):

```yaml
# config.yaml
tts:
  provider: mms_hinglish
  model_dir: models/tts/mms-tts-hin   # ← local path from download_mms_models.py
```

And in `factory.py`, pass `model_dir` when constructing the provider:

```python
config = load_config()  # however you read config.yaml
model_dir = config.get("tts", {}).get("model_dir", None)
provider = MmsHinglishProvider(model_dir=model_dir)
```

---

## Voice modulation

The provider supports speed and pitch adjustments that fit your project's
voice modulation feature. These can be exposed via your existing API:

```python
provider = MmsHinglishProvider()
provider.set_speed(1.2)    # 20% faster
provider.set_pitch(+2)     # 2 semitones higher (requires librosa)
```

Or add query params to your `/api/tts` endpoint in `tts_router.py`:

```python
@router.post("/api/tts")
async def tts(text: str, speed: float = 1.0, pitch: int = 0):
    _provider.set_speed(speed)
    _provider.set_pitch(pitch)
    waveform, sr = _provider.synthesize(text)
    ...
```

---

## Dependencies to add to requirements.txt

```txt
# MMS Hinglish TTS
transformers        # already in your requirements.txt
torch               # already in your requirements.txt
ai4bharat-transliteration   # new — IndicXlit
soundfile           # already in your requirements.txt
scipy               # already in your requirements.txt (for speed modulation)
librosa             # optional — only needed for pitch shifting
```

Only `ai4bharat-transliteration` is a new dependency.

---

## Latency expectations (CPU, no GPU)

| Component        | Typical latency    |
|------------------|--------------------|
| IndicXlit        | 5–30ms per sentence|
| MMS-TTS Hindi    | 200–800ms (varies with sentence length) |
| Total (short)    | ~300ms             |
| Total (long)     | ~1000ms            |

RTF (Real Time Factor) target: < 0.5 on modern CPU
(meaning synthesis is 2x faster than audio playback duration)

---

## Troubleshooting

**IndicXlit returns wrong transliteration for a word:**
Add it to the `ABBREVIATIONS` dict in `hinglish_transliterator.py`.

**MMS produces garbled audio:**
Usually means the Devanagari input has a character the model's vocabulary
doesn't cover. Check the `devanagari` field in `report.json` for each sentence.

**Very slow inference:**
MMS is a VITS model — inference is heavier than Piper but lighter than XTTS.
On CPU expect 300ms–1s per sentence. If you have a GPU, pass `device="cuda"`
in `mms_hinglish_provider.py` in the `load()` method.
