# routers/tts_router.py
# ─────────────────────────────────────────────────────────────
# CHANGES FROM ORIGINAL:
#   1. synthesize() — ONE new line: calls modulation.process()
#      after getting audio bytes from the provider
#   2. NEW: PUT /api/admin/voice-config — save admin settings
#   3. NEW: GET /api/admin/voice-config — load current settings
#   4. NEW: POST /api/admin/preview — preview with custom sentence
# Everything else identical to original.
# ─────────────────────────────────────────────────────────────

import time
import yaml
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import Optional
import secrets

from core.base_tts import BaseTTSProvider
from monitor.stats import Stats
from modules.modulation import process as apply_modulation
# ↑ This single import is the only new dependency in this file.
# process() takes wav_bytes in, returns modulated wav_bytes out.

router = APIRouter(prefix="/api", tags=["TTS"])
security = HTTPBasic()
# HTTPBasic = simple username/password auth sent in request headers.
# Good enough for a local laptop tool — no tokens needed.

_provider: BaseTTSProvider = None


def init(provider: BaseTTSProvider) -> None:
    global _provider
    _provider = provider
    _provider.load()
    print("[TTSRouter] Provider ready.")


# ── Auth helper ───────────────────────────────────────────────
def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Dependency function that checks admin credentials.
    Used by all admin-only endpoints via Depends(verify_admin).

    Depends() means FastAPI calls this function automatically
    before the route handler runs — it's like middleware per-route.
    """
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    admin_cfg = cfg.get("admin", {})

    # secrets.compare_digest prevents timing attacks —
    # always takes the same time regardless of where strings differ.
    username_ok = secrets.compare_digest(
        credentials.username,
        admin_cfg.get("username", "admin")
    )
    password_ok = secrets.compare_digest(
        credentials.password,
        admin_cfg.get("password", "changeme123")
    )

    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"}
        )
    return credentials.username


# ── Request/Response models ───────────────────────────────────
class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "default"


class VoiceConfigRequest(BaseModel):
    """
    Shape of the JSON the admin panel POSTs when saving settings.
    All fields optional — only send what changed.
    """
    active_preset:    Optional[str]   = None
    expressiveness:   Optional[int]   = None
    speed_model:      Optional[float] = None
    pitch_semitones:  Optional[int]   = None
    speed_post:       Optional[float] = None
    volume_gain:      Optional[float] = None
    bass:             Optional[int]   = None
    treble:           Optional[int]   = None
    reverb:           Optional[bool]  = None
    denoise:          Optional[bool]  = None
    normalize:        Optional[bool]  = None


class PreviewRequest(BaseModel):
    """Shape of the JSON for the preview endpoint."""
    text: str
    # voice config is read from config.yaml (whatever admin set)
    # so no config fields here — preview always uses current settings


# ── User endpoint ─────────────────────────────────────────────
@router.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """
    POST /api/synthesize — public, used by regular users.
    Synthesize text → audio, apply modulation, return WAV bytes.
    """
    if _provider is None:
        raise HTTPException(status_code=500, detail="TTS provider not initialised.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    start = time.perf_counter()

    with Stats.track("tts", text_length=len(req.text)):

        # Step 1: Generate raw audio from the TTS model
        audio_bytes = _provider.synthesize(req.text, req.voice_id)

        # ── THE ONE NEW LINE ───────────────────────────────────
        # Step 2: Apply Layer 2 post-processing modulation.
        # apply_modulation() reads config.yaml internally so it
        # always uses the latest admin settings.
        audio_bytes = apply_modulation(audio_bytes)
        # ──────────────────────────────────────────────────────

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    print(f"[TTS] {len(req.text)} chars in {latency_ms}ms")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"X-Latency-Ms": str(latency_ms)}
    )


# ── Admin endpoints ───────────────────────────────────────────
@router.get("/admin/voice-config")
async def get_voice_config(username: str = Depends(verify_admin)):
    """
    GET /api/admin/voice-config
    Returns the current modulation settings + all presets.
    Called when admin panel loads to populate the sliders.
    Protected: requires admin credentials.
    """
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    return JSONResponse({
        "modulation": cfg.get("modulation", {}),
        "presets":    cfg.get("presets", {})
    })


@router.put("/admin/voice-config")
async def save_voice_config(
    req: VoiceConfigRequest,
    username: str = Depends(verify_admin)
):
    """
    PUT /api/admin/voice-config
    Save admin's slider values to config.yaml.
    Only updates the fields that were sent — others stay unchanged.
    Protected: requires admin credentials.
    """
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    mod = cfg.setdefault("modulation", {})
    # setdefault(key, default) returns existing value or sets it
    # to default if missing — safe way to ensure the key exists

    # Update only the fields that were actually sent
    # (req.field is None if not sent — we skip those)
    if req.active_preset   is not None: mod["active_preset"]   = req.active_preset
    if req.expressiveness  is not None: mod["expressiveness"]  = req.expressiveness
    if req.speed_model     is not None: mod["speed_model"]     = req.speed_model
    if req.pitch_semitones is not None: mod["pitch_semitones"] = req.pitch_semitones
    if req.speed_post      is not None: mod["speed_post"]      = req.speed_post
    if req.volume_gain     is not None: mod["volume_gain"]     = req.volume_gain
    if req.bass            is not None: mod["bass"]            = req.bass
    if req.treble          is not None: mod["treble"]          = req.treble
    if req.reverb          is not None: mod["reverb"]          = req.reverb
    if req.denoise         is not None: mod["denoise"]         = req.denoise
    if req.normalize       is not None: mod["normalize"]       = req.normalize

    # If saving as custom preset, also update the custom preset values
    if req.active_preset == "custom":
        custom = cfg.setdefault("presets", {}).setdefault("custom", {})
        for field in ["expressiveness","speed_model","pitch_semitones",
                      "speed_post","volume_gain","bass","treble",
                      "reverb","denoise","normalize"]:
            val = getattr(req, field)
            if val is not None:
                custom[field] = val

    # Write back to config.yaml
    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        # allow_unicode=True preserves Hindi characters if any

    return JSONResponse({"status": "saved"})


@router.post("/admin/preview")
async def preview_voice(
    req: PreviewRequest,
    username: str = Depends(verify_admin)
):
    """
    POST /api/admin/preview
    Admin types a sentence → hears it with current modulation settings.
    Uses the same synthesize + modulate pipeline as /synthesize
    so what admin hears is exactly what users will hear.
    Protected: requires admin credentials.
    """
    if _provider is None:
        raise HTTPException(status_code=500, detail="TTS provider not initialised.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Preview text cannot be empty.")

    # Reuse the exact same pipeline as the user endpoint
    audio_bytes = _provider.synthesize(req.text, "default")
    audio_bytes = apply_modulation(audio_bytes)

    return Response(
        content=audio_bytes,
        media_type="audio/wav"
    )