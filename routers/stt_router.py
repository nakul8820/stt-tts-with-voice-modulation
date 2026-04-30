# FastAPI route for speech-to-text
# exposes post / transcribe - accepts audio file, returns text

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from core.base_stt import BaseSTTProvider
from monitor.stats import Stats

router = APIRouter(prefix="/api", tags=["STT"])

# Global reference to whichever STT provider is loaded
# set by init() called from main.py at startup.
_provider: BaseSTTProvider = None

def init(provider: BaseSTTProvider) -> None:
    """
    Called once at start up with the configured STT provider.
    Loads the model into memory and stores it for route use.
    """
    global _provider
    _provider = provider
    _provider.load()
    print("[STTRouter] Provider ready")

@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    POST /api/transcribe

    Accepts: multipart/form-data with an 'audio' file field.
    Returns: JSON with transcribed text and detected language.

    Example response:
    {
        "text": "Mera naam Rahul hai and I am from Mumbai",
        "language": "hi",
        "latency_ms": 3240.5
    }
    """
    if _provider is None:
        # This shouldn't happen if startup ran correctly,
        # but it's good practice to guard against it.
        raise HTTPException(
            status_code=500,
            detail="STT provider not initialised. Check server startup logs."
        )

    # Read the uploaded file into memory as raw bytes
    audio_bytes = await audio.read()
    # 'await' is needed because UploadFile.read() is async —
    # FastAPI uses async I/O to handle files without blocking.

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Run transcription inside the stats tracker.
    # The 'with' block measures latency + CPU + RAM automatically.
    import time
    start = time.perf_counter()

    with Stats.track("stt", text_length=len(audio_bytes)):
        result = _provider.transcribe(audio_bytes)
        # result = {"text": "...", "language": "hi"}

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return JSONResponse({
        "text": result["text"],
        "language": result.get("language", "hi"),
        "latency_ms": latency_ms
    })