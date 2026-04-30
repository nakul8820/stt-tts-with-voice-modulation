# main.py
# ─────────────────────────────────────────────────────────────
# Entry point for the FastAPI application.
# This file:
#   1. Creates the FastAPI app
#   2. At startup: reads config, loads models via factory
#   3. Registers all routers
#   4. Serves the static frontend (index.html)
#
# To run:
#   uvicorn main:app --reload --port 8000
#
# Then open: http://localhost:8000
# ─────────────────────────────────────────────────────────────

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from core.factory import get_stt_provider, get_tts_provider
from routers import stt_router, tts_router
from monitor.stats import Stats  # Add this import

# ── Environment Setup ──
# Force Coqui TTS and other models to use the local 'models' folder
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["TTS_HOME"] = os.path.join(PROJECT_ROOT, "models", "tts")
os.environ["XDG_DATA_HOME"] = os.path.join(PROJECT_ROOT, "models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — replaces the deprecated
    @app.on_event("startup") pattern.

    Everything BEFORE 'yield' runs at startup.
    Everything AFTER 'yield' runs at shutdown.
    """
    # ── STARTUP ──
    print("Starting voice_app...")

    # Read config.yaml → get correct provider classes → load models
    stt = get_stt_provider()   # e.g. WhisperFasterProvider
    tts = get_tts_provider()   # e.g. CoquiXTTSProvider

    # Pass providers to routers so routes can use them.
    # init() calls provider.load() which loads weights into memory.
    stt_router.init(stt)
    tts_router.init(tts)

    print("All models loaded. App is ready.")
    print("Open http://localhost:8000 in your browser.")

    yield   # ← app runs here, handling requests

    # ── SHUTDOWN ──
    # Clean up resources if needed (model memory, file handles, etc.)
    print("Shutting down voice_app.")


# Create the FastAPI application
app = FastAPI(
    title="Hinglish Voice Transcriber",
    description="Local offline STT and TTS for Hindi/English/Hinglish",
    version="1.0.0",
    lifespan=lifespan   # hook in our startup/shutdown logic
)

# Allow the browser (served on port 8000) to call our API.
# Without this, browsers block cross-origin requests (CORS policy).
# Since frontend and backend are on the same origin here, this is
# mostly a safety net for development with --reload.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],     # allow GET, POST, etc.
    allow_headers=["*"],
)

# Register our routers — this adds all the /api/... routes
app.include_router(stt_router.router)
app.include_router(tts_router.router)

# ── Monitor Endpoints ──────────────────────────────────────────
# These are called by the performance dashboard in the frontend.

@app.get("/api/stats")
async def get_stats():
    """Returns averages of all recorded STT/TTS requests."""
    return Stats.summary()

@app.get("/api/system")
async def get_system_stats():
    """Returns live system CPU and RAM usage."""
    return Stats.get_system_stats()

# Serve the static folder (index.html) at the root URL.
# IMPORTANT: mount static LAST — after all API routes.
# If you mount it first, it will intercept /api/... requests.
app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static"
)