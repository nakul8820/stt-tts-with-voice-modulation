# monitor/stats.py
# ─────────────────────────────────────────────────────────────
# Tracks performance stats for every STT and TTS request:
# - Latency (how long each request took)
# - CPU usage during the request
# - RAM usage during the request
# - A rolling log of the last N requests
#
# Usage pattern: wrap any code block with the timer context manager
#
#   with Stats.track("stt"):
#       result = stt_provider.transcribe(audio)
#
# Then call Stats.summary() to get a dict of averages.
# ─────────────────────────────────────────────────────────────

import time
import psutil
# psutil = process and system utilities.
# Lets us read CPU%, RAM usage, and other system metrics.

from collections import deque
# deque = double-ended queue.
# We use it as a fixed-size rolling log — when it fills up,
# the oldest entry is automatically dropped from the left.

from contextlib import contextmanager
# contextmanager lets us write a generator function and use
# it as a 'with' statement. This is how we build the timer.

from dataclasses import dataclass, field
from typing import Literal
# Literal["stt", "tts"] means the type can only be one of
# those two string values — helps catch typos early.


@dataclass
class RequestStat:
    """
    A single recorded measurement for one request.
    dataclass auto-generates __init__, __repr__, etc.
    """
    operation: str      # "stt" or "tts"
    latency_ms: float   # how long it took in milliseconds
    cpu_percent: float  # CPU usage during the request (0-100)
    ram_mb: float       # RAM used by this process in MB
    text_length: int    # chars of input/output text (for context)
    timestamp: float    # Unix timestamp when it happened


class StatsTracker:
    """
    Singleton-style stats tracker.
    Keeps a rolling window of the last 100 requests.
    Thread-safe enough for our single-laptop use case.
    """

    def __init__(self, max_history: int = 100):
        # deque with maxlen automatically drops old entries
        # when new ones are added beyond max_history.
        self.history: deque[RequestStat] = deque(maxlen=max_history)
        self._process = psutil.Process()
        # psutil.Process() gives us metrics for THIS Python process
        # (not the whole system — just our app's usage)

    @contextmanager
    def track(self, operation: str, text_length: int = 0):
        """
        Context manager that measures latency + CPU + RAM.

        Usage:
            with stats.track("stt", text_length=len(audio_bytes)):
                result = provider.transcribe(audio_bytes)

        Everything inside the 'with' block is measured.
        """

        # ── BEFORE the request ──
        start_time = time.perf_counter()
        # perf_counter() is the highest-resolution timer available.
        # Much more precise than time.time() for short durations.

        # Sample CPU before. interval=None means non-blocking —
        # returns CPU% since the last call (or 0 on first call).
        cpu_before = self._process.cpu_percent(interval=None)

        # ── RUN the request ──
        yield
        # Everything between 'with stats.track():' and the end
        # of that with-block runs HERE at this yield point.

        # ── AFTER the request ──
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # Convert seconds → milliseconds (* 1000)

        cpu_after = self._process.cpu_percent(interval=None)
        cpu_used = max(cpu_before, cpu_after)
        # Take the peak CPU reading. max() because CPU can spike
        # during inference and we want to capture the peak.

        ram_mb = self._process.memory_info().rss / (1024 * 1024)
        # .rss = Resident Set Size = actual RAM held in memory
        # Divide by 1024*1024 to convert bytes → megabytes

        # Record this measurement
        stat = RequestStat(
            operation=operation,
            latency_ms=round(elapsed_ms, 2),
            cpu_percent=round(cpu_used, 1),
            ram_mb=round(ram_mb, 1),
            text_length=text_length,
            timestamp=time.time()
        )
        self.history.append(stat)

        # Print to console so you can watch in terminal while testing
        print(
            f"[Stats] {operation.upper()} | "
            f"{stat.latency_ms}ms | "
            f"CPU: {stat.cpu_percent}% | "
            f"RAM: {stat.ram_mb}MB"
        )

    def summary(self) -> dict:
        """
        Return average stats across all recorded history.
        Called by the GET /stats endpoint.
        """
        if not self.history:
            return {"message": "No requests recorded yet."}

        # Separate STT and TTS entries for individual averages
        stt_entries = [s for s in self.history if s.operation == "stt"]
        tts_entries = [s for s in self.history if s.operation == "tts"]

        def avg(entries, field):
            """Calculate average of a field across a list of entries."""
            if not entries:
                return 0
            return round(
                sum(getattr(e, field) for e in entries) / len(entries),
                2
            )
            # getattr(obj, "field_name") is like obj.field_name
            # but works when the field name is a variable string

        return {
            "total_requests": len(self.history),
            "stt": {
                "count": len(stt_entries),
                "avg_latency_ms": avg(stt_entries, "latency_ms"),
                "avg_cpu_percent": avg(stt_entries, "cpu_percent"),
                "avg_ram_mb": avg(stt_entries, "ram_mb"),
            },
            "tts": {
                "count": len(tts_entries),
                "avg_latency_ms": avg(tts_entries, "latency_ms"),
                "avg_cpu_percent": avg(tts_entries, "cpu_percent"),
                "avg_ram_mb": avg(tts_entries, "ram_mb"),
            },
            "recent": [
                # Last 10 requests as a list for the UI to display
                {
                    "op": s.operation,
                    "latency_ms": s.latency_ms,
                    "cpu_percent": s.cpu_percent,
                    "ram_mb": s.ram_mb,
                }
                for s in list(self.history)[-10:]
            ]
        }

    def get_system_stats(self) -> dict:
        """
        Return current live system stats (not per-request averages).
        Called by the UI to show a live dashboard.
        """
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            # interval=0.1 = measure CPU over 0.1 seconds (more accurate)

            "ram_used_mb": round(
                psutil.virtual_memory().used / (1024 * 1024), 1
            ),
            "ram_total_mb": round(
                psutil.virtual_memory().total / (1024 * 1024), 1
            ),
            "ram_percent": psutil.virtual_memory().percent,
            # Percentage of RAM currently in use (0-100)

            "process_ram_mb": round(
                self._process.memory_info().rss / (1024 * 1024), 1
            ),
            # RAM used specifically by our voice_app process
        }


# Create one global instance — imported by routers and main.py
# This is the "singleton" pattern: one shared tracker for the whole app.
Stats = StatsTracker()