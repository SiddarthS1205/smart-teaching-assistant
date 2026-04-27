"""
Observability module — structured logging, request timing, and query tracking.
"""
import logging
import time
import json
import os
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

from config import settings

# ── Ensure log directory exists ──────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

# ── Root logger setup ─────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", encoding="utf-8"),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


# ── Query tracker (in-memory, swap for a DB in production) ───────────────────
_query_log: list[dict] = []


def track_query(query: str, response: str, agent: str, duration_ms: float) -> None:
    """Persist a query/response pair for observability dashboards."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "query": query,
        "response_preview": response[:200],
        "duration_ms": round(duration_ms, 2),
    }
    _query_log.append(entry)

    # Also write to a JSONL file for offline analysis
    with open("logs/queries.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_query_log() -> list[dict]:
    """Return the in-memory query log."""
    return _query_log


# ── Timing decorator ──────────────────────────────────────────────────────────
def timed(logger: logging.Logger | None = None):
    """Decorator that logs execution time of any function."""
    def decorator(fn: Callable) -> Callable:
        _log = logger or get_logger(fn.__module__)

        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                _log.debug("⏱  %s completed in %.2f ms", fn.__name__, elapsed)
                return result
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                _log.error("❌ %s failed after %.2f ms — %s", fn.__name__, elapsed, exc)
                raise

        return wrapper
    return decorator
