"""Unified logger for the entire project — writes to both stdout and a log file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with the given name (configured only once)."""
    global _CONFIGURED

    # Deferred import to avoid circular imports and let config control the log level
    from config import settings

    if not _CONFIGURED:
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = settings.LOGS_DIR / "robot.log"

        root = logging.getLogger("target_track_robot")
        root.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
        root.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            # No write permission (e.g. in a test environment) — continue with console only
            pass

        _CONFIGURED = True

    return logging.getLogger(f"target_track_robot.{name}")
