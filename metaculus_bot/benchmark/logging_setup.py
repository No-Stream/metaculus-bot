"""Logging utilities for benchmark CLI entry points."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


class FlushingStreamHandler(logging.StreamHandler):
    """Stream handler that flushes stdout and stderr after each record."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
        finally:
            try:
                sys.stderr.flush()
            except Exception:  # pragma: no cover - best effort
                pass


def configure_benchmark_logging(log_dir: str = "benchmarks") -> Path:
    """Configure console and file logging for benchmark runs.

    Returns the path to the log file for convenience.
    """

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = FlushingStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    forecasting_logger = logging.getLogger("forecasting_tools")
    forecasting_logger.setLevel(logging.INFO)
    forecasting_logger.propagate = True

    main_logger = logging.getLogger("__main__")
    main_logger.setLevel(logging.INFO)

    for noisy in ["LiteLLM", "httpx", "httpcore", "urllib3", "aiohttp"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.getLogger(noisy).propagate = False

    return log_path


__all__ = ["configure_benchmark_logging"]
