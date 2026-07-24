"""Run-log telemetry harvest: parse structured markers from GHA run_logs artifacts.

The bot's prod runs tee stdout+stderr to ``run_logs/`` artifacts (since 2026-07-12),
carrying greppable telemetry markers (``EXTRACTION_RUNG``, ``GAP_FILL_V2``,
``GHOST_PRE[_JSON]``, ``GHOST_FORECAST[_JSON]``, ``OPEN_BOUND_PILING``,
``CREDIT_*``). GHA artifacts expire at
90 days, so this package harvests them into a durable local archive under
``backtests/telemetry_archive/`` — the same silent-loss motivation behind the
research-archive sync (see ``scripts/download_research.py``).
"""
