from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Tuple

from metaculus_bot.constants import COMMENT_CHAR_LIMIT, REPORT_SECTION_CHAR_LIMIT

logger = logging.getLogger(__name__)

TRIM_NOTICE: Final[str] = "[... trimmed for length]"

# Marker that ends the summary section of a bot comment. When trimming the
# whole comment, we want to preserve everything up to and including the
# summary so the residual-analysis parsers (which match *Forecaster N*: value
# bullets in the summary) keep working. This marker is chosen because the
# framework consistently emits "### Research Summary" immediately after the
# summary bullets and before the first R1 rationale.
_SUMMARY_END_MARKER: Final[str] = "### Research Summary"

# Max chars reserved for the summary head when the comment must be trimmed.
# Sized so it fits the summary + a safety margin while only nibbling a small
# fraction of the total budget; the rest goes to the tail (which holds the
# STACKED=<bool> marker and as many R1 rationales as fit).
_COMMENT_HEAD_BUDGET: Final[int] = 10_000


@dataclass(frozen=True, slots=True)
class TrimConfig:
    notice: str = TRIM_NOTICE
    section_limit: int = REPORT_SECTION_CHAR_LIMIT
    comment_limit: int = COMMENT_CHAR_LIMIT
    summary_end_marker: str = _SUMMARY_END_MARKER
    head_budget: int = _COMMENT_HEAD_BUDGET


def _trim_with_notice(text: str, limit: int, notice: str, *, preserve_header: bool) -> Tuple[str, bool]:
    if limit <= 0:
        return "", bool(text)
    if len(text) <= limit:
        return text, False
    if limit <= len(notice):
        return notice[:limit], True

    if preserve_header:
        header, separator, remainder = text.partition("\n")
        if separator:
            available = limit - len(header) - len(notice) - 2
            if available > 0:
                tail = remainder[-available:]
                return f"{header}\n{notice}\n{tail}", True
            truncated_header = header[: max(0, limit - len(notice) - 1)]
            if truncated_header:
                return f"{truncated_header}\n{notice}", True
            return notice[:limit], True

    tail_available = limit - len(notice) - 1
    if tail_available <= 0:
        return notice[:limit], True
    tail = text[-tail_available:]
    return f"{notice}\n{tail}", True


def trim_section(text: str, section_name: str, *, config: TrimConfig | None = None) -> str:
    cfg = config or TrimConfig()
    trimmed, did_trim = _trim_with_notice(text, cfg.section_limit, cfg.notice, preserve_header=True)
    if did_trim:
        logger.warning(
            "Trimmed section '%s' from %s to %s characters",
            section_name,
            len(text),
            len(trimmed),
        )
    return trimmed


def _trim_preserving_summary_and_tail(text: str, cfg: TrimConfig) -> Tuple[str, bool]:
    """Trim the middle, keep the summary head and the tail.

    The bot's published comment has the structure:

        # SUMMARY
        ...
        ## Report 1 Summary
        ### Forecasts
        *Forecaster 1*: ...          <- residual-analysis parsers read these
        *Forecaster 2*: ...
        ### Research Summary         <- ``cfg.summary_end_marker``
        ...
        ## R1: Forecaster 1 Reasoning
        ...
        <!-- STACKED=true -->        <- residual-analysis marker

    If the comment overflows ``cfg.comment_limit``, we carve up to
    ``cfg.head_budget`` chars for the head (everything up to and including
    the summary-end marker) and use the remaining budget for the tail
    (which preserves the STACKED marker and as many R1 rationales as fit).
    The middle gets replaced with the trim notice.

    Falls back to (text, False) if the marker isn't present — caller should
    then use the plain tail-only trim.
    """
    marker_idx = text.find(cfg.summary_end_marker)
    if marker_idx < 0:
        return text, False

    head_end = marker_idx + len(cfg.summary_end_marker)
    head = text[:head_end]
    notice = cfg.notice

    # Reserve head, newline, notice, newline before giving the rest to tail.
    tail_budget = cfg.comment_limit - len(head) - len(notice) - 2
    if len(head) > cfg.head_budget or tail_budget <= 0:
        # Head is pathologically large, or no room left for tail + notice.
        # Let caller fall back to plain tail-only trim.
        return text, False

    tail = text[-tail_budget:]
    trimmed = f"{head}\n{notice}\n{tail}"
    return trimmed, True


def trim_comment(text: str, *, config: TrimConfig | None = None) -> str:
    cfg = config or TrimConfig()
    if len(text) <= cfg.comment_limit:
        return text

    # Try summary-and-tail preservation first; it keeps the *Forecaster N*
    # bullets intact for residual-analysis parsers. If the comment lacks the
    # expected structure (or is degenerate), fall back to tail-only trimming.
    trimmed, used_structured = _trim_preserving_summary_and_tail(text, cfg)
    if not used_structured:
        trimmed, _ = _trim_with_notice(text, cfg.comment_limit, cfg.notice, preserve_header=False)

    logger.warning(
        "Trimmed Metaculus comment from %s to %s characters (structured=%s)",
        len(text),
        len(trimmed),
        used_structured,
    )
    return trimmed


__all__ = ["TrimConfig", "trim_comment", "trim_section", "TRIM_NOTICE"]
