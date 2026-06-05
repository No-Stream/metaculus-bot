from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final

from metaculus_bot.constants import COMMENT_CHAR_LIMIT, REPORT_SECTION_CHAR_LIMIT

logger = logging.getLogger(__name__)

TRIM_NOTICE: Final[str] = "[... trimmed for length]"

# Top-level h1 section headers in the framework's unified comment
# (forecast_bot.py:538-550). The section-aware trim splits on these; see
# ``_trim_research_section_first`` for the priority rationale.
_RESEARCH_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^# RESEARCH$", re.MULTILINE)
_FORECASTS_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^# FORECASTS$", re.MULTILINE)

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


def _trim_with_notice(text: str, limit: int, notice: str, *, preserve_header: bool) -> tuple[str, bool]:
    if limit <= 0:
        return "", bool(text)
    if len(text) <= limit:
        return text, False
    if limit <= len(notice):
        return notice[:limit], True

    if preserve_header:
        # Strip leading newlines so the header is the first non-empty line.
        # The framework's clean_indents output starts with "\n# SUMMARY"; without
        # this, partition("\n") would yield an empty header and the output would
        # lose its leading "#", breaking the validator invariant.
        text = text.lstrip("\n")
        header, separator, remainder = text.partition("\n")
        if separator:
            available = limit - len(header) - len(notice) - 2
            if available > 0:
                tail = remainder[-available:]
                return f"{header}\n{notice}\n{tail}", True
            truncated_header = header[: max(0, limit - len(notice) - 1)]
            if truncated_header:
                return f"{truncated_header}\n{notice}", True
            # Header alone exceeds the budget. Truncate it but keep the leading
            # character so a '#'-leading input stays '#'-leading.
            return text[:limit], True
        # Single-line blob (no newline). Truncate from the front, appending the
        # notice when there's room, so the leading character is preserved.
        if limit > len(notice) + 1:
            return f"{text[: limit - len(notice) - 1]}\n{notice}", True
        return text[:limit], True

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


def _trim_preserving_summary_and_tail(text: str, cfg: TrimConfig) -> tuple[str, bool]:
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


def _trim_research_section_first(text: str, cfg: TrimConfig) -> tuple[str, bool]:
    """Absorb the overflow by shrinking only the ``# RESEARCH`` section.

    Priority is model reasoning > research, so when the comment overflows we
    deterministically shrink the research middle before touching anything else.
    The comment is split on its top-level headers into::

        # SUMMARY ...        <- head: bullets + ### Research Summary marker
        # RESEARCH ...        <- middle: summary-style, front-loaded research
        # FORECASTS ...       <- tail: rationales + trailing STACKED/TOOLS markers

    We keep SUMMARY and FORECASTS whole and shrink RESEARCH to fit. Research is
    summary-style and front-loaded, so we keep the *front* of its body (header +
    notice + as much head as the budget allows).

    Returns ``(text, False)`` (caller falls back to ``summary_and_tail``) when
    the comment lacks the recognizable ``# RESEARCH`` / ``# FORECASTS``
    structure, OR when shrinking research alone can't cover the overflow (a
    pathologically large SUMMARY head or FORECASTS tail). In that case the
    summary-and-tail path — which anchors on ``### Research Summary`` and drops
    everything between it and the tail — handles it correctly while still
    preserving the leading ``#`` and the per-model bullets.
    """
    research_match = _RESEARCH_HEADER_RE.search(text)
    forecasts_match = _FORECASTS_HEADER_RE.search(text)
    if research_match is None or forecasts_match is None or research_match.start() >= forecasts_match.start():
        return text, False

    head = text[: research_match.start()]  # everything up to "# RESEARCH"
    research_header = research_match.group(0)  # "# RESEARCH" (from the regex)
    research_body = text[research_match.end() : forecasts_match.start()]
    tail = text[forecasts_match.start() :]  # "# FORECASTS" onward (incl. markers)

    notice = cfg.notice
    fixed = len(head) + len(research_header) + len(tail)

    # Budget left for the research body after the fixed head/header/tail. The 3
    # joining newlines: after the header, after the notice, and before the tail
    # (so "# FORECASTS" stays on its own line even when research is truncated).
    # When this is < 0, head + tail alone overflow, so shrinking research can't
    # help — defer to the summary-and-tail path.
    research_budget = cfg.comment_limit - fixed - len(notice) - 3
    if research_budget < 0:
        return text, False

    kept_research = research_body[:research_budget].rstrip("\n") if research_budget > 0 else ""
    trimmed = f"{head}{research_header}\n{notice}\n{kept_research}\n{tail}"
    return trimmed, True


def trim_comment(text: str, *, config: TrimConfig | None = None) -> str:
    cfg = config or TrimConfig()
    if len(text) <= cfg.comment_limit:
        return text

    # Ordered fallback chain. Each strategy preserves the leading "#", so the
    # validator invariant holds regardless of which one fires. The last resort
    # (plain header-preserving trim) handles comments with no recognizable
    # section structure at all.
    trimmed, used = _trim_research_section_first(text, cfg)
    if used:
        strategy = "research_first"
    else:
        trimmed, used = _trim_preserving_summary_and_tail(text, cfg)
        if used:
            strategy = "summary_and_tail"
        else:
            trimmed, _ = _trim_with_notice(text, cfg.comment_limit, cfg.notice, preserve_header=True)
            strategy = "header_and_tail"

    logger.warning(
        "Trimmed Metaculus comment from %s to %s characters (strategy=%s)",
        len(text),
        len(trimmed),
        strategy,
    )
    return trimmed


__all__ = ["TrimConfig", "trim_comment", "trim_section", "TRIM_NOTICE"]
