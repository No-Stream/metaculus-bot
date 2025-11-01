from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Tuple

from metaculus_bot.constants import COMMENT_CHAR_LIMIT, REPORT_SECTION_CHAR_LIMIT

logger = logging.getLogger(__name__)

TRIM_NOTICE: Final[str] = "[... trimmed for length]"


@dataclass(frozen=True, slots=True)
class TrimConfig:
    notice: str = TRIM_NOTICE
    section_limit: int = REPORT_SECTION_CHAR_LIMIT
    comment_limit: int = COMMENT_CHAR_LIMIT


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


def trim_comment(text: str, *, config: TrimConfig | None = None) -> str:
    cfg = config or TrimConfig()
    trimmed, did_trim = _trim_with_notice(text, cfg.comment_limit, cfg.notice, preserve_header=False)
    if did_trim:
        logger.warning(
            "Trimmed Metaculus comment from %s to %s characters",
            len(text),
            len(trimmed),
        )
    return trimmed


__all__ = ["TrimConfig", "trim_comment", "trim_section", "TRIM_NOTICE"]
