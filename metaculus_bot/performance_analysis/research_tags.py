"""Research-archive treatment tags for performance records.

Reads ``backtests/research_archive/latest/<qid>.json`` and derives per-question
treatment flags — did the forecasters' research bundle carry a Time Series Anchor
section, an Agentic Research Findings (gap-fill v2) section — so calibration cuts
can split treated from untreated records without re-deriving the tags out-of-band
each round (the residual rounds' ``anchor_tags.json`` is the reference output).

Detection rules (source-class-aware, per the 2026-08-24 research-archive-qa dim):

* Header greps are DEPTH-AGNOSTIC (``^#{1,4}``): artifact records head sections at
  ``## ``, while comment-backfill records re-head everything one level deeper.
* The section flag means "the forecasters actually read this" and is the treatment
  marker. ``gfv2_loop_ran`` is deliberately separate: the v2 driver loop banks a
  transcript on the record even when it soft-fails and contributes NO section to
  the bundle, so payload-presence overstates treatment.
* A ``False`` anchor read is qualified by ``anchor_confidence``: the trim-immune
  ``## Provider Diagnostics`` block corroborates it when present
  (``diag_confirms_absent``), and a trimmed record with no diagnostics line stays
  ``ambiguous_trimmed_no_diag`` — trimming keeps header + tail, so a leading
  section can be eaten without that meaning the provider never fired.
* A question with NO archive record gets None on every tag, never False — absence
  of evidence is not an untreated record.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RESEARCH_ARCHIVE_LATEST = Path("backtests/research_archive/latest")

ANCHOR_SECTION_TITLE = "Time Series Anchor"
GFV2_SECTION_TITLE = "Agentic Research Findings"

# Depth-agnostic, exact-title header matches (see module docstring).
_ANCHOR_HEADER_RE = re.compile(r"^#{1,4}\s*" + re.escape(ANCHOR_SECTION_TITLE) + r"\s*$", re.MULTILINE)
_GFV2_HEADER_RE = re.compile(r"^#{1,4}\s*" + re.escape(GFV2_SECTION_TITLE) + r"\s*$", re.MULTILINE)

# One line of the Provider Diagnostics block, e.g.
# ``- timeseries_anchor: ok | 855 chars | 1171 ms``.
_DIAG_TS_ANCHOR_RE = re.compile(
    r"^-\s*timeseries_anchor:\s*(?P<status>ok|empty|errored|fallback|skipped|timeout)\s*\|\s*\d+\s*chars",
    re.MULTILINE,
)

_ABSENT_TAGS: dict[str, None] = {
    "anchor_present": None,
    "gfv2_present": None,
    "gfv2_loop_ran": None,
    "anchor_confidence": None,
    "research_source_class": None,
}


def _timeseries_anchor_diag_status(record: dict) -> str | None:
    """The ``timeseries_anchor:`` status from the Provider Diagnostics block, if any.

    Prefers the record's dedicated ``provider_diagnostics_block`` field; older
    records only carry the block inline in ``research_text``, so fall back to
    slicing the text from the block's title onward.
    """
    block = record.get("provider_diagnostics_block") or ""
    if not block:
        text = record.get("research_text") or ""
        idx = text.find("Provider Diagnostics")
        block = text[idx:] if idx >= 0 else ""
    match = _DIAG_TS_ANCHOR_RE.search(block)
    return match.group("status") if match else None


def research_tags_for_record(record: dict) -> dict:
    """Derive the treatment tags from one research-archive record."""
    text = record.get("research_text") or ""
    anchor_present = bool(_ANCHOR_HEADER_RE.search(text))
    diag_status = _timeseries_anchor_diag_status(record)

    if anchor_present:
        confidence = "header"
    elif diag_status in ("ok", "fallback"):
        # The provider ran and produced a section, but the text no longer carries
        # the header — e.g. trimming ate it. Treatment status is genuinely unclear.
        confidence = "diag_ok_header_missing"
    elif diag_status in ("empty", "errored", "skipped", "timeout"):
        confidence = "diag_confirms_absent"
    elif record.get("is_trimmed"):
        confidence = "ambiguous_trimmed_no_diag"
    else:
        confidence = "absent_no_diag"

    return {
        "anchor_present": anchor_present,
        "gfv2_present": bool(_GFV2_HEADER_RE.search(text)),
        "gfv2_loop_ran": bool(record.get("gap_fill_v2")),
        "anchor_confidence": confidence,
        "research_source_class": record.get("source"),
    }


def research_tags_for_qid(qid: object, latest_dir: Path | str = DEFAULT_RESEARCH_ARCHIVE_LATEST) -> dict:
    """Tags for one question id off ``latest/<qid>.json``; all-None when no record exists."""
    if qid is None:
        return dict(_ABSENT_TAGS)
    path = Path(latest_dir) / f"{qid}.json"
    if not path.exists():
        return dict(_ABSENT_TAGS)
    try:
        record = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Unreadable research-archive record {path}: {exc}")
        return dict(_ABSENT_TAGS)
    return research_tags_for_record(record)


def attach_research_tags(records: list[dict], latest_dir: Path | str = DEFAULT_RESEARCH_ARCHIVE_LATEST) -> None:
    """Stamp the treatment tags onto each performance record in place.

    Keyed on the record's ``question_id`` — ``latest/`` is question-id-keyed, and a
    record without an archive counterpart (pre-archive eras, expired artifacts,
    fresh clones without ``backtests/``) gets the all-None tags rather than False.
    """
    tagged = 0
    for record in records:
        tags = research_tags_for_qid(record.get("question_id"), latest_dir)
        record.update(tags)
        if tags["anchor_present"] is not None:
            tagged += 1
    logger.info(f"Research tags attached: {tagged}/{len(records)} records have an archive record")
