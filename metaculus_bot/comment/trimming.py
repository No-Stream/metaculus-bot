"""Trim an assembled Metaculus comment, and its individual sections, to the platform's char limits.

Two entry points. ``trim_section`` shrinks one section of the framework's unified comment to its
own budget (``_section_budget`` maps the section name to it), and ``trim_comment`` shrinks the
whole assembled comment through an ordered strategy chain: shrink ``# RESEARCH`` alone
(``_trim_research_section_first``), else keep the summary head and the tail
(``_trim_preserving_summary_and_tail``), else fall back to a plain header-preserving trim
(``_trim_with_notice``).

Two invariants every path holds. The output keeps its leading ``#``, which the publish validator
requires. And the bytes ``performance_analysis`` later parses out of the published comment survive:
each rationale's ``Model: openrouter/...`` attribution line and its trailing fenced json forecast
block, the per-model ``*Forecaster N*:`` bullets in the summary, the trailing ``STACKED=<bool>``
marker, and the provider-diagnostics ``lost=`` token. That is why the over-budget paths trim from
*within* each rationale block rather than keeping a header plus one long tail: a naive
header-and-tail trim ate Forecaster 1's ``Model:`` line in 29 of 29 measured July 2026 trims.

The section budgets, that measurement, and the stacker-combined misattribution this file exists to
prevent: docs/architecture.md "Comment trimming".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final

from metaculus_bot.comment.markers import BASE_MODEL_SUBBLOCK_SPLIT_RE, STACKED_BASE_REASONING_HEADER
from metaculus_bot.constants import (
    COMMENT_CHAR_LIMIT,
    FORECASTS_SECTION_CHAR_LIMIT,
    RESEARCH_SECTION_CHAR_LIMIT,
    SUMMARY_SECTION_CHAR_LIMIT,
)
from metaculus_bot.prompts import SUMMARIZER_SOFT_FAIL_BANNER
from metaculus_bot.research.provider_diagnostics import PROVIDER_DIAGNOSTICS_HEADER

logger = logging.getLogger(__name__)

TRIM_NOTICE: Final[str] = "[... trimmed for length]"

# The h1 headers ForecastBot._create_comment emits; the section-aware trim splits on them.
_RESEARCH_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^# RESEARCH$", re.MULTILINE)
_FORECASTS_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^# FORECASTS$", re.MULTILINE)

# The framework emits this right after the per-model bullets the residual parsers read.
_SUMMARY_END_MARKER: Final[str] = "### Research Summary"

# Fits the summary with margin, leaving most of the budget to the tail; see docs/architecture.md.
_COMMENT_HEAD_BUDGET: Final[int] = 10_000


@dataclass(frozen=True, slots=True)
class TrimConfig:
    notice: str = TRIM_NOTICE
    # Fallback for unknown section names: the most permissive budget, so a bare trim never over-trims.
    section_limit: int = FORECASTS_SECTION_CHAR_LIMIT
    summary_limit: int = SUMMARY_SECTION_CHAR_LIMIT
    research_limit: int = RESEARCH_SECTION_CHAR_LIMIT
    forecasts_limit: int = FORECASTS_SECTION_CHAR_LIMIT
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
        # clean_indents output opens with a newline, which would leave an empty header and lose the "#".
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
            # Header alone overflows; truncate it but keep the leading "#".
            return text[:limit], True
        # Single-line blob: truncate from the front so the leading character survives.
        if limit > len(notice) + 1:
            return f"{text[: limit - len(notice) - 1]}\n{notice}", True
        return text[:limit], True

    tail_available = limit - len(notice) - 1
    if tail_available <= 0:
        return notice[:limit], True
    tail = text[-tail_available:]
    return f"{notice}\n{tail}", True


# The report number is always 1 in prod, but the pattern stays robust to multi-report comments.
_RATIONALE_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^##\s+R\d+:\s+Forecaster\s+\d+\s+Reasoning[ \t]*$")

# Byte-stable because performance_analysis/comment_sections.py keys on it; see docs/architecture.md.
_MODEL_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^Model:[ \t]*[^\n]*$")

# Ends every rationale and carries the values the residual pipeline parses, so trims keep it.
_JSON_BLOCK_RE: Final[re.Pattern[str]] = re.compile(r"```json\b.*?```", re.DOTALL)


def _pin_summarizer_banner(text: str, limit: int, notice: str) -> tuple[str, bool]:
    """Header, degradation banner and trimmed tail, for a research section carrying
    the summarizer soft-fail banner.

    The plain header-preserving trim keeps header plus TAIL, which drops the banner,
    and a soft-fail is itself what pushes a bundle over the research budget, so the
    disclosure was most likely dropped precisely on the questions where it fired.
    The banner is located by search rather than assumed to lead the body, because
    the orchestrator prepends it to the AskNews provider's own text. Returns
    ``(text, False)`` when the section carries no banner or the budget cannot seat
    header, banner and notice, so the caller falls back to the plain trim. Why the
    pin at all: docs/architecture.md "Comment trimming".
    """
    stripped = text.lstrip("\n")
    header, separator, remainder = stripped.partition("\n")
    if not separator:
        return text, False
    banner_idx = remainder.find(SUMMARIZER_SOFT_FAIL_BANNER)
    if banner_idx < 0:
        return text, False

    after_banner = remainder[banner_idx + len(SUMMARIZER_SOFT_FAIL_BANNER) :]
    # header + \n + banner + \n + notice + \n + tail -> 3 joining newlines.
    tail_available = limit - len(header) - len(SUMMARIZER_SOFT_FAIL_BANNER) - len(notice) - 3
    if tail_available <= 0:
        return text, False
    return f"{header}\n{SUMMARIZER_SOFT_FAIL_BANNER}\n{notice}\n{after_banner[-tail_available:]}", True


def _section_budget(section_name: str, cfg: TrimConfig) -> tuple[int, bool]:
    """Map a comment section name to its ``(char_limit, block_aware)`` policy.

    The framework's unified comment is assembled from three sections whose
    ``trim_section`` names end in ``_summary`` / ``_research`` / ``_rationales``
    (see forecaster.py and comment.formatting). Each gets its own budget so the
    parser-critical FORECASTS (rationales) section is not starved by a uniform
    cap. The rationales section is additionally trimmed block-by-block
    (``block_aware=True``) so per-forecaster ``Model:`` attribution and JSON
    forecast blocks survive an overflow. Unknown names fall back to the generic
    per-section default.
    """
    if section_name.endswith("_rationales"):
        return cfg.forecasts_limit, True
    if section_name.endswith("_research"):
        return cfg.research_limit, False
    if section_name.endswith("_summary"):
        return cfg.summary_limit, False
    return cfg.section_limit, False


def trim_section(text: str, section_name: str, *, config: TrimConfig | None = None) -> str:
    cfg = config or TrimConfig()
    limit, block_aware = _section_budget(section_name, cfg)
    if block_aware:
        trimmed, did_trim = _trim_rationales_within_blocks(text, limit, cfg.notice)
    else:
        # The summarizer-degradation banner is pinned ahead of the tail; every other shape falls through.
        trimmed, did_trim = _pin_summarizer_banner(text, limit, cfg.notice) if len(text) > limit else (text, False)
        if not did_trim:
            trimmed, did_trim = _trim_with_notice(text, limit, cfg.notice, preserve_header=True)
    if did_trim:
        logger.warning(
            "Trimmed section '%s' from %s to %s characters",
            section_name,
            len(text),
            len(trimmed),
        )
    return trimmed


def _allocate_block_budgets(sizes: list[int], total: int) -> list[int]:
    """Water-fill ``total`` chars across blocks by size.

    Blocks that fit within the even share keep their full size; the freed budget
    is redistributed to the larger blocks. This keeps small rationales whole
    while trimming only the ones that actually overflow, and always sums exactly
    to ``total`` (any rounding remainder lands on the first still-unfilled
    block).
    """
    budgets = [0] * len(sizes)
    remaining_total = total
    remaining_idx = list(range(len(sizes)))
    while remaining_idx:
        share = remaining_total // len(remaining_idx)
        fits = [i for i in remaining_idx if sizes[i] <= share]
        if not fits:
            for i in remaining_idx:
                budgets[i] = share
            budgets[remaining_idx[0]] += remaining_total - share * len(remaining_idx)
            break
        for i in fits:
            budgets[i] = sizes[i]
            remaining_total -= sizes[i]
            remaining_idx.remove(i)
    return budgets


def _trim_block(block: str, budget: int, notice: str) -> str:
    """Trim one rationale block to ``budget`` chars, preserving attribution.

    Dispatches on block shape. A **stacker-combined** block (one that folds the
    stacker meta-analysis and every base model's reasoning under
    ``STACKED_BASE_REASONING_HEADER`` — the single-R1-block shape stacking
    produces) is trimmed per-sub-block by ``_trim_stacker_combined_block`` so no
    base model loses its ``Model:`` attribution or forecast block. Every other
    block (the non-stacked prod case: one forecaster per R1 block) is trimmed as
    a single body by ``_trim_single_body``. The return value never exceeds
    ``budget``.
    """
    if len(block) <= budget:
        return block
    if STACKED_BASE_REASONING_HEADER in block:
        return _trim_stacker_combined_block(block, budget, notice)
    return _trim_single_body(block, budget, notice)


def _trim_single_body(block: str, budget: int, notice: str) -> str:
    """Trim one single-forecaster body to ``budget`` chars, preserving attribution.

    Keeps (in order) the leading header line (the ``## R1: Forecaster N
    Reasoning`` header for a full block, or the ``Model:`` line for a base
    sub-block), the ``Model:`` line if it opens the body, a head of the
    reasoning prose, the trim notice, and the trailing fenced ```json forecast
    block. So every kept body retains the two things the residual pipeline
    parses — its model attribution and its forecast values — even when the
    middle prose is sacrificed. The return value never exceeds ``budget``.
    """
    if len(block) <= budget:
        return block

    header, separator, rest = block.partition("\n")
    if not separator:
        return block[:budget]

    head = header
    body = rest
    lead = rest.lstrip("\n")
    model_match = _MODEL_PREFIX_RE.match(lead)
    if model_match:
        head = f"{header}\n{model_match.group(0)}"
        body = lead[model_match.end() :]

    json_matches = list(_JSON_BLOCK_RE.finditer(body))
    json_tail = json_matches[-1].group(0) if json_matches else ""
    prose_before_json = body[: json_matches[-1].start()] if json_matches else body

    # head + \n + prose + \n + notice + \n + json_tail  -> 3 joining newlines.
    prose_budget = budget - len(head) - len(notice) - len(json_tail) - 3
    if prose_budget > 0:
        prose_head = prose_before_json[:prose_budget].rstrip("\n")
        parts = [head, prose_head, notice, json_tail]
        return "\n".join(p for p in parts if p)
    if json_tail and len(head) + len(notice) + len(json_tail) + 2 <= budget:
        return f"{head}\n{notice}\n{json_tail}"
    if len(head) + len(notice) + 1 <= budget:
        return f"{head}\n{notice}"
    return head[:budget]


def _trim_stacker_combined_block(block: str, budget: int, notice: str) -> str:
    """Trim a stacker-combined R1 body, keeping every base model's attribution.

    ``combine_stacker_and_base_reasoning`` folds the stacker's meta-analysis and all
    N base reasonings into one ``## R1: Forecaster 1 Reasoning`` block, separated by
    ``STACKED_BASE_REASONING_HEADER``, each part ending with its own fenced json
    forecast block. So this splits on the same delimiter and ``Model:`` regex the
    parser uses, water-fills ``budget`` across the parts with
    ``_allocate_block_budgets``, trims each from within via ``_trim_single_body``,
    and re-emits the delimiter verbatim; the return value never exceeds ``budget``.
    The misattribution a single-body trim causes instead: docs/architecture.md
    "Comment trimming".
    """
    stacker_portion, base_portion = block.split(STACKED_BASE_REASONING_HEADER, 1)
    matches = list(BASE_MODEL_SUBBLOCK_SPLIT_RE.finditer(base_portion))
    if not matches:
        # Delimiter but no base sub-blocks: nothing to attribute per base model.
        return _trim_single_body(block, budget, notice)

    stacker_unit = stacker_portion.rstrip()
    base_units = [
        base_portion[m.start() : (matches[i + 1].start() if i + 1 < len(matches) else len(base_portion))].rstrip()
        for i, m in enumerate(matches)
    ]
    units = [stacker_unit, *base_units]

    # The delimiter, its two newlines and the blank lines between base sub-blocks sit outside the budgets.
    fixed = len(STACKED_BASE_REASONING_HEADER) + 2 + 2 * (len(base_units) - 1)
    usable = budget - fixed
    if usable <= 0:
        # Unreachable in prod, but keep the block coherent rather than emit a headerless fragment.
        return _trim_single_body(block, budget, notice)

    budgets = _allocate_block_budgets([len(u) for u in units], usable)
    stacker_trimmed = _trim_single_body(stacker_unit, budgets[0], notice)
    base_trimmed = [_trim_single_body(unit, budgets[i + 1], notice) for i, unit in enumerate(base_units)]
    return f"{stacker_trimmed}\n{STACKED_BASE_REASONING_HEADER}\n" + "\n\n".join(base_trimmed)


def _trim_rationales_within_blocks(text: str, limit: int, notice: str) -> tuple[str, bool]:
    """Trim the FORECASTS rationales section, block by block.

    Splits the section into its ``## R1: Forecaster N Reasoning`` blocks and
    shrinks each over-budget block from *within* (see ``_trim_block``), so every
    forecaster keeps its ``Model:`` attribution line and JSON forecast block. A
    plain header+tail trim (``_trim_with_notice``) drops the head of the first
    block — including its ``Model:`` line — which the residual pipeline parses.

    Falls back to the plain header-preserving trim when the section has no
    recognizable rationale headers (or the budget is too small to seat them).
    The returned text never exceeds ``limit``.
    """
    if len(text) <= limit:
        return text, False

    headers = list(_RATIONALE_HEADER_RE.finditer(text))
    if not headers:
        return _trim_with_notice(text, limit, notice, preserve_header=True)

    preamble = text[: headers[0].start()]
    blocks = [
        text[h.start() : (headers[i + 1].start() if i + 1 < len(headers) else len(text))].rstrip()
        for i, h in enumerate(headers)
    ]

    separator = "\n\n"
    usable = limit - len(preamble) - len(separator) * (len(blocks) - 1)
    if usable <= 0:
        return _trim_with_notice(text, limit, notice, preserve_header=True)

    budgets = _allocate_block_budgets([len(b) for b in blocks], usable)
    trimmed_blocks = [_trim_block(block, budgets[i], notice) for i, block in enumerate(blocks)]
    return preamble + separator.join(trimmed_blocks), True


def _trim_preserving_summary_and_tail(text: str, cfg: TrimConfig) -> tuple[str, bool]:
    """Trim the middle, keep the summary head and the tail.

    The head runs to ``cfg.summary_end_marker`` ("### Research Summary"), which the
    framework emits right after the per-model ``*Forecaster N*:`` bullets the
    residual parsers read; the tail carries the trailing ``STACKED`` marker and as
    many R1 rationales as fit. On overflow, up to ``cfg.head_budget`` chars go to
    the head, the remaining budget to the tail, and the middle becomes the trim
    notice. Returns ``(text, False)`` when the marker is absent, so the caller falls
    back to the plain tail-only trim.
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
        # Pathologically large head, or no room for tail plus notice: let the caller fall back.
        return text, False

    tail = text[-tail_budget:]
    trimmed = f"{head}\n{notice}\n{tail}"
    return trimmed, True


def _trim_research_section_first(text: str, cfg: TrimConfig) -> tuple[str, bool]:
    """Absorb the overflow by shrinking only the ``# RESEARCH`` section.

    Model reasoning outranks research, so an overflowing comment shrinks the research
    middle first and keeps ``# SUMMARY`` (bullets and the summary-end marker) and
    ``# FORECASTS`` (rationales and the trailing markers) whole. Research is
    summary-style and front-loaded, so the *front* of its body is what survives.

    Returns ``(text, False)`` when the comment lacks the ``# RESEARCH`` /
    ``# FORECASTS`` structure, or when shrinking research alone cannot cover the
    overflow; ``_trim_preserving_summary_and_tail`` handles those correctly and still
    preserves the leading ``#`` and the per-model bullets.
    """
    research_match = _RESEARCH_HEADER_RE.search(text)
    forecasts_match = _FORECASTS_HEADER_RE.search(text)
    if research_match is None or forecasts_match is None or research_match.start() >= forecasts_match.start():
        return text, False

    head = text[: research_match.start()]  # everything up to "# RESEARCH"
    research_header = research_match.group(0)  # "# RESEARCH" (from the regex)
    research_body = text[research_match.end() : forecasts_match.start()]
    tail = text[forecasts_match.start() :]  # "# FORECASTS" onward (incl. markers)

    # Appended, so a front-only keep would drop the lost= token naming which source degraded.
    diagnostics_idx = research_body.find(PROVIDER_DIAGNOSTICS_HEADER)
    pinned_diagnostics = "" if diagnostics_idx < 0 else research_body[diagnostics_idx:].rstrip("\n")

    notice = cfg.notice
    fixed = len(head) + len(research_header) + len(tail)

    # Three joining newlines; the last keeps "# FORECASTS" on its own line when research is cut.
    research_budget = cfg.comment_limit - fixed - len(notice) - 3
    if research_budget < 0:
        # Head plus tail already overflow, so shrinking research cannot help.
        return text, False

    # Drop the pin rather than the strategy: summary_and_tail would discard the whole research body.
    pin_reservation = len(pinned_diagnostics) + 1 if pinned_diagnostics else 0
    if pin_reservation >= research_budget:
        pinned_diagnostics = ""
        pin_reservation = 0
    front_budget = research_budget - pin_reservation

    kept_research = research_body[:front_budget].rstrip("\n") if front_budget > 0 else ""
    if pinned_diagnostics:
        kept_research = f"{kept_research}\n{pinned_diagnostics}" if kept_research else pinned_diagnostics
    trimmed = f"{head}{research_header}\n{notice}\n{kept_research}\n{tail}"
    return trimmed, True


def trim_comment(text: str, *, config: TrimConfig | None = None) -> str:
    cfg = config or TrimConfig()
    if len(text) <= cfg.comment_limit:
        return text

    # Every strategy in this chain preserves the leading "#", so the validator invariant holds.
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


__all__ = ["TRIM_NOTICE", "TrimConfig", "trim_comment", "trim_section"]
