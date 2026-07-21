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
    # Generic per-section fallback (unknown section names). Set to the most
    # permissive section budget so a bare trim_section never over-trims.
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


# A single forecaster's rationale header inside the FORECASTS section, e.g.
# "## R1: Forecaster 3 Reasoning". report_number is always 1 in production
# (single report), but \d+ keeps this robust to multi-report comments. The
# block-aware trim splits the section on these so each forecaster keeps its own
# attribution when the section overflows.
_RATIONALE_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^##\s+R\d+:\s+Forecaster\s+\d+\s+Reasoning[ \t]*$")

# The bot-injected "Model: openrouter/<provider>/<name>" attribution line that
# opens each rationale body. Kept byte-stable because
# performance_analysis/parsing.py (_R1_MODEL_RE, _REASONING_MODEL_PREFIX_RE)
# keys per-model attribution on it — the exact line a naive header+tail trim
# destroyed (measured: Forecaster 1's Model: line eaten in 29/29 July trims).
_MODEL_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^Model:[ \t]*[^\n]*$")

# A fenced ```json STRUCTURED FORECAST block. Each rationale ends with one (the
# block-last prompt requirement); it carries the per-model forecast values the
# residual pipeline parses, so a within-block trim keeps it in the kept tail.
_JSON_BLOCK_RE: Final[re.Pattern[str]] = re.compile(r"```json\b.*?```", re.DOTALL)


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

    When stacking fires, ``combine_stacker_and_base_reasoning`` folds the
    stacker's meta-analysis and all N base reasonings into a single
    ``## R1: Forecaster 1 Reasoning`` block: a stacker portion (which itself
    ends with the stacker's own json forecast block), the
    ``STACKED_BASE_REASONING_HEADER`` delimiter, then one
    ``Model: openrouter/...`` sub-block per base model, each ending with its own
    fenced json forecast block.

    Trimming this as a single body keeps only the LAST json block in the whole
    combined body and orphans it from its ``Model:`` line — so
    ``performance_analysis.parsing._split_stacker_combined_body`` re-attributes
    that trailing model's forecast values to the last SURVIVING base model
    (silent misattribution, not just loss). Instead we split on the same
    delimiter + ``Model:`` regex the parser uses, water-fill ``budget`` across
    the stacker portion and each base sub-block with ``_allocate_block_budgets``,
    and trim each from within via ``_trim_single_body`` — so every base model
    keeps its ``Model:`` line paired with its own json block, sacrificing only
    per-sub-block prose. The delimiter is re-emitted verbatim so the parser's
    stacker-body detection still fires. The return value never exceeds
    ``budget``.
    """
    stacker_portion, base_portion = block.split(STACKED_BASE_REASONING_HEADER, 1)
    matches = list(BASE_MODEL_SUBBLOCK_SPLIT_RE.finditer(base_portion))
    if not matches:
        # Delimiter present but no base ``Model:`` sub-blocks (e.g. a body
        # already truncated inside the stacker portion). Nothing to attribute
        # per base model, so fall back to the single-body trim — it still keeps
        # the R1 header and a trailing json block.
        return _trim_single_body(block, budget, notice)

    stacker_unit = stacker_portion.rstrip()
    base_units = [
        base_portion[m.start() : (matches[i + 1].start() if i + 1 < len(matches) else len(base_portion))].rstrip()
        for i, m in enumerate(matches)
    ]
    units = [stacker_unit, *base_units]

    # Reassembly:
    #   {stacker_unit}\n{DELIMITER}\n{base_0}\n\n{base_1}\n\n...
    # Fixed overhead the per-unit budgets can't touch: the delimiter, the
    # newline before it, the newline after it, and the blank line between
    # consecutive base sub-blocks.
    fixed = len(STACKED_BASE_REASONING_HEADER) + 2 + 2 * (len(base_units) - 1)
    usable = budget - fixed
    if usable <= 0:
        # Budget too small to seat the delimiter + units. Does not occur in prod
        # (the single stacked R1 block gets the whole FORECASTS section budget),
        # but keep the block coherent rather than emit a headerless fragment.
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
