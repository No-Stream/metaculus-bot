"""Derive the checked-in miniature comment fixture from a local performance pull.

Why this exists
---------------
``tests/test_performance_analysis_parsing.py::TestRealDataRegression`` guards
per-model attribution recovery — the mechanism the whole residual/calibration
workflow depends on. It used to read only ``scratch/performance_data.json``,
which is gitignored, untracked, and rewritten by every ``spring-aib-2026`` pull.
So the class never ran in CI and its local results were not reproducible
run-to-run.

This script distills that big local pull into
``tests/data/performance_comments_mini.jsonl``: one record per distinct comment
SHAPE, small enough to check in, deterministic, and safe to publish. The test
class reads the miniature as its CI floor and additionally sweeps the big local
file when present.

Faithfulness invariant
----------------------
A record is only included when every parser in ``parser_outputs`` returns
IDENTICAL output on the shrunken comment and on the original. A miniature that
parses differently from its source would make the test guard a fiction, so the
check is a hard filter rather than a warning.

``parse_per_model_reasoning_text`` is public but deliberately OUT of scope: the
redaction below exists to elide rationale prose, which is exactly what that
parser returns, so it necessarily diverges on every record. Its key set is
preserved; only the bodies shrink.

Redaction
---------
Comments are real published Metaculus text. Everything that carries no parser
signal is elided: research prose, per-model rationale prose, third-party news
headlines, and the question title. What survives is the structural skeleton the
parsers key on — section headers, ``*Forecaster N*`` bullets, ``Model:`` lines,
percentile/probability/option value lines, and fenced JSON blocks.

Usage
-----
    uv run python scripts/derive_mini_comment_fixture.py
    uv run python scripts/derive_mini_comment_fixture.py --emit-expectations

Re-run the first form after a fixture pull introduces a genuinely new comment
shape (the shape guard in the test suite fails loudly when the miniature stops
covering one). The second form re-renders the test's exact per-record
expectation table from the CHECKED-IN fixture, so it needs no local pull; see
``render_expectations`` for what it emits and why the diff deserves a read.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from metaculus_bot.performance_analysis.parsing import (
    parse_forecaster_model_map,
    parse_inferred_stacker_outcome,
    parse_per_model_forecasts,
    parse_per_model_mc_option_probs,
    parse_per_model_numeric_percentiles,
    parse_stacked_marker,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger: logging.Logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SOURCE_PATH: Path = REPO_ROOT / "scratch" / "performance_data.json"
OUTPUT_PATH: Path = REPO_ROOT / "tests" / "data" / "performance_comments_mini.jsonl"

# The test module that consumes ``--emit-expectations`` output, and the name it
# binds the table to. Passed to the formatter as the stdin filename so the
# emitted block is wrapped exactly as it will be once pasted in.
TEST_MODULE_PATH: Path = REPO_ROOT / "tests" / "test_performance_analysis_parsing.py"
EXPECTATIONS_VARIABLE: str = "_EXPECTED_PARSES_BY_POST"

TRIM_NOTICE: str = "[... trimmed for length]"
RESEARCH_SUMMARY_MARKER: str = "### Research Summary"

RESEARCH_STUB: str = "[research prose elided in test fixture]"
PROSE_STUB: str = "[prose elided in test fixture]"
TITLE_STUB: str = "*Question*: [title elided in test fixture]"

_FORECAST_SECTION_RE: re.Pattern[str] = re.compile(r"^=+\s*\nFORECAST SECTION:", re.MULTILINE)
_R1_BLOCK_SPLIT_RE: re.Pattern[str] = re.compile(r"(?m)^(?=##\s+R\d+:\s+Forecaster\s+\d+\s+Reasoning)")
_TITLE_RE: re.Pattern[str] = re.compile(r"(?m)^\*Question\*:.*$")

# Lines carrying a forecast value the parsers extract: percentile lines, binary
# probability lines, and multiple-choice option lines.
_VALUE_LINE_RE: re.Pattern[str] = re.compile(
    r"(?mi)\A\s*(?:Percentile\s+[\d.]+\s*:|(?:final\s+)?probability\s*:|-\s+.+:\s*[\d.]+\s*%)"
)

# Structural lines in the comment HEAD (everything above the rationales). Bold
# prose (``**Some news headline**``) is deliberately excluded — it is
# third-party text, not structure.
_STRUCTURAL_HEAD_RE: re.Pattern[str] = re.compile(
    r"\A(?:#{1,3} \S|\*Forecaster\s+\d+|-\s+.+:\s*[\d.]+\s*%|\*Question\*|\*Final Prediction\*)"
)

# Fields the parsing tests read, plus identity/era context for a human reader.
_KEPT_FIELDS: tuple[str, ...] = (
    "post_id",
    "question_id",
    "type",
    "comment_text",
    "bot_comment_created_at",
    "stacker_outcome",
    "stacker_outcome_source",
    "forecasters_used",
    "forecasters_configured",
)


def _elide_gaps(lines: list[str], keep_indices: set[int], stub: str) -> str:
    """Join ``lines`` keeping only ``keep_indices``, one ``stub`` per dropped run."""
    kept: list[str] = []
    previous = -1
    for index in sorted(keep_indices):
        if index > previous + 1:
            kept.append(stub)
        kept.append(lines[index])
        previous = index
    return "\n".join(kept)


def _shrink_rationale_block(block: str) -> str:
    """Elide prose from one ``## R1: Forecaster N Reasoning`` block.

    Keeps the header, the ``Model:`` line, every forecast-value line, every
    fenced JSON block verbatim, sub-headers, and the block's final two lines
    (a trailing ``Probability:`` often sits under the closing prose).
    """
    lines = block.split("\n")
    keep: set[int] = {index for index, line in enumerate(lines) if _VALUE_LINE_RE.match(line)}
    keep.update(range(min(2, len(lines))))  # header + Model: line
    keep.update(index for index, line in enumerate(lines) if line.startswith("## "))
    keep.update(range(max(0, len(lines) - 2), len(lines)))

    inside_fence = False
    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            inside_fence = not inside_fence
            keep.add(index)
        elif inside_fence:
            keep.add(index)

    return _elide_gaps(lines, keep, PROSE_STUB)


def _shrink_head(head: str, *, comment: str) -> str:
    """Elide research prose from the comment head, preserving its structure."""
    marker_index = head.find(RESEARCH_SUMMARY_MARKER)
    if marker_index >= 0:
        shrunk = head[: marker_index + len(RESEARCH_SUMMARY_MARKER)] + "\n" + RESEARCH_STUB + "\n"
        if TRIM_NOTICE in comment:
            shrunk += TRIM_NOTICE + "\n"
        return shrunk

    # No ``### Research Summary`` marker: this record exercises the
    # boundary-fallback path in ``_summary_section_for_bullets``, so the ABSENCE
    # of the marker is itself the property under test and must survive.
    lines = head.split("\n")
    keep = {index for index, line in enumerate(lines) if _STRUCTURAL_HEAD_RE.match(line)}
    shrunk = _elide_gaps(lines, keep, RESEARCH_STUB) + "\n"
    if TRIM_NOTICE in comment and TRIM_NOTICE not in shrunk:
        shrunk += TRIM_NOTICE + "\n"
    return shrunk


def shrink_comment(comment: str) -> str:
    """Return a small, redacted comment that parses identically to ``comment``."""
    forecast_section = _FORECAST_SECTION_RE.search(comment)
    if forecast_section is not None:
        head, rest = comment[: forecast_section.start()], comment[forecast_section.start() :]
    else:
        split = _R1_BLOCK_SPLIT_RE.split(comment, maxsplit=1)
        head = split[0]
        rest = split[1] if len(split) > 1 else ""

    blocks = _R1_BLOCK_SPLIT_RE.split(rest)
    rationales = [blocks[0]] + [_shrink_rationale_block(block) for block in blocks[1:]]
    shrunk = _shrink_head(head, comment=comment) + "".join(rationales)
    return _TITLE_RE.sub(TITLE_STUB, shrunk)


def parser_outputs(comment: str) -> dict[str, object]:
    """Every public per-model parse of ``comment``, keyed by parser name.

    Two consumers: the faithfulness filter in ``build_fixture`` (miniature must
    parse identically to its source) and ``render_expectations``, which writes
    this same mapping into the test suite as the exact per-record expectation.
    Keyed by NAME rather than position for the second consumer's sake — a tuple
    would let a reordering here silently re-label every expectation in the test.
    """
    return {
        "parse_per_model_forecasts": parse_per_model_forecasts(comment),
        "parse_forecaster_model_map": parse_forecaster_model_map(comment),
        "parse_per_model_numeric_percentiles": parse_per_model_numeric_percentiles(comment),
        "parse_per_model_mc_option_probs": parse_per_model_mc_option_probs(comment),
        "parse_inferred_stacker_outcome": parse_inferred_stacker_outcome(comment),
        "parse_stacked_marker": parse_stacked_marker(comment),
    }


def comment_shape(record: dict) -> dict[str, str]:
    """Classify a record by the comment properties the parsers branch on.

    Keyed by axis NAME rather than position: the shape is written into the fixture
    and re-checked by the test suite, and positional access there would silently
    check the wrong axis if this ever gained or reordered a field.
    """
    comment = record["comment_text"]
    anonymized = [key for key in parse_per_model_forecasts(comment) if key.startswith("Forecaster ")]
    return {
        "attribution": "map" if parse_forecaster_model_map(comment) else "nomap",
        "trim": "trim" if TRIM_NOTICE in comment else "intact",
        "boundary_marker": "res" if RESEARCH_SUMMARY_MARKER in comment else "nores",
        "question_type": str(record["type"]),
        "naming": "anon" if anonymized else "named",
    }


def build_fixture(records: list[dict]) -> list[dict]:
    """Pick one faithful miniature per distinct comment shape, ordered by post_id."""
    with_comments = sorted(
        (record for record in records if record.get("comment_text")),
        key=lambda record: record["post_id"],
    )
    seen: set[tuple[tuple[str, str], ...]] = set()
    skipped_unfaithful: Counter[tuple[tuple[str, str], ...]] = Counter()
    fixture: list[dict] = []
    for record in with_comments:
        shape = comment_shape(record)
        shape_key = tuple(sorted(shape.items()))
        if shape_key in seen:
            continue
        miniature = shrink_comment(record["comment_text"])
        if parser_outputs(miniature) != parser_outputs(record["comment_text"]):
            skipped_unfaithful[shape_key] += 1
            continue
        seen.add(shape_key)
        entry = {field: record.get(field) for field in _KEPT_FIELDS}
        entry["comment_text"] = miniature
        entry["_shape"] = shape
        fixture.append(entry)

    for shape, count in sorted(skipped_unfaithful.items()):
        logger.info(f"skipped unfaithful candidates before finding a keeper: {shape=} {count=}")
    return sorted(fixture, key=lambda entry: entry["post_id"])


def render_expectations(fixture: list[dict]) -> str:
    """Render the test suite's exact per-record expectation table as Python source.

    The values come from running the REAL parsers over ``fixture``, so this is a
    CHARACTERIZATION of current behavior, not an independent specification. That
    is the point (hand-transcribing twelve nested dicts is how typos become
    "expected" values) and also the risk: regenerating after a parser change will
    happily bless the change. A diff in the emitted table is a signal to READ the
    diff and confirm each moved value is intended, never to rubber-stamp.
    """
    table = {entry["post_id"]: parser_outputs(entry["comment_text"]) for entry in fixture}
    source = f"{EXPECTATIONS_VARIABLE} = {table!r}\n"
    # Formatted by the repo's own formatter so the emitted block can be pasted in
    # without a follow-up `make format` reflowing it into a different diff.
    formatted = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", str(TEST_MODULE_PATH), "-"],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    )
    return formatted.stdout


def _load_fixture(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _emit_expectations(fixture_path: Path) -> None:
    if not fixture_path.exists():
        raise SystemExit(f"miniature fixture not found at {fixture_path}; derive it first (drop --emit-expectations)")
    sys.stdout.write(render_expectations(_load_fixture(fixture_path)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH, help="big local performance pull")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="miniature fixture to write")
    parser.add_argument(
        "--emit-expectations",
        action="store_true",
        help=(
            f"print the test suite's {EXPECTATIONS_VARIABLE} table for the EXISTING miniature at --output "
            "and exit, without touching the fixture or needing --source"
        ),
    )
    args = parser.parse_args()

    if args.emit_expectations:
        _emit_expectations(args.output)
        return

    if not args.source.exists():
        raise SystemExit(
            f"source pull not found at {args.source}; run the performance-analysis collector first "
            "(see AGENTS.md 'Residual / performance analysis')"
        )

    with args.source.open() as handle:
        records = json.load(handle)

    fixture = build_fixture(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for entry in fixture:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    comment_bytes = sum(len(entry["comment_text"]) for entry in fixture)
    logger.info(f"wrote {len(fixture)} records to {args.output} ({comment_bytes=})")
    for entry in fixture:
        logger.info(f"  post={entry['post_id']} type={entry['type']} shape={entry['_shape']}")


if __name__ == "__main__":
    main()
