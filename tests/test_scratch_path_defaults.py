"""No argparse default or module constant may name a round-dated ``scratch/`` path.

Two different failures come from the same literal. A fresh clone of this public repo has no
``scratch/`` at all (it is gitignored, and on the operator's machine it is a symlink into a
private artifact repo), so a default pointing there gives a confusing missing-file error in
code that looks configured. Worse, a default naming a SPECIFIC round directory
(``scratch/residual_2026-09-01/...``) keeps resolving on the operator's own machine for weeks
after that round is superseded, so the run quietly measures last month's dataset. Silent stale
data is the expensive half; the missing file is merely rude.

The rule is therefore narrow on purpose: a round-dated scratch path may not be a default, and
must be passed. An undated one (``scratch/probes`` as an output root, ``scratch/performance_data.json``
behind an explicit existence check that names what to run) is fine and stays.

Docstrings and comments are invisible here, so the receipts convention -- a module citing
``scratch/residual_2026-08-24/degraded_cohort.json`` as the provenance of a constant -- is
untouched. Only executable values are checked.
"""

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Where a default that a fresh clone has to satisfy can live: the package and its tooling.
_SCANNED_ROOTS: tuple[str, ...] = ("metaculus_bot", "scripts")

_ROUND_DATE = re.compile(r"\d{4}-\d{2}")


def _string_constants(node: ast.AST) -> list[str]:
    return [child.value for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)]


def _names_a_round_dated_scratch_path(node: ast.AST) -> bool:
    """Whether the expression's string parts together spell a dated path under ``scratch``.

    Both spellings have to be caught: one flat literal (``"scratch/residual_2026-09-01/x.json"``)
    and a ``Path`` join whose segments are separate constants (``ROOT / "scratch" / "residual_2026-09-01"``).
    """
    constants = _string_constants(node)
    return any("scratch" in text for text in constants) and any(_ROUND_DATE.search(text) for text in constants)


def _offending_expressions(tree: ast.Module) -> list[tuple[int, str]]:
    """Every module-level constant and every ``add_argument(default=...)`` that names one."""
    offenders: list[tuple[int, str]] = []

    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        if _names_a_round_dated_scratch_path(value):
            names = ", ".join(ast.unparse(target) for target in targets)
            offenders.append((statement.lineno, f"module constant {names}"))

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
        ):
            continue
        flag = next((text for text in _string_constants(node) if text.startswith("-")), "<positional>")
        for keyword in node.keywords:
            if keyword.arg == "default" and _names_a_round_dated_scratch_path(keyword.value):
                offenders.append((node.lineno, f"argparse default for {flag}"))

    return offenders


def test_no_round_dated_scratch_path_is_a_default() -> None:
    findings: list[str] = []
    for root in _SCANNED_ROOTS:
        for path in sorted((_REPO_ROOT / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            relative = path.relative_to(_REPO_ROOT).as_posix()
            findings += [f"{relative}:{lineno}: {what}" for lineno, what in _offending_expressions(tree)]

    assert not findings, "round-dated scratch paths must be passed, not defaulted:\n" + "\n".join(findings)


class TestTheGuardItself:
    """The check has to catch both spellings and stay quiet on the legitimate ones."""

    def test_a_flat_literal_default_is_caught(self) -> None:
        tree = ast.parse('p.add_argument("--cached", default="scratch/residual_2026-09-01/perf_all_tagged.json")')

        assert [what for _lineno, what in _offending_expressions(tree)] == ["argparse default for --cached"]

    def test_a_path_join_split_across_constants_is_caught(self) -> None:
        tree = ast.parse('PAIRS = ROOT / "scratch" / "cost_pass_2026-09-09" / "pairs.jsonl"')

        assert [what for _lineno, what in _offending_expressions(tree)] == ["module constant PAIRS"]

    def test_an_undated_scratch_output_root_is_allowed(self) -> None:
        tree = ast.parse('OUT = ROOT / "scratch" / "probes"\np.add_argument("--out-root", default=OUT)')

        assert _offending_expressions(tree) == []

    def test_a_dated_path_outside_scratch_is_allowed(self) -> None:
        tree = ast.parse('ARCHIVE = ROOT / "backtests" / "research_archive" / "2026-09-01"')

        assert _offending_expressions(tree) == []

    def test_a_receipt_in_a_docstring_or_comment_is_invisible(self) -> None:
        tree = ast.parse('"""Receipts: scratch/residual_2026-08-24/degraded_cohort.json."""\n# scratch/x_2026-01/y\n')

        assert _offending_expressions(tree) == []
