"""Standing scoring-exclusion cohorts and the ``--exclude-qids`` token parser.

A cohort is defined by an INCIDENT (a since-fixed pipeline bug, a degraded-run
window), not by what resolved into any one pull, so these sets are cross-cutting:
the width monitor, per-question audits and round scripts all filter on them. They
live in this leaf module — importing nothing from the package — so any sibling
module can reach them at module top without a cycle.
"""

from __future__ import annotations

import logging

logger: logging.Logger = logging.getLogger(__name__)

# The CANONICAL known-pipeline-bug cohort: questions whose published forecast was
# produced by a since-fixed pipeline defect rather than by judgment, so pooling them
# into a calibration row measures the old bug instead of the current bot. Not
# excluded by default — callers pass the set explicitly so an exclusion is always a
# visible choice, and every row reports how many it dropped. Import this constant
# rather than re-hardcoding the ids: analysis scripts that kept private copies have
# already drifted from it.
#
# - 43746 (Minions & Monsters) and 43747 (Toy Story 5) opening-weekend gross:
#   the pre-2026-07-07 open-bound arithmetic bug.
# - 43913 (WSOP bracelets held by the 2026 Main Event winner), added 2026-08-25:
#   the pre-`9f1175c` discrete max-step cap. All six forecasters stated 79.5-83%
#   on the outcome that resolved (exactly 1 bracelet) and the published CDF carried
#   20.00%, its first bin pinned at exactly 0.200000 on an 11-point grid — the
#   201-grid ceiling applied to a 10-bin question whose real server ceiling was 4.0.
#   Receipts: scratch/residual_2026-08-24/dossiers/43913_dossier.md and
#   dim_discrete-maxstep-counterfactual.md. The fix reached prod inside `b4e9df0`
#   (2026-07-21T17:07:37Z), so no post-triple-era question can carry this shape.
KNOWN_BUG_QIDS: frozenset[str] = frozenset({"43746", "43747", "43913"})

# The CANONICAL degraded-ensemble cohort: the dry-donated-key incident window
# (2026-07-26 .. 07-28), when OpenRouter reported the drained donated key's breached spend
# cap as a 403 the classifier vetoed, so the wrapper never fell back to the funded personal
# key. These questions published at 1 of 3 forecasters — gemini alone, the only slot pinned
# to the personal key by DONATED_KEY_BLOCKED_GOOGLE_MODELS — with native search, the AskNews
# summarizer, the financial classifier, prediction-market keyword extraction and BOTH
# gap-fill passes down alongside it. Their forecasts came from a thinned pipeline rather
# than the current bot, so they belong in the same "exclude from headline aggregates, report
# separately" bucket as KNOWN_BUG_QIDS.
#
# THESE ARE QUESTION IDS. The same eight questions carry post ids 44721-44728, and the two
# spaces share one integer namespace: minibench POST ids 44873-44877 land inside this
# question-id range, so a join that matches "either id" silently admits five unrelated
# questions. Key every join on the question id, translating through
# ``performance_analysis.id_mapping`` when the other side is post-keyed.
#
# Receipts: scratch/residual_2026-08-24/degraded_cohort.json; incident write-up in
# docs/operations.md "What a dry donated key actually returns".
DEGRADED_RUN_QIDS: frozenset[str] = frozenset({"44870", "44871", "44872", "44873", "44874", "44875", "44876", "44877"})

# The PARTIAL half of the same incident: published at 2 of 3 forecasters. Kept separate
# because the forecaster count differs, but note 44841 and 44856 are degraded IDENTICALLY
# to the full cohort on the RESEARCH side (native search errored, both gap-fill passes
# dead), so any research-conditioned cut must exclude both sets together even though a
# forecaster-count cut can legitimately keep these three.
PARTIAL_DEGRADED_QIDS: frozenset[str] = frozenset({"44841", "44856", "44912"})

# The known-bug cohort's CLI shorthand. Named because --help quotes it as the worked
# example, and a literal there could drift from the dict key below.
KNOWN_BUG_SHORTHAND = "known_bug"

# The CLI tokens standing in for a whole cohort in --exclude-qids. Each name is the
# cohort's canonical shorthand; adding a cohort here is all it takes to make it available
# to every caller of parse_exclude_qids.
EXCLUSION_COHORTS: dict[str, frozenset[str]] = {
    KNOWN_BUG_SHORTHAND: KNOWN_BUG_QIDS,
    "degraded_run": DEGRADED_RUN_QIDS,
    "partial_degraded": PARTIAL_DEGRADED_QIDS,
}


def parse_exclude_qids(raw: str) -> frozenset[str]:
    """A ``--exclude-qids`` comma list, with every cohort shorthand expanded in place.

    A shorthand COMPOSES with explicit ids rather than only standing alone. It used to be
    recognized only as the whole argument, so ``--exclude-qids known_bug,43800`` produced the
    literal set ``{"known_bug", "43800"}``: no question id matches the word, so the bug pair
    stayed in every row while the table's ``excl`` column reported one exclusion and looked
    like it had worked.

    A token that is neither a known cohort name nor a bare question id RAISES. With one
    shorthand a typo was survivable; with three, ``degraded`` for ``degraded_run`` would
    exclude nothing while the ``excl`` column read 0 — indistinguishable from a cohort
    whose questions simply aren't in this pull.
    """
    tokens = {token.strip() for token in raw.split(",") if token.strip()}
    cohort_names = tokens & EXCLUSION_COHORTS.keys()
    # ASCII-only digits: str.isdigit() also accepts fullwidth and superscript digits, which
    # would pass this guard and then match no question id, i.e. the silent no-op it exists
    # to prevent.
    unknown = sorted(t for t in tokens - cohort_names if not (t.isascii() and t.isdigit()))
    if unknown:
        raise ValueError(
            f"--exclude-qids: {unknown} is neither a question id nor a cohort shorthand "
            f"({', '.join(sorted(EXCLUSION_COHORTS))})"
        )
    qids = tokens - cohort_names
    for name in sorted(cohort_names):
        cohort = EXCLUSION_COHORTS[name]
        logger.info(f"--exclude-qids: expanded cohort {name} -> {len(cohort)} question ids")
        qids |= cohort
    return frozenset(qids)
