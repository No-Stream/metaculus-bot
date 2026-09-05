"""Detecting an IN-PLACE re-resolution or re-score between two performance pulls.

Metaculus can change a question's resolution after the fact without moving any timestamp
we store. On 2026-08-31 it resolved q44798 (post 44645, "Halo: Campaign Evolved
Metascore") at 80 — the PS5 hero card on Metacritic — and then, some time in the next
26 hours, edited it to 82, the Xbox card the resolution criteria actually name.
``resolution_set_time`` still read 2026-08-31T21:38:45Z afterwards, which PRECEDES the
pull that read 80, so nothing timestamp-shaped could have flagged the edit. The record's
spot peer went from +5.41 to -5.42 between two consecutive rounds, and every table the
earlier round had published about that question was silently stale.

So the only reliable detector is a value-level diff of the pull against its predecessor.
:func:`diff_platform_rescores` does that, tagging each re-pulled record in place. It
compares two things:

* the RESOLUTION itself (``resolution_raw`` and its parsed form), and
* every field of ``metaculus_scores`` — the union of the keys on both sides, so a field
  Metaculus adds later is diffed without an edit here.

The tag is deliberately a THREE-state answer, because "we compared and nothing moved" and
"we never compared" are different facts and the second is what a run with no ``--prior``
produces:

* ``platform_rescored is None`` — no prior record existed for this (question, post), so
  nothing was compared. Also the state of every record when no prior dataset is supplied
  at all.
* ``platform_rescored is False`` — compared against a prior record, nothing moved.
* ``platform_rescored is True`` — compared, and at least one field moved.
  ``platform_rescored_fields`` names which, ``prior_resolution`` carries the prior
  ``resolution_raw`` (equal to the current one on a score-only re-score, so a reader can
  tell the two cases apart), and ``prior_metaculus_scores`` carries the prior score block
  so a downstream table can show both numbers without re-reading the old file.

Naming note: the field is ``platform_rescored_fields`` rather than the ``rescored_fields``
the prototype used, because ``scratch/residual_2026-09-01/bucket_by_era.py`` already
spends ``rescored_fields`` on something else entirely — the BOT-side score fields
``collector.rescore_records`` healed from stored inputs. One name for "our scorer changed
its mind" and "Metaculus changed the resolution" would make the round's own JSON
unreadable.

Bot-side score fields (``log_score``, ``numeric_log_score``, ``mc_log_score``,
``brier_score``) are NOT compared here. They are pure functions of inputs the record
carries, so a change in one means our scorer changed, not the platform — that is
``rescore_records``'s job, and mixing the two would attribute our own fix to Metaculus.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

logger: logging.Logger = logging.getLogger(__name__)

# Resolution fields compared verbatim. ``resolution_raw`` is the string Metaculus reported
# ("82", "yes", "above_upper_bound"); ``resolution_parsed`` is our coercion of it, carried
# too so a parse change shows up as its own field rather than hiding behind an equal raw.
RESOLUTION_FIELDS: tuple[str, ...] = ("resolution_raw", "resolution_parsed")

PLATFORM_SCORE_BLOCK = "metaculus_scores"

# Recomputation/serialization wiggle versus a genuinely different value. The platform's own
# scores round-trip through JSON exactly and our scorer reproduces them to ~1e-14, while the
# real gaps this exists to catch are whole points (the known-stale q44798 gaps start at 0.6).
# ``collector.rescore_records`` imports this rather than keeping its own copy, so the
# "changed" threshold is the same number on both sides of a round comparison.
RESCORE_ATOL: float = 1e-6


@dataclass(frozen=True, slots=True)
class FieldChange:
    """One field whose value differs between the prior pull and this one."""

    question_id: int | None
    post_id: int | None
    field: str
    old: object
    new: object


@dataclass(frozen=True, slots=True)
class RescoreDiff:
    """What the comparison found, for the caller to print and for tests to assert on."""

    compared: int
    rescored: int
    unmatched: int
    changes: tuple[FieldChange, ...] = ()
    duplicate_prior_keys: tuple[tuple[int | None, int | None], ...] = ()

    @property
    def rescored_question_ids(self) -> tuple[int | None, ...]:
        """Question ids with at least one changed field, in first-seen order."""
        return tuple(dict.fromkeys(change.question_id for change in self.changes))


def _record_key(record: dict) -> tuple[int | None, int | None]:
    """The identity a record is matched on. Both ids, never one.

    Question ids and post ids share one integer space on Metaculus (the 2026-07-26 degraded
    cohort's question ids 44870-44877 overlap minibench POST ids), so matching on either
    alone can pair two different questions.
    """
    return (record.get("question_id"), record.get("post_id"))


def _values_differ(old: object, new: object) -> bool:
    """Whether two stored values are genuinely different, with float tolerance.

    None on one side and a value on the other IS a change: a score appearing or vanishing
    is exactly the kind of platform move this exists to surface. Numbers compare within
    ``RESCORE_ATOL`` (which covers ``resolution_parsed``'s bool arm too, since a binary
    resolution flipping reads as 1 against 0); everything else compares exactly, which is
    what ``resolution_raw`` strings and multiple-choice option labels need.
    """
    if old is None or new is None:
        return old is not new
    if isinstance(old, int | float) and isinstance(new, int | float):
        return abs(float(old) - float(new)) > RESCORE_ATOL
    return old != new


def _index_prior(
    prior_records: Iterable[dict],
) -> tuple[dict[tuple[int | None, int | None], dict], list[tuple[int | None, int | None]]]:
    """Index the prior pull by (question_id, post_id), reporting any duplicated key.

    A duplicate means the prior file holds two records for one question, which makes the
    comparison ambiguous — the last wins, and the key is reported rather than swallowed.
    """
    index: dict[tuple[int | None, int | None], dict] = {}
    duplicates: list[tuple[int | None, int | None]] = []
    for record in prior_records:
        key = _record_key(record)
        if key in index:
            duplicates.append(key)
        index[key] = record
    return index, duplicates


def _score_field_names(prior_scores: dict, new_scores: dict) -> list[str]:
    """Every platform score key on either side, prior order first then new-only keys."""
    names = list(prior_scores)
    names.extend(name for name in new_scores if name not in prior_scores)
    return names


def _changes_for_record(prior: dict, record: dict) -> list[FieldChange]:
    """Every resolution or platform-score field that moved between the two records."""
    question_id, post_id = _record_key(record)
    changes: list[FieldChange] = []

    for name in RESOLUTION_FIELDS:
        old, new = prior.get(name), record.get(name)
        if _values_differ(old, new):
            changes.append(FieldChange(question_id=question_id, post_id=post_id, field=name, old=old, new=new))

    prior_scores = prior.get(PLATFORM_SCORE_BLOCK) or {}
    new_scores = record.get(PLATFORM_SCORE_BLOCK) or {}
    for name in _score_field_names(prior_scores, new_scores):
        old, new = prior_scores.get(name), new_scores.get(name)
        if _values_differ(old, new):
            changes.append(
                FieldChange(
                    question_id=question_id,
                    post_id=post_id,
                    field=f"{PLATFORM_SCORE_BLOCK}.{name}",
                    old=old,
                    new=new,
                )
            )
    return changes


def diff_platform_rescores(prior_records: Sequence[dict], new_records: Sequence[dict]) -> RescoreDiff:
    """Tag every record in ``new_records`` that Metaculus re-resolved or re-scored.

    Mutates ``new_records`` in place (the same shape ``collector.rescore_records`` uses) and
    returns the summary. Every record gets ``platform_rescored`` /
    ``platform_rescored_fields`` / ``prior_resolution`` / ``prior_metaculus_scores`` set,
    with None on the two absent-prior arms so a downstream cut can never read "not compared"
    as "unchanged" (see the module docstring for the three states).

    Emits one WARN per changed field, so a re-resolution is visible in a round's log without
    reading the returned object.
    """
    prior_index, duplicates = _index_prior(prior_records)
    if duplicates:
        logger.warning(
            f"PLATFORM_RESCORED: prior dataset holds {len(duplicates)} duplicated "
            f"(question_id, post_id) key(s), last wins; first five: {duplicates[:5]} "
            "(all of them are on the returned RescoreDiff.duplicate_prior_keys)"
        )

    compared = 0
    unmatched = 0
    all_changes: list[FieldChange] = []

    for record in new_records:
        prior = prior_index.get(_record_key(record))
        if prior is None:
            _apply_tag(record, prior=None, changes=[])
            unmatched += 1
            continue
        compared += 1
        changes = _changes_for_record(prior, record)
        _apply_tag(record, prior=prior, changes=changes)
        for change in changes:
            logger.warning(
                f"PLATFORM_RESCORED: question={change.question_id} post={change.post_id} "
                f"field={change.field} old={change.old} new={change.new}"
            )
        all_changes.extend(changes)

    diff = RescoreDiff(
        compared=compared,
        rescored=len(dict.fromkeys((c.question_id, c.post_id) for c in all_changes)),
        unmatched=unmatched,
        changes=tuple(all_changes),
        duplicate_prior_keys=tuple(duplicates),
    )
    logger.info(
        f"PLATFORM_RESCORED_SUMMARY: compared={diff.compared} rescored={diff.rescored} "
        f"unmatched={diff.unmatched} changed_fields={len(diff.changes)}"
    )
    return diff


def _apply_tag(record: dict, *, prior: dict | None, changes: Sequence[FieldChange]) -> None:
    """Write the four tag fields onto one record."""
    if prior is None:
        record["platform_rescored"] = None
        record["platform_rescored_fields"] = None
        record["prior_resolution"] = None
        record["prior_metaculus_scores"] = None
        return
    record["platform_rescored"] = bool(changes)
    record["platform_rescored_fields"] = [change.field for change in changes]
    # Only carried on a record that actually moved: on an unchanged record the current
    # values ARE the prior ones, and copying them onto every record would double the size
    # of a dataset to say nothing.
    record["prior_resolution"] = prior.get("resolution_raw") if changes else None
    record["prior_metaculus_scores"] = prior.get(PLATFORM_SCORE_BLOCK) if changes else None


def _tag_field_values(record: dict, name: str) -> tuple[object, object]:
    """The (old, new) pair behind one entry of a record's ``platform_rescored_fields``."""
    if name.startswith(f"{PLATFORM_SCORE_BLOCK}."):
        key = name.split(".", 1)[1]
        prior_scores = record.get("prior_metaculus_scores") or {}
        new_scores = record.get(PLATFORM_SCORE_BLOCK) or {}
        return prior_scores.get(key), new_scores.get(key)
    if name == "resolution_raw":
        return record.get("prior_resolution"), record.get("resolution_raw")
    # resolution_parsed: the tag stores only the raw prior value, so the old parse is not
    # recoverable from the record. Reported as None rather than guessed — resolution_raw
    # moves with it in every real case, and that row carries the values.
    return None, record.get(name)


def render_rescore_summary(records: Sequence[dict]) -> list[str]:
    """Human-readable summary of the rescore tags already on ``records``.

    Reads the tags rather than re-running the comparison, so a caller that has already
    diffed (``build_performance_dataset(prior_records=...)``, or
    :func:`diff_platform_rescores` directly) gets its summary without a second pass and
    without a second set of WARN lines. Every old value printed comes off the record's own
    ``prior_resolution`` / ``prior_metaculus_scores``.

    Carries the module's three-state tag through to the printed text: a zero ``compared``
    count says nothing moved BECAUSE nothing was compared, which is a different fact from a
    clean diff and must not print the reassurance. The reachable trigger is a ``--prior``
    pointed at another round or another tournament, whose records share no
    (question_id, post_id) with this pull at all.
    """
    compared = sum(1 for record in records if record.get("platform_rescored") is not None)
    rescored = [record for record in records if record.get("platform_rescored") is True]
    lines = [
        f"Platform re-resolution diff: {compared} of {len(records)} record(s) matched a prior pull, "
        f"{len(rescored)} re-scored or re-resolved."
    ]
    if compared == 0:
        # Cause-agnostic: a dataset that never went through the diff and a prior pull with no
        # overlapping key produce the identical tags, so the text claims neither.
        lines.append(
            "  Nothing was compared: no record carries a prior-pull match, so this says nothing "
            "about whether prior-round tables are current."
        )
        return lines
    if not rescored:
        lines.append("  No resolution or platform score moved. Prior-round tables remain current.")
        return lines
    for record in rescored:
        for name in record.get("platform_rescored_fields") or []:
            old, new = _tag_field_values(record, name)
            lines.append(f"  q{record.get('question_id')} (post {record.get('post_id')}) {name}: {old!r} -> {new!r}")
    lines.append(
        "  Metaculus changed the resolution or the score on the questions above. Any prior-round table "
        "quoting them is stale, and their own timestamps may not have moved at all."
    )
    return lines
