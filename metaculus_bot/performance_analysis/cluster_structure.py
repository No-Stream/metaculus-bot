"""The correlation-cluster rule: what a round quotes as effective n instead of a record count.

Several resolved questions in any wave are one real-world event or one dated statistical release
seen from different angles, so a record count overstates the evidence. Each question id gets
exactly one cluster: ``strong`` means one driver mechanically resolves every member, so it
collapses to one observation; ``weak`` means a shared regime, so residuals correlate but the draws
are separate and it collapses only as a sensitivity; ``single`` means independent. The same
resolution-set DATE, the same CATEGORY and the same question TEMPLATE are none of them cluster
bases, and template families (method correlation) are reported without ever being collapsed.

Which questions share a driver is a human's per-round judgement, so the tables live in the curated
JSON asset beside this module (:data:`CURATED_TABLES_PATH`) rather than in code, and this module is
only the rule over them. The emitted ``cluster_structure.json`` is what ``era_gap --clusters``
reads. The rule's history, the asset's shape and how a round edits it:
``docs/performance_analysis.md`` "Building a round's cluster structure".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from metaculus_bot.performance_analysis.eras import POST_FLIP, TRIPLE_ERA

logger: logging.Logger = logging.getLogger(__name__)

CURATED_TABLES_PATH: Path = Path(__file__).with_name("cluster_tables.json")

# The rule statement carried into every emitted artifact; the two counts are the round's own.
RULE_TEMPLATE = (
    "One cluster per shared resolution driver. strength=strong collapses to one observation; "
    "strength=weak means correlated residuals, collapse only as a sensitivity; strength=single "
    "means independent. The same resolution-set DATE, the same CATEGORY and the same question "
    "TEMPLATE are all NOT cluster bases: Metaculus spread this wave's resolutions over "
    "{n_days} days with at most {biggest_batch} on any one day, on unrelated quantities, and "
    "template families are reported separately."
)

SINGLETON_BASIS = "no sibling inside the cohort"


class ClusterStrength(StrEnum):
    """How far a cluster collapses: strong to one observation, weak only as a sensitivity, single never."""

    STRONG = "strong"
    WEAK = "weak"
    SINGLE = "single"


def singleton_cluster_id(question_id: int | str) -> str:
    """The cluster id an unclustered question gets, so every record has one."""
    return f"single_{question_id}"


@dataclass(frozen=True, slots=True)
class Cluster:
    cluster_id: str
    strength: ClusterStrength
    basis: str
    question_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TemplateFamily:
    """A question TEMPLATE shared by unrelated quantities: reported for method correlation, never collapsed."""

    name: str
    basis: str
    question_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RoundNotes:
    """The round's own prose: which round curated the tables, and the caveats it published with them."""

    label: str
    prior_round: str
    caveats: tuple[str, ...]
    siblings_note: str

    @classmethod
    def from_dict(cls, payload: dict) -> RoundNotes:
        return cls(
            label=payload["label"],
            prior_round=payload["prior_round"],
            caveats=tuple(payload["caveats"]),
            siblings_note=payload["siblings_note"],
        )


@dataclass(frozen=True, slots=True)
class ClusterTables:
    """One round's curated cluster judgement, read from the tracked JSON asset.

    Every id here is a QUESTION id, never a post id: the two share one integer namespace, so a
    table matched against post ids admits unrelated questions (see ``cohorts.py``).
    """

    round_notes: RoundNotes
    clusters: tuple[Cluster, ...]
    retired_clusters: dict[str, dict]
    template_families: tuple[TemplateFamily, ...]
    rejected_links: tuple[dict, ...]
    pending_additions: tuple[dict, ...]

    @classmethod
    def from_dict(cls, payload: dict) -> ClusterTables:
        return cls(
            round_notes=RoundNotes.from_dict(payload["round_notes"]),
            clusters=tuple(
                Cluster(
                    cluster_id=cid,
                    strength=ClusterStrength(entry["strength"]),
                    basis=entry["basis"],
                    question_ids=tuple(entry["question_ids"]),
                )
                for cid, entry in payload["clusters"].items()
            ),
            retired_clusters=payload["retired_clusters"],
            template_families=tuple(
                TemplateFamily(name=name, basis=entry["basis"], question_ids=tuple(entry["question_ids"]))
                for name, entry in payload["template_families"].items()
            ),
            rejected_links=tuple(payload["rejected_links"]),
            pending_additions=tuple(payload["pending_additions"]),
        )

    @classmethod
    def load(cls, path: str | Path = CURATED_TABLES_PATH) -> ClusterTables:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass(frozen=True, slots=True)
class Assignment:
    """Every cohort question id mapped to its one cluster id, with each cluster's collapse strength."""

    cluster_of: dict[int, str]
    strength_of: dict[str, ClusterStrength]

    def cluster_id(self, question_id: int) -> str:
        return self.cluster_of.get(question_id, singleton_cluster_id(question_id))

    def strength(self, question_id: int) -> ClusterStrength:
        return self.strength_of.get(self.cluster_id(question_id), ClusterStrength.SINGLE)

    def collapse_key(self, question_id: int, *, collapse_weak: bool) -> str:
        """The key one question contributes to effective n: its cluster only where that cluster collapses."""
        strength = self.strength(question_id)
        collapses = strength is ClusterStrength.STRONG or (collapse_weak and strength is ClusterStrength.WEAK)
        return self.cluster_id(question_id) if collapses else singleton_cluster_id(question_id)


def assign_clusters(cohort_question_ids: Iterable[int], tables: ClusterTables) -> Assignment:
    """Label the cohort, failing shut on a member outside it, a double claim or a live retired id."""
    cohort = set(cohort_question_ids)
    cluster_of: dict[int, str] = {}
    outside: dict[str, list[int]] = {}
    for cluster in tables.clusters:
        for question_id in cluster.question_ids:
            if question_id not in cohort:
                outside.setdefault(cluster.cluster_id, []).append(question_id)
                continue
            if question_id in cluster_of:
                raise ValueError(
                    f"qid {question_id} claimed by both {cluster_of[question_id]} and {cluster.cluster_id}"
                )
            cluster_of[question_id] = cluster.cluster_id
    if outside:
        raise ValueError(f"cluster members outside the cohort must be named in basis text only: {outside}")
    for cid, meta in tables.retired_clusters.items():
        still_owned = [q for q in meta["members"] if cluster_of.get(q) == cid]
        if still_owned:
            raise ValueError(f"retired cluster {cid} still owns {still_owned}; un-retire it")
    return Assignment(
        cluster_of=cluster_of,
        strength_of={cluster.cluster_id: cluster.strength for cluster in tables.clusters},
    )


def effective_n(question_ids: Iterable[int], assignment: Assignment, *, collapse_weak: bool) -> int:
    """Distinct collapse keys: the number of independent draws the cohort's records really carry."""
    return len({assignment.collapse_key(q, collapse_weak=collapse_weak) for q in question_ids})


@dataclass(frozen=True, slots=True)
class RoundCohort:
    """A round's measured cohort: the wave new since the prior round UNION the whole live-roster era."""

    records: dict[int, dict]
    new_question_ids: tuple[int, ...]
    triple_question_ids: tuple[int, ...]

    @classmethod
    def from_records(cls, records: Sequence[dict]) -> RoundCohort:
        new = [r for r in records if r.get("is_new_since_prior")]
        triple = [r for r in records if r.get("config_era") == TRIPLE_ERA]
        union = {r["question_id"]: r for r in (*new, *triple)}
        return cls(
            records=dict(sorted(union.items())),
            new_question_ids=tuple(sorted(r["question_id"] for r in new)),
            triple_question_ids=tuple(sorted(r["question_id"] for r in triple)),
        )

    @property
    def question_ids(self) -> tuple[int, ...]:
        return tuple(self.records)

    @property
    def n_overlap(self) -> int:
        return len(set(self.new_question_ids) & set(self.triple_question_ids))

    def slices(self) -> dict[str, list[int]]:
        """The cohorts effective n is reported for; each name carries its own record count."""
        known_bug = {q for q, r in self.records.items() if r.get("known_bug")}
        new_post_flip = [q for q in self.new_question_ids if self.records[q].get("config_era") == POST_FLIP]
        new_triple = [q for q in self.new_question_ids if self.records[q].get("config_era") == TRIPLE_ERA]
        clean = [q for q in self.triple_question_ids if not self.records[q].get("degraded_run") and q not in known_bug]
        strict = [q for q in clean if not self.records[q].get("partial_degraded")]
        strict_new_only = [q for q in strict if q in set(new_triple)]
        named = (
            ("new_all", list(self.new_question_ids)),
            ("new_post_flip", new_post_flip),
            ("new_triple", new_triple),
            ("triple_all", list(self.triple_question_ids)),
            ("triple_clean", clean),
            ("triple_strict", strict),
            ("triple_strict_new_only", strict_new_only),
        )
        return {f"{name}_{len(qids)}": qids for name, qids in named}


def _member(question_id: int, record: dict) -> dict:
    """One cluster member as the artifact carries it; field order is load-bearing, rounds get diffed."""
    return {
        "question_id": question_id,
        "post_id": record.get("post_id"),
        "type": record.get("type"),
        "config_era": record.get("config_era"),
        "triple_subera_fine": record.get("triple_subera_fine"),
        "category": (record.get("category") or "").replace("\xad", "") or None,
        "spot_peer_score": record.get("spot_peer_score"),
        "peer_score": record.get("peer_score"),
        "question_weight": record.get("question_weight"),
        "actual_resolve_time": record.get("actual_resolve_time"),
        "resolution_set_time": record.get("resolution_set_time"),
        "is_new_since_prior": bool(record.get("is_new_since_prior")),
        "degraded_run": bool(record.get("degraded_run")),
        "partial_degraded": bool(record.get("partial_degraded")),
        "known_bug": bool(record.get("known_bug")),
        "title": record.get("title"),
    }


def _cluster_entries(cohort: RoundCohort, tables: ClusterTables, assignment: Assignment) -> dict[str, dict]:
    basis_of = {cluster.cluster_id: cluster.basis for cluster in tables.clusters}
    entries: dict[str, dict] = {}
    for question_id, record in cohort.records.items():
        cid = assignment.cluster_id(question_id)
        entry = entries.setdefault(
            cid,
            {
                "cluster_id": cid,
                "strength": str(assignment.strength(question_id)),
                "basis": basis_of.get(cid, SINGLETON_BASIS),
                "members": [],
            },
        )
        entry["members"].append(_member(question_id, record))
    return entries


def _effective_n_block(cohort: RoundCohort, assignment: Assignment) -> dict[str, dict]:
    return {
        name: {
            "n_records": len(qids),
            "effective_n_strong_only": effective_n(qids, assignment, collapse_weak=False),
            "effective_n_strong_and_weak": effective_n(qids, assignment, collapse_weak=True),
            "qids": qids,
        }
        for name, qids in cohort.slices().items()
    }


def _siblings_numbers(effective: dict[str, dict], cohort: RoundCohort, note: str) -> dict[str, str]:
    def quote(key: str) -> str:
        block = effective[key]
        return (
            f"{block['n_records']} records, effective n {block['effective_n_strong_only']} "
            f"({block['effective_n_strong_and_weak']} collapsing weak clusters)"
        )

    strict_key = next(name for name in effective if name.startswith("triple_strict_") and "new_only" not in name)
    return {
        "triple_strict": quote(strict_key),
        "new_wave": quote(f"new_all_{len(cohort.new_question_ids)}"),
        "note": note,
    }


def build_structure(records: Sequence[dict], tables: ClusterTables, *, source: str) -> dict:
    """The round's ``cluster_structure.json`` payload: the rule applied to the curated tables."""
    cohort = RoundCohort.from_records(records)
    assignment = assign_clusters(cohort.question_ids, tables)
    effective = _effective_n_block(cohort, assignment)
    set_days = Counter((r.get("resolution_set_time") or "")[:10] for r in cohort.records.values())
    new_set_days = Counter((cohort.records[q].get("resolution_set_time") or "")[:10] for q in cohort.new_question_ids)
    notes = tables.round_notes
    return {
        "generated": notes.label,
        "source": source,
        "cohort": (
            f"({len(cohort.new_question_ids)} records new since {notes.prior_round}) UNION (all "
            f"{len(cohort.triple_question_ids)} triple_era records) = {len(cohort.records)} records. "
            f"{cohort.n_overlap} records are in both."
        ),
        "rule": RULE_TEMPLATE.format(
            n_days=len(new_set_days), biggest_batch=max(new_set_days.values()) if new_set_days else 0
        ),
        "qid_to_cluster": {str(q): assignment.cluster_id(q) for q in cohort.question_ids},
        "clusters": _cluster_entries(cohort, tables, assignment),
        "retired_clusters": tables.retired_clusters,
        "effective_n": effective,
        "template_families": {
            family.name: {
                "basis": family.basis,
                "qids": sorted({q for q in family.question_ids if q in cohort.records}),
            }
            for family in tables.template_families
        },
        "rejected_links": list(tables.rejected_links),
        "pending_additions": list(tables.pending_additions),
        "resolution_set_date_histogram": dict(sorted(set_days.items())),
        "primary_numbers_for_siblings": _siblings_numbers(effective, cohort, notes.siblings_note),
        "caveats": list(notes.caveats),
    }


def render_report(structure: dict) -> str:
    """The console view: the non-singleton clusters, effective n per cohort, and the judgement calls."""
    clusters = structure["clusters"]
    counts = Counter(entry["strength"] for entry in clusters.values())
    lines = [
        structure["cohort"],
        "",
        "CLUSTERS (strong and weak; the rest are singletons). Scores are SPOT peer, peer in parens.",
    ]
    for cid, entry in clusters.items():
        if entry["strength"] == ClusterStrength.SINGLE:
            continue
        lines += ["", f"  [{entry['strength']}] {cid}  n={len(entry['members'])}"]
        for m in entry["members"]:
            spot, peer = m["spot_peer_score"], m["peer_score"]
            lines.append(
                f"     qid={m['question_id']:<6} {m['type']:<15} era={m['config_era']:<10} "
                f"new={int(m['is_new_since_prior'])} deg={int(m['degraded_run'])}{int(m['partial_degraded'])} "
                f"spot={'-' if spot is None else f'{spot:+7.2f}'} (peer {'-' if peer is None else f'{peer:+6.2f}'}) "
                f"res={(m['actual_resolve_time'] or '')[:10]}  {(m['title'] or '')[:58]}"
            )
    lines += [
        "",
        f"  strong: {counts[ClusterStrength.STRONG]}   weak: {counts[ClusterStrength.WEAK]}   "
        f"singletons: {counts[ClusterStrength.SINGLE]}   total: {len(clusters)}",
        f"  retired clusters: {list(structure['retired_clusters'])}",
        "",
        "EFFECTIVE n BY COHORT",
        f"  {'cohort':<32} {'records':>8} {'eff n (strong)':>15} {'eff n (strong+weak)':>21}",
    ]
    for name, block in structure["effective_n"].items():
        lines.append(
            f"  {name:<32} {block['n_records']:>8} {block['effective_n_strong_only']:>15} "
            f"{block['effective_n_strong_and_weak']:>21}"
        )
    lines += ["", "TEMPLATE FAMILIES (method correlation, reported and NEVER collapsed for effective n)"]
    lines += [f"  {name:<38} n={len(block['qids']):>2}" for name, block in structure["template_families"].items()]
    lines += ["", "REJECTED LINKS (auditable judgment calls)"]
    lines += [f"  {link['qids']}: {link['reason'][:140]}" for link in structure["rejected_links"]]
    lines += ["", "RESOLUTION-SET-DATE HISTOGRAM (union cohort). Same-day batches are NOT clusters."]
    lines += [f"  {day or '(none)'}  {'#' * n} ({n})" for day, n in structure["resolution_set_date_histogram"].items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Label a round's records with correlation clusters and report effective n (read-only, offline)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="The round's tagged dataset JSON; records need question_id, config_era and is_new_since_prior.",
    )
    parser.add_argument(
        "--tables",
        default=str(CURATED_TABLES_PATH),
        help="The curated cluster tables. Default: %(default)s.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to write the cluster_structure.json that `era_gap --clusters` reads.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
    with open(args.dataset) as f:
        records = json.load(f)
    structure = build_structure(records, ClusterTables.load(args.tables), source=args.dataset)

    # Logging is pinned to stderr above so the rendered report can be piped on its own.
    print(render_report(structure))  # noqa: T201

    if args.output_json:
        tmp = f"{args.output_json}.tmp"
        with open(tmp, "w") as f:
            json.dump(structure, f, indent=2)
        os.replace(tmp, args.output_json)
        logger.info(
            f"Wrote {args.output_json}: {len(structure['qid_to_cluster'])} qids -> {len(structure['clusters'])} clusters"
        )


if __name__ == "__main__":
    main()
