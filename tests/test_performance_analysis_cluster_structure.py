"""The correlation-cluster rule and the curated tables it reads.

Three concerns kept apart. The RULE (which strengths collapse, and the arithmetic of effective n)
is pinned on synthetic cohorts built inline, so the numbers are checkable by hand. The GUARDS are
pinned as raises, because a cluster naming a question outside the cohort or claimed twice silently
mislabels a round's evidence. The tracked ASSET is pinned as a file git really carries and as a
shape the rule accepts, because the repo's blanket ``*.json`` ignore has already let a fixture
exist locally, be absent on a fresh clone, and leave the tests green by construction.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from metaculus_bot.performance_analysis.cluster_structure import (
    CURATED_TABLES_PATH,
    Cluster,
    ClusterStrength,
    ClusterTables,
    RoundCohort,
    RoundNotes,
    TemplateFamily,
    assign_clusters,
    build_structure,
    effective_n,
    render_report,
    singleton_cluster_id,
)
from metaculus_bot.performance_analysis.era_gap import ClusterMap
from metaculus_bot.performance_analysis.eras import POST_FLIP, TRIPLE_ERA

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROUND_ARTIFACTS = _REPO_ROOT / "scratch" / "residual_2026-09-09"

_NOTES = RoundNotes(label="2026-09-09", prior_round="a/prior/round", caveats=("one caveat",), siblings_note="a note")


def _record(question_id: int, **overrides: object) -> dict:
    record = {
        "question_id": question_id,
        "post_id": question_id - 100,
        "type": "binary",
        "config_era": TRIPLE_ERA,
        "is_new_since_prior": True,
        "resolution_set_time": "2026-09-01T00:00:00Z",
        "actual_resolve_time": "2026-09-01T00:00:00Z",
        "title": f"question {question_id}",
        "spot_peer_score": 1.0,
        "peer_score": 1.0,
    }
    return record | overrides


def _tables(
    *clusters: Cluster, retired: dict[str, dict] | None = None, families: tuple[TemplateFamily, ...] = ()
) -> ClusterTables:
    return ClusterTables(
        round_notes=_NOTES,
        clusters=clusters,
        retired_clusters=retired or {},
        template_families=families,
        rejected_links=(),
        pending_additions=(),
    )


def _cluster(cluster_id: str, strength: ClusterStrength, *question_ids: int) -> Cluster:
    return Cluster(cluster_id=cluster_id, strength=strength, basis="because", question_ids=question_ids)


@pytest.fixture(scope="module")
def curated_tables() -> ClusterTables:
    return ClusterTables.load()


class TestEffectiveN:
    """Strong collapses always, weak only as a sensitivity, single never."""

    STRONG = _cluster("one_event", ClusterStrength.STRONG, 1, 2, 3)
    WEAK = _cluster("one_regime", ClusterStrength.WEAK, 4, 5)
    TABLES = _tables(STRONG, WEAK)
    COHORT = (1, 2, 3, 4, 5, 6, 7)

    def test_strong_collapses_and_weak_does_not(self) -> None:
        assignment = assign_clusters(self.COHORT, self.TABLES)
        # 1 (the strong trio) + 2 (weak, still separate) + 2 unclustered = 5.
        assert effective_n(self.COHORT, assignment, collapse_weak=False) == 5
        # The weak pair collapses too: 1 + 1 + 2 = 4.
        assert effective_n(self.COHORT, assignment, collapse_weak=True) == 4

    def test_records_outside_the_measured_slice_do_not_collapse_it(self) -> None:
        assignment = assign_clusters(self.COHORT, self.TABLES)
        assert effective_n([1], assignment, collapse_weak=False) == 1
        assert effective_n([1, 6], assignment, collapse_weak=False) == 2

    def test_an_unclustered_question_is_its_own_cluster(self) -> None:
        assignment = assign_clusters(self.COHORT, self.TABLES)
        assert assignment.cluster_id(6) == singleton_cluster_id(6) == "single_6"
        assert assignment.strength(6) is ClusterStrength.SINGLE

    def test_collapse_key_is_the_singleton_id_for_anything_that_does_not_collapse(self) -> None:
        assignment = assign_clusters(self.COHORT, self.TABLES)
        assert assignment.collapse_key(1, collapse_weak=False) == "one_event"
        assert assignment.collapse_key(4, collapse_weak=False) == "single_4"
        assert assignment.collapse_key(4, collapse_weak=True) == "one_regime"


class TestGuardsFailShut:
    def test_a_member_outside_the_cohort_raises_and_names_its_cluster(self) -> None:
        tables = _tables(_cluster("half_outside", ClusterStrength.STRONG, 1, 99))
        with pytest.raises(ValueError, match="half_outside"):
            assign_clusters([1, 2], tables)

    def test_a_question_claimed_by_two_clusters_raises(self) -> None:
        tables = _tables(
            _cluster("first", ClusterStrength.STRONG, 1, 2),
            _cluster("second", ClusterStrength.WEAK, 2, 3),
        )
        with pytest.raises(ValueError, match="claimed by both first and second"):
            assign_clusters([1, 2, 3], tables)

    def test_a_retired_cluster_that_still_owns_a_member_raises(self) -> None:
        tables = _tables(
            _cluster("still_live", ClusterStrength.STRONG, 1, 2),
            retired={"still_live": {"strength": "strong", "members": [1, 2], "note": "retired too early"}},
        )
        with pytest.raises(ValueError, match="un-retire"):
            assign_clusters([1, 2], tables)

    def test_a_retired_cluster_whose_members_moved_to_new_ids_is_fine(self) -> None:
        """The split case: a retired id keeps continuity while its members live on under new ids."""
        tables = _tables(
            _cluster("the_split_half", ClusterStrength.STRONG, 1, 2),
            retired={"the_old_name": {"strength": "weak", "members": [1, 2, 3], "note": "split"}},
        )
        assert assign_clusters([1, 2], tables).cluster_of == {1: "the_split_half", 2: "the_split_half"}


class TestRoundCohort:
    RECORDS = (
        _record(1),
        _record(2, config_era=POST_FLIP),
        _record(3, is_new_since_prior=False),
        _record(4, is_new_since_prior=False, config_era=POST_FLIP),
        _record(5, degraded_run=True),
        _record(6, partial_degraded=True),
        _record(7, known_bug=True),
    )

    def test_the_cohort_is_the_new_wave_union_the_live_era(self) -> None:
        cohort = RoundCohort.from_records(self.RECORDS)
        assert cohort.question_ids == (1, 2, 3, 5, 6, 7)
        assert cohort.new_question_ids == (1, 2, 5, 6, 7)
        assert cohort.triple_question_ids == (1, 3, 5, 6, 7)
        assert cohort.n_overlap == 4

    def test_slice_names_carry_their_record_count(self) -> None:
        slices = RoundCohort.from_records(self.RECORDS).slices()
        assert list(slices) == [
            "new_all_5",
            "new_post_flip_1",
            "new_triple_4",
            "triple_all_5",
            "triple_clean_3",
            "triple_strict_2",
            "triple_strict_new_only_1",
        ]

    def test_clean_drops_degraded_and_known_bug_and_strict_also_drops_partial(self) -> None:
        slices = RoundCohort.from_records(self.RECORDS).slices()
        assert slices["triple_clean_3"] == [1, 3, 6]
        assert slices["triple_strict_2"] == [1, 3]
        assert slices["triple_strict_new_only_1"] == [1]


class TestBuildStructure:
    RECORDS = (_record(1), _record(2), _record(3, resolution_set_time="2026-09-02T00:00:00Z"))
    TABLES = _tables(
        _cluster("one_event", ClusterStrength.STRONG, 1, 2),
        families=(TemplateFamily(name="a_family", basis="one template", question_ids=(1, 3, 3, 99)),),
    )

    @pytest.fixture
    def structure(self) -> dict:
        return build_structure(self.RECORDS, self.TABLES, source="a/dataset.json")

    def test_every_cohort_record_gets_a_cluster(self, structure: dict) -> None:
        assert structure["qid_to_cluster"] == {"1": "one_event", "2": "one_event", "3": "single_3"}

    def test_cluster_entries_carry_the_strength_era_gap_selects_on(self, structure: dict) -> None:
        assert structure["clusters"]["one_event"]["strength"] == "strong"
        assert structure["clusters"]["single_3"]["strength"] == "single"
        assert [m["question_id"] for m in structure["clusters"]["one_event"]["members"]] == [1, 2]

    def test_template_families_are_deduped_sorted_and_filtered_to_the_cohort(self, structure: dict) -> None:
        assert structure["template_families"]["a_family"]["qids"] == [1, 3]

    def test_the_rule_text_reports_the_batching_the_rule_refuses_to_cluster_on(self, structure: dict) -> None:
        assert "over 2 days with at most 2 on any one day" in structure["rule"]
        assert structure["resolution_set_date_histogram"] == {"2026-09-01": 2, "2026-09-02": 1}

    def test_the_round_notes_pass_through_verbatim(self, structure: dict) -> None:
        assert structure["generated"] == "2026-09-09"
        assert structure["caveats"] == ["one caveat"]
        assert structure["primary_numbers_for_siblings"]["note"] == "a note"
        assert "3 records new since a/prior/round" in structure["cohort"]

    def test_era_gap_reads_only_the_strong_members_back_out(self, structure: dict) -> None:
        """The contract that matters: what this writes is what `era_gap --clusters` collapses."""
        cluster_map = ClusterMap.from_cluster_structure(structure, source="a/structure.json")
        assert cluster_map.strong == {"1": "one_event", "2": "one_event"}

    def test_the_console_report_names_the_clusters_and_the_cohorts(self, structure: dict) -> None:
        report = render_report(structure)
        assert "[strong] one_event  n=2" in report
        assert "single_3" not in report
        assert "new_all_3" in report


class TestCuratedTablesAsset:
    """The tracked asset: really tracked, and a shape the rule accepts."""

    def test_git_tracks_the_asset_so_a_fresh_clone_has_it(self) -> None:
        listed = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "--error-unmatch", str(CURATED_TABLES_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert listed.returncode == 0, (
            f"{CURATED_TABLES_PATH.name} is untracked: the blanket `*.json` ignore needs its negation in .gitignore"
        )

    def test_the_curated_ids_are_question_ids_not_post_ids(self, curated_tables: ClusterTables) -> None:
        """Post and question ids share one integer namespace, so the asset states which it holds."""
        every_id = [q for cluster in curated_tables.clusters for q in cluster.question_ids]
        assert every_id
        assert all(isinstance(q, int) for q in every_id)
        assert len(every_id) == len(set(every_id)), "a question id is claimed by two clusters"

    def test_cluster_ids_are_unique_and_retired_ids_are_not_reused(self, curated_tables: ClusterTables) -> None:
        live = [cluster.cluster_id for cluster in curated_tables.clusters]
        assert len(live) == len(set(live))
        assert not set(live) & set(curated_tables.retired_clusters)

    def test_every_strength_is_one_the_rule_knows(self, curated_tables: ClusterTables) -> None:
        assert {cluster.strength for cluster in curated_tables.clusters} <= set(ClusterStrength)

    def test_no_cluster_is_a_lone_member(self, curated_tables: ClusterTables) -> None:
        """A one-member cluster is a singleton wearing a name; it would collapse nothing."""
        assert all(len(cluster.question_ids) >= 2 for cluster in curated_tables.clusters)


@pytest.mark.skipif(
    not (_ROUND_ARTIFACTS / "cluster_structure.json").exists(),
    reason="the dated round artifacts live in the private sibling repo `scratch/` symlink, absent in CI",
)
class TestReproducesTheCuratedRound:
    """The asset plus the rule reproduce the 2026-09-09 round's artifact, the one era_gap was read on."""

    def test_every_field_and_its_order_matches_but_for_the_dataset_path(self) -> None:
        dataset = _ROUND_ARTIFACTS / "dim_category_slim.json"
        expected = json.loads((_ROUND_ARTIFACTS / "cluster_structure.json").read_text())
        built = build_structure(json.loads(dataset.read_text()), ClusterTables.load(), source=expected["source"])
        assert built == expected
        # Rounds are diffed against each other, so key order is part of the artifact.
        assert json.dumps(built, indent=2) == json.dumps(expected, indent=2)
