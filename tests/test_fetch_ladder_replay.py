"""Tests for the archive replay tool (``scripts/fetch_ladder_replay.py``).

Five things are pinned. The two record parsers against inline fixtures shaped like the real
archive; the reconstruction of a direct :class:`FetchResult` from an archived record, including
the rescued record whose direct status only survives as an escalation attempt's ``from_status``;
the totality of the inverse loop-status table over the status literals the gap-fill v2 tool
handlers actually emit; that a loop outcome produced by the loop's OWN rescuing rung is counted
but never replayed; and one end-to-end replay whose per-record rung sequence is derived below.

Why those three sequences, read off ``fetch_ladder/ladder._escalate_unresolved`` and the rung
trigger predicates, with every rung stubbed to record its attempt and decline:

``blocked`` with ``http_status=403`` — ``_impersonate_rung_applies`` is ``blocked`` plus an HTTP
status in ``impersonated_fetch.IMPERSONATE_TRIGGER_STATUSES`` (403), so ``impersonate`` fires;
``_rendered_rung_applies`` takes only ``js_wall`` and the ``thin_page`` reason, so both browser
rungs are never consulted; ``blocked`` is in ``rungs._WAYBACK_TRIGGER_STATUSES``, so ``wayback``
fires; ``_url_context_rung_applies`` takes ``blocked`` with no excluded reason, so
``url_context`` fires last. Sequence: impersonate, wayback, url_context.

``js_wall`` with ``http_status=200`` — not ``blocked``, so no impersonated retry;
``_rendered_rung_applies`` is true, so the dispatcher enters the browser gate and consults
``derived_api`` then ``rendered``; ``js_wall`` is deliberately absent from the Wayback triggers
(the archive stores the unrendered shell); ``js_wall`` is a paid-read trigger. Sequence:
derived_api, rendered, url_context.

``success`` — ``_escalate_unresolved`` returns the direct result before any rung. Sequence: empty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

import metaculus_bot.research.agentic as agentic_package
from metaculus_bot.research.fetch_ladder import policy as ladder_policy
from metaculus_bot.research.impersonated_fetch import IMPERSONATE_TRIGGER_STATUSES
from scripts.fetch_ladder_replay import (
    LOOP_RESCUE_METHODS,
    LOOP_STATUS_TO_FETCH_STATUS,
    ArchivedFetch,
    direct_result,
    discover_presets,
    fetcher_records,
    loop_records,
    main,
    parse_tool_headers,
    recover_http_status,
    replay_archive,
)

_STATUS_LITERAL_RE = re.compile(r'status="([a-z_]+)"')

_BLOCKED_403 = {
    "url": "https://blocked.example.gov/a",
    "status": "blocked",
    "text": "",
    "http_status": 403,
    "content_type": "text/html",
}
_JS_WALL = {
    "url": "https://wall.example.org/b",
    "status": "js_wall",
    "text": "",
    "http_status": 200,
    "content_type": "text/html; charset=utf-8",
}
_SUCCESS = {
    "url": "https://plain.example.net/c",
    "status": "success",
    "text": "The committee published the figure on 3 March.",
    "http_status": 200,
    "content_type": "text/html",
}
_RESCUED_BY_PAID_READ = {
    "url": "https://rescued.example.gov/d",
    "status": "no_resolving_content",
    "status_reason": "not_addressed",
    "text": "",
    "http_status": 403,
    "content_type": "text/html",
    "route": "url_context",
    "rung_attempts": [
        {
            "rung": "wayback",
            "from_status": "blocked",
            "url": "https://rescued.example.gov/d",
            "started_at": 1.0,
            "wall_s": 2.0,
            "outcome": "stale_data",
            "skipped_reason": "",
        },
        {
            "rung": "url_context",
            "from_status": "blocked",
            "url": "https://rescued.example.gov/d",
            "started_at": 4.0,
            "wall_s": 2.0,
            "outcome": "no_resolving_content",
            "skipped_reason": "",
        },
    ],
}


def _write_archive(root: Path, *, fetcher_payload: list[dict[str, Any]], transcript: list[dict[str, Any]]) -> Path:
    """A miniature research archive with one raw run file and one by_qid question file."""
    (root / "raw").mkdir(parents=True)
    (root / "by_qid").mkdir(parents=True)
    (root / "raw" / "1001.jsonl").write_text(
        json.dumps({"qid": 44001, "provider": "resolution_source", "phase": None, "payload": fetcher_payload})
        + "\n"
        + json.dumps({"qid": 44001, "provider": "asknews", "phase": "hot", "payload": [{"url": "ignored"}]})
        + "\n"
    )
    (root / "by_qid" / "44002.jsonl").write_text(
        json.dumps({"qid": 44002, "run_id": "r1", "gap_fill_v2": {"transcript": transcript}}) + "\n"
    )
    return root


def _tool_exchange(call_id: str, url: str, body: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "fetch", "arguments": json.dumps({"url": url})},
                }
            ],
        },
        {"role": "tool", "name": "fetch", "tool_call_id": call_id, "content": body},
    ]


class TestFetcherRecordParsing:
    def test_reads_only_the_resolution_source_provider(self, tmp_path: Path) -> None:
        archive = _write_archive(tmp_path / "archive", fetcher_payload=[_BLOCKED_403, _JS_WALL], transcript=[])
        records = list(fetcher_records(archive))
        assert [record.url for record in records] == [_BLOCKED_403["url"], _JS_WALL["url"]]
        assert [record.direct_status for record in records] == ["blocked", "js_wall"]
        assert [record.http_status for record in records] == [403, 200]

    def test_a_record_without_a_route_field_reads_as_direct_and_says_so(self, tmp_path: Path) -> None:
        archive = _write_archive(tmp_path / "archive", fetcher_payload=[_BLOCKED_403], transcript=[])
        record = next(iter(fetcher_records(archive)))
        assert record.archived_route == "direct"
        assert any("route" in note for note in record.notes)

    def test_a_rescued_record_takes_its_direct_status_from_the_first_escalation_attempt(self, tmp_path: Path) -> None:
        archive = _write_archive(tmp_path / "archive", fetcher_payload=[_RESCUED_BY_PAID_READ], transcript=[])
        record = next(iter(fetcher_records(archive)))
        assert record.archived_route == "url_context"
        assert record.direct_status == "blocked"
        assert record.status_reason is None
        assert record.http_status == 403


class TestDirectResultReconstruction:
    def test_a_success_carries_placeholder_text_and_a_failure_carries_none(self) -> None:
        success = direct_result(
            ArchivedFetch(
                source="fetcher",
                qid="1",
                url="https://a.example/x",
                direct_status="success",
                status_reason=None,
                http_status=200,
                content_type="text/html",
                archived_route="direct",
                notes=(),
            )
        )
        assert success.text
        assert success.status == "success"
        failure = direct_result(
            ArchivedFetch(
                source="fetcher",
                qid="1",
                url="https://a.example/y",
                direct_status="blocked",
                status_reason=None,
                http_status=403,
                content_type="text/html",
                archived_route="direct",
                notes=(),
            )
        )
        assert failure.text == ""
        assert failure.http_status == 403

    def test_the_status_reason_survives_reconstruction(self) -> None:
        result = direct_result(
            ArchivedFetch(
                source="fetcher",
                qid="1",
                url="https://a.example/z",
                direct_status="no_resolving_content",
                status_reason="thin_page",
                http_status=200,
                content_type="text/html",
                archived_route="direct",
                notes=(),
            )
        )
        assert result.status_reason == "thin_page"


class TestLoopRecordParsing:
    def test_headers_and_the_recovered_http_status(self) -> None:
        body = "tool: fetch\nstatus: blocked\nmethod: plain\n\nFetch blocked with HTTP 403.\n[budget: 500s remaining]"
        assert parse_tool_headers(body) == ("blocked", "plain")
        assert recover_http_status(body) == 403

    def test_a_not_found_recovers_through_the_fetchers_own_http_status_table(self, tmp_path: Path) -> None:
        body = "tool: fetch\nstatus: error\nmethod: plain\n\nFetch failed with HTTP 404.\n[budget: 490s remaining]"
        archive = _write_archive(
            tmp_path / "archive",
            fetcher_payload=[],
            transcript=_tool_exchange("c1", "https://gone.example.gov/p", body),
        )
        record = next(iter(loop_records(archive)))
        assert record.url == "https://gone.example.gov/p"
        assert record.direct_status == "not_found"
        assert record.http_status == 404

    def test_the_platform_self_reference_refusal_reads_as_that_reason(self, tmp_path: Path) -> None:
        body = (
            "tool: fetch\nstatus: blocked\nmethod: plain\n\n"
            "Metaculus pages are already reflected in the question brief; do not fetch metaculus.com URLs."
        )
        archive = _write_archive(
            tmp_path / "archive",
            fetcher_payload=[],
            transcript=_tool_exchange("c1", "https://www.metaculus.com/questions/1/", body),
        )
        record = next(iter(loop_records(archive)))
        assert record.direct_status == "blocked"
        assert record.status_reason == "metaculus_self_ref"

    def test_an_outcome_produced_by_the_loops_own_rescue_rung_is_recorded_but_not_replayable(
        self, tmp_path: Path
    ) -> None:
        body = "tool: fetch\nstatus: ok\nmethod: rendered\n\nThe rendered page's text."
        archive = _write_archive(
            tmp_path / "archive",
            fetcher_payload=[],
            transcript=_tool_exchange("c1", "https://spa.example.org/q", body),
        )
        record = next(iter(loop_records(archive)))
        assert record.archived_route == "rendered"
        assert record.direct_status is None
        assert not record.replayable
        assert "rendered" in LOOP_RESCUE_METHODS

    def test_the_status_table_is_total_over_the_loops_own_status_literals(self) -> None:
        """Every ``status="..."`` literal in the gap-fill v2 tool handlers needs an inverse entry.

        Derived from source rather than pinned, so a new tool status fails here until the table
        names it. Its one weakness is a status named only in prose, which reads as a real token
        and demands an entry; that is a cheap failure to resolve and the reverse is not.
        """
        agentic_dir = Path(agentic_package.__file__).parent
        emitted = {
            literal
            for path in sorted(agentic_dir.glob("*.py"))
            for literal in _STATUS_LITERAL_RE.findall(path.read_text())
        }
        assert emitted
        assert emitted <= set(LOOP_STATUS_TO_FETCH_STATUS)


class TestPresetDiscovery:
    def test_every_ladder_policy_attribute_is_found_by_default(self) -> None:
        found = discover_presets(None)
        assert "RESOLUTION_SOURCE_POLICY" in found
        assert found["RESOLUTION_SOURCE_POLICY"] is ladder_policy.RESOLUTION_SOURCE_POLICY

    def test_an_unknown_preset_name_is_an_error(self) -> None:
        with pytest.raises(SystemExit):
            discover_presets(["NO_SUCH_POLICY"])


class TestEndToEndReplay:
    @pytest.mark.asyncio
    async def test_the_rung_sequence_per_archived_outcome(self, tmp_path: Path) -> None:
        assert 403 in IMPERSONATE_TRIGGER_STATUSES
        archive = _write_archive(
            tmp_path / "archive",
            fetcher_payload=[_BLOCKED_403, _JS_WALL, _SUCCESS],
            transcript=[],
        )
        report = await replay_archive(archive, {"RESOLUTION_SOURCE_POLICY": ladder_policy.RESOLUTION_SOURCE_POLICY})
        by_url = {replay.record.url: replay for replay in report.replays}
        assert by_url[_BLOCKED_403["url"]].sequence == ("impersonate", "wayback", "url_context")
        assert by_url[_JS_WALL["url"]].sequence == ("derived_api", "rendered", "url_context")
        assert by_url[_SUCCESS["url"]].sequence == ()

    @pytest.mark.asyncio
    async def test_every_replayed_status_equals_the_direct_status(self, tmp_path: Path) -> None:
        archive = _write_archive(
            tmp_path / "archive", fetcher_payload=[_BLOCKED_403, _JS_WALL, _SUCCESS], transcript=[]
        )
        report = await replay_archive(archive, {"RESOLUTION_SOURCE_POLICY": ladder_policy.RESOLUTION_SOURCE_POLICY})
        assert all(replay.final_status == replay.record.direct_status for replay in report.replays)

    @pytest.mark.asyncio
    async def test_a_rescue_the_dispatcher_no_longer_attempts_is_flagged(self, tmp_path: Path) -> None:
        """A js_wall page the archive rescued from Wayback: js_wall is not a Wayback trigger today."""
        archived_wayback_rescue = dict(_JS_WALL, route="wayback")
        archive = _write_archive(tmp_path / "archive", fetcher_payload=[archived_wayback_rescue], transcript=[])
        report = await replay_archive(archive, {"RESOLUTION_SOURCE_POLICY": ladder_policy.RESOLUTION_SOURCE_POLICY})
        assert [replay.rescue_lost for replay in report.replays] == [True]

    @pytest.mark.asyncio
    async def test_a_rescue_that_served_bytes_leaves_its_rung_verdict_undecidable(self, tmp_path: Path) -> None:
        """A rung that served bytes reports its OWN http_status, so the impersonated retry's trigger
        (a 403 on the direct fetch) is no longer readable off the record and the verdict must not
        claim the rung would be skipped."""
        rescued = dict(
            _SUCCESS,
            http_status=200,
            route="impersonate",
            rung_attempts=[
                {
                    "rung": "impersonate",
                    "from_status": "blocked",
                    "url": _SUCCESS["url"],
                    "started_at": 1.0,
                    "wall_s": 1.0,
                    "outcome": "success",
                    "skipped_reason": "",
                }
            ],
        )
        archive = _write_archive(tmp_path / "archive", fetcher_payload=[rescued], transcript=[])
        report = await replay_archive(archive, {"RESOLUTION_SOURCE_POLICY": ladder_policy.RESOLUTION_SOURCE_POLICY})
        assert [replay.rescue_lost for replay in report.replays] == [False]
        assert [replay.rescue_undecidable for replay in report.replays] == [True]


class TestCli:
    def test_a_missing_archive_directory_exits_non_zero_naming_the_flag(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--archive-dir", str(tmp_path / "absent")])
        assert excinfo.value.code != 0

    def test_the_table_names_the_presets_it_replayed(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        archive = _write_archive(
            tmp_path / "archive", fetcher_payload=[_BLOCKED_403, _JS_WALL, _SUCCESS], transcript=[]
        )
        main(["--archive-dir", str(archive)])
        printed = capsys.readouterr().out
        assert "RESOLUTION_SOURCE_POLICY" in printed
        assert "impersonate" in printed

    def test_the_json_format_carries_full_urls(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        archive = _write_archive(tmp_path / "archive", fetcher_payload=[_BLOCKED_403], transcript=[])
        main(["--archive-dir", str(archive), "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert _BLOCKED_403["url"] in json.dumps(payload)
