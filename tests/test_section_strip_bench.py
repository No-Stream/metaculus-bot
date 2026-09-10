"""The section-strip bench (``scripts/probes/section_strip_bench``) without a network.

The bench is a PAID probe on the operator's personal OpenRouter key, so what these tests pin is
everything that decides whether and how money moves: the stripping is exact (each arm loses one
whole section and nothing else), the question each pair is scored as is the one production would
build, the prompt is the production template for the type with the clock anchored to the archived
run date, the scoring plumbing turns a canned reply into the platform score through the real
extraction ladder and CDF build, the spend gates refuse and abort where the estimate says they must,
and the donated key is unreachable on the paid path. The model call is a fake returning canned
replies with real ``STRUCTURED FORECAST`` blocks; ``tests/conftest.py``'s egress guard has nothing
to block.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import litellm
import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion, OutOfBoundsResolution

from metaculus_bot import prompts as prompts_module
from metaculus_bot import value_extraction
from metaculus_bot.credit_telemetry import reset_role_spend
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.scoring_common import binary_log_score, mc_log_score
from scripts.probes import section_strip_bench as bench
from scripts.probes.section_strip_bench import cli, plan, run

pytestmark = pytest.mark.usefixtures("_clean_ledger")


@pytest.fixture
def _clean_ledger() -> None:
    reset_role_spend()


RUN_DAY = "2026-07-23"
WINDOW = (
    f"Today: {RUN_DAY}\n"
    "Question opened: 2026-07-01 (22 days ago)\n"
    "Scheduled to resolve: 2026-08-31 (39 days from now)\n"
    "Forecasting window: open date → resolution date. Events occurring BEFORE the open date do NOT resolve this "
    "question YES unless the resolution criteria explicitly say they count. If the question uses forward-looking "
    "language ('will X occur by DATE'), interpret it as asking about the open→resolution window, not all of history."
)
FIRST_PASS = "## News Articles (AskNews)\nFirst-pass evidence.\n\n---\n\n## Native Search\nMore evidence."
V1_BODY = "### Gap 1: What is the latest count?\n*Why it matters:* it decides the bin.\n\n- 4,173 cases as of July 24."
V2_BODY = '### ⚠ Corrections to the briefing\nClaim: none.\n\n### Latest count\n- [fetched] 4,173 cases. "quote"'
OPTIONS = ["Texas", "Arizona", "Nevada", "Other"]
METADATA = {"open_time": "2026-07-01T12:00:00Z", "scheduled_resolve_time": "2026-08-31T16:00:00Z"}


def bundle_text(first: str = FIRST_PASS, v1: str = V1_BODY, v2: str = V2_BODY) -> str:
    """A bundle as gap_fill_stages appends it: separator, header, blank line, body, twice."""
    return f"{first}{bench.SECTION_SEPARATOR}{bench.V1_SECTION_HEADER}\n\n{v1}{bench.SECTION_SEPARATOR}{bench.V2_SECTION_HEADER}\n\n{v2}"


def _header(qtype: str, title: str) -> str:
    if qtype == "binary":
        return f"{title}\n\nType: binary (probability of YES)"
    if qtype == "multiple_choice":
        return f"{title}\n\nType: multiple choice\nOptions: {', '.join(OPTIONS)}"
    return f"{title}\n\nType: numeric\nUnits: Cases\nDisplayed range: [7000.0, 14000.0] (lower bound open, upper bound open)"


def make_pair(qid: int, qtype: str, *, resolution: Any, published: dict[str, Any] | None, title: str) -> dict[str, Any]:
    resolved = resolution is not None
    return {
        "question_id": qid,
        "post_id": qid + 100,
        "title": title,
        "type": qtype,
        "resolution_criteria": "Resolves per the CDC count.",
        "fine_print": None,
        "question_header": _header(qtype, title),
        "forecasting_window": WINDOW,
        "first_pass_research": FIRST_PASS,
        "resolution_raw": (str(resolution).lower() if resolved else None),
        "resolution_parsed": resolution,
        "published_forecast": published,
        "spot_peer_score": 12.5 if resolved else None,
    }


def make_perf(qid: int, qtype: str, *, inbound: int = 200) -> dict[str, Any]:
    scaling = {
        "range_min": 7000,
        "range_max": 14000,
        "nominal_min": 7000,
        "nominal_max": 14000,
        "zero_point": None,
        "inbound_outcome_count": inbound,
    }
    return {
        "question_id": qid,
        "type": qtype,
        "metadata": METADATA,
        "scaling": scaling if qtype in ("numeric", "discrete") else None,
        "open_lower_bound": True,
        "open_upper_bound": True,
        "options": OPTIONS if qtype == "multiple_choice" else None,
    }


def _cdf_201() -> list[float]:
    """A legal open-bound 201-point CDF climbing evenly from 0.02 to 0.98."""
    return [0.02 + 0.96 * i / 200 for i in range(201)]


BINARY_YES = make_pair(101, "binary", resolution=True, published={"prob_yes": 0.7}, title="Will the count pass 10k?")
BINARY_NO = make_pair(
    102, "binary", resolution=False, published={"prob_yes": 0.3}, title="Will the strait stay closed?"
)
MC = make_pair(
    103,
    "multiple_choice",
    resolution="Arizona",
    published={"forecast_values": [0.1, 0.6, 0.2, 0.1], "options": OPTIONS},
    title="Which state hosts the plant?",
)
NUMERIC = make_pair(
    104, "numeric", resolution=9500.0, published={"forecast_values": _cdf_201()}, title="How many cases by September?"
)
DISCRETE = make_pair(
    105,
    "discrete",
    resolution="above_upper_bound",
    published={"forecast_values": [0.01 + 0.98 * i / 81 for i in range(82)]},
    title="How many points will the lead be?",
)
UNRESOLVED = make_pair(106, "binary", resolution=None, published=None, title="Will humans go extinct?")
ALL_PAIRS = [BINARY_YES, BINARY_NO, MC, NUMERIC, DISCRETE, UNRESOLVED]
RESOLVED_IDS = [101, 102, 103, 104, 105]


@pytest.fixture
def dataset(tmp_path: Path) -> dict[str, Path]:
    """pairs.jsonl, perf.json and an archive dir carrying every pair's bundle, under tmp_path."""
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(p) + "\n" for p in ALL_PAIRS), encoding="utf-8")
    perf_rows = [
        make_perf(p["question_id"], p["type"], inbound=81 if p["type"] == "discrete" else 200) for p in ALL_PAIRS
    ]
    perf_path = tmp_path / "perf.json"
    perf_path.write_text(json.dumps(perf_rows), encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    for pair in ALL_PAIRS:
        record = {"qid": pair["question_id"], "source": "artifact", "research_text": bundle_text()}
        (archive / f"{pair['question_id']}.json").write_text(json.dumps(record), encoding="utf-8")
    return {"pairs": pairs_path, "perf": perf_path, "archive": archive, "out": tmp_path / "out"}


def _dataset_args(dataset: dict[str, Path]) -> list[str]:
    return [
        "--pairs",
        str(dataset["pairs"]),
        "--perf-json",
        str(dataset["perf"]),
        "--archive-dir",
        str(dataset["archive"]),
        "--out-root",
        str(dataset["out"]),
    ]


BINARY_REPLY = 'Reasoning.\n\n```json\n{"question_type": "binary", "posterior_prob": 0.22}\n```\n'
MC_REPLY = (
    "Reasoning.\n\n```json\n"
    '{"question_type": "multiple_choice", "option_probs": {"Texas": 0.1, "Arizona": 0.5, "Nevada": 0.3, "Other": 0.1}}'
    "\n```\n"
)


def numeric_reply(values: list[float]) -> str:
    pairs = ", ".join(f'"{p}": {v}' for p, v in zip(STANDARD_PERCENTILES, values, strict=True))
    return (
        "Reasoning.\n\n```json\n"
        f'{{"question_type": "numeric", "declared_percentiles": {{{pairs}}}, "outcome_type": "continuous"}}'
        "\n```\n"
    )


NUMERIC_REPLY = numeric_reply([7500 + 500 * i for i in range(len(STANDARD_PERCENTILES))])
NO_BLOCK_REPLY = "I think about 22 percent but I forgot the block."

REPLY_BY_TITLE = {
    BINARY_YES["title"]: BINARY_REPLY,
    BINARY_NO["title"]: BINARY_REPLY,
    MC["title"]: MC_REPLY,
    NUMERIC["title"]: NUMERIC_REPLY,
    DISCRETE["title"]: NUMERIC_REPLY,
}


class FakeModel:
    """A ``ModelCall`` answering from the question title it finds in the prompt, at a fixed cost per call."""

    def __init__(self, replies: dict[str, str] | None = None, *, cost_usd: float = 0.001) -> None:
        self.replies = replies or REPLY_BY_TITLE
        self.cost_usd = cost_usd
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> run.ModelReply:
        await asyncio.sleep(0)
        self.prompts.append(prompt)
        text = next(reply for title, reply in self.replies.items() if title in prompt)
        return run.ModelReply(
            text=text, prompt_tokens=1000, completion_tokens=300, reasoning_tokens=200, cost_usd=self.cost_usd
        )


def parser_stub() -> MagicMock:
    """A parser the ladder never reaches while the block parses; its ``model`` is what the salvage rung would read."""
    stub = MagicMock()
    stub.model = "openrouter/test/parser"
    return stub


def _run(
    plan_items: list[plan.PlanItem], call: Any, *, cap_usd: float = 10.0, concurrency: int = 2
) -> list[bench.CallRow]:
    meter = run.SpendMeter(cap_usd=cap_usd, extra_usd=lambda: 0.0)
    return asyncio.run(
        run.run_plan(
            plan_items,
            call=call,
            parser_llm=parser_stub(),
            model_name="fake",
            meter=meter,
            concurrency=concurrency,
            on_row=lambda _row: None,
        )
    )


class TestSectionStripping:
    def test_each_arm_loses_exactly_the_named_section(self) -> None:
        sections = bench.split_bundle(bundle_text())
        arms = bench.arm_texts(sections)

        assert arms["full"] == bundle_text()
        assert arms["minus_both"] == FIRST_PASS
        assert arms["minus_v1"] == FIRST_PASS + sections.v2_block
        assert arms["minus_v2"] == FIRST_PASS + sections.v1_block
        assert bench.V1_SECTION_HEADER not in arms["minus_v1"]
        assert V1_BODY not in arms["minus_v1"]
        assert bench.V2_SECTION_HEADER not in arms["minus_v2"]
        assert V2_BODY not in arms["minus_v2"]
        # The separator also sits between first-pass providers, so the cut keys on the headers, never on it.
        assert bench.SECTION_SEPARATOR in arms["minus_both"]
        assert list(arms) == list(bench.ARMS)

    def test_the_blocks_carry_their_own_separator_and_header(self) -> None:
        sections = bench.split_bundle(bundle_text())

        assert sections.v1_block.startswith(bench.SECTION_SEPARATOR + bench.V1_SECTION_HEADER)
        assert sections.v2_block.startswith(bench.SECTION_SEPARATOR + bench.V2_SECTION_HEADER)
        assert sections.v2_block.endswith(V2_BODY)

    @pytest.mark.parametrize(
        ("bundle", "reason"),
        [
            (bundle_text(v1=V1_BODY + "\n" + bench.V2_SECTION_HEADER), "exactly one"),
            (FIRST_PASS + bench.SECTION_SEPARATOR + bench.V2_SECTION_HEADER + "\n\n" + V2_BODY, "exactly one"),
            (
                bundle_text().replace(
                    bench.SECTION_SEPARATOR + bench.V1_SECTION_HEADER, "\n" + bench.V1_SECTION_HEADER
                ),
                "separator",
            ),
        ],
        ids=["duplicate-v2-header", "missing-v1", "v1-without-separator"],
    )
    def test_a_malformed_bundle_is_refused(self, bundle: str, reason: str) -> None:
        with pytest.raises(ValueError, match=reason):
            bench.split_bundle(bundle)

    def test_v2_before_v1_is_refused(self) -> None:
        swapped = (
            f"{FIRST_PASS}{bench.SECTION_SEPARATOR}{bench.V2_SECTION_HEADER}\n\n{V2_BODY}"
            f"{bench.SECTION_SEPARATOR}{bench.V1_SECTION_HEADER}\n\n{V1_BODY}"
        )
        with pytest.raises(ValueError, match="precede"):
            bench.split_bundle(swapped)


class TestQuestionConstruction:
    def test_binary(self) -> None:
        question = bench.build_question(BINARY_YES, make_perf(101, "binary"))

        assert isinstance(question, BinaryQuestion)
        assert question.id_of_question == 101
        assert question.id_of_post == 201
        assert question.page_url == "https://www.metaculus.com/questions/201/"
        assert question.open_time == datetime(2026, 7, 1, 12, tzinfo=UTC)
        assert question.fine_print == ""

    def test_multiple_choice_carries_the_tagged_options(self) -> None:
        question = bench.build_question(MC, make_perf(103, "multiple_choice"))

        assert isinstance(question, MultipleChoiceQuestion)
        assert question.options == OPTIONS

    def test_numeric_reads_bounds_grid_and_unit(self) -> None:
        question = bench.build_question(NUMERIC, make_perf(104, "numeric"))

        assert isinstance(question, NumericQuestion)
        assert not isinstance(question, DiscreteQuestion)
        assert (question.lower_bound, question.upper_bound) == (7000.0, 14000.0)
        assert question.open_lower_bound
        assert question.open_upper_bound
        assert question.cdf_size == 201
        assert question.unit_of_measure == "Cases"
        assert question.nominal_upper_bound == 14000

    def test_discrete_is_a_discrete_question_on_its_own_grid(self) -> None:
        question = bench.build_question(DISCRETE, make_perf(105, "discrete", inbound=81))

        assert isinstance(question, DiscreteQuestion)
        assert question.cdf_size == 82

    def test_the_unitless_placeholder_reads_back_as_none(self) -> None:
        pair = {
            **NUMERIC,
            "question_header": NUMERIC["question_header"].replace("Cases", "unspecified (assume unitless)"),
        }
        question = bench.build_question(pair, make_perf(104, "numeric"))

        assert question.unit_of_measure is None

    def test_resolutions_per_type(self) -> None:
        binary = bench.build_question(BINARY_NO, make_perf(102, "binary"))
        mc = bench.build_question(MC, make_perf(103, "multiple_choice"))
        numeric = bench.build_question(NUMERIC, make_perf(104, "numeric"))
        discrete = bench.build_question(DISCRETE, make_perf(105, "discrete", inbound=81))

        # A NO is False, and stays a scoreable resolution: the pair file's own caveat.
        assert bench.typed_resolution(BINARY_NO, binary) is False
        assert bench.typed_resolution(MC, mc) == "Arizona"
        assert bench.typed_resolution(NUMERIC, numeric) == 9500.0
        assert bench.typed_resolution(DISCRETE, discrete) is OutOfBoundsResolution.ABOVE_UPPER_BOUND

    def test_a_resolution_off_the_option_list_is_refused(self) -> None:
        mc = bench.build_question(MC, make_perf(103, "multiple_choice"))
        with pytest.raises(ValueError, match="not one of"):
            bench.typed_resolution({**MC, "resolution_parsed": "Utah"}, mc)

    def test_load_keeps_every_resolved_pair_including_a_no_and_skips_the_unresolved(
        self, dataset: dict[str, Path]
    ) -> None:
        questions = bench.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"])

        assert [q.question_id for q in questions] == RESOLVED_IDS
        by_id = {q.question_id: q for q in questions}
        assert by_id[102].resolution is False
        assert by_id[101].today == RUN_DAY
        assert by_id[101].arms["minus_both"] == FIRST_PASS
        assert by_id[101].published_score == pytest.approx(binary_log_score(0.7, True))
        assert by_id[103].published_score == pytest.approx(mc_log_score([0.1, 0.6, 0.2, 0.1], 1))
        numeric_published = by_id[104].published_score
        assert numeric_published is not None
        assert math.isfinite(numeric_published)

    def test_load_narrows_to_the_requested_ids_and_names_a_miss(self, dataset: dict[str, Path]) -> None:
        questions = bench.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"], only={101, 104})
        assert [q.question_id for q in questions] == [101, 104]

        with pytest.raises(ValueError, match=r"\[106, 999\]"):
            bench.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"], only={101, 106, 999})

    def test_load_refuses_an_archive_that_disagrees_with_the_pair(self, dataset: dict[str, Path]) -> None:
        record = {"qid": 101, "source": "artifact", "research_text": bundle_text(first="## Something else entirely")}
        (dataset["archive"] / "101.json").write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(ValueError, match="disagrees"):
            bench.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"])


class TestPromptSelection:
    def _questions(self, dataset: dict[str, Path]) -> dict[int, bench.BenchQuestion]:
        return {
            q.question_id: q for q in bench.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"])
        }

    def test_binary_prompt_is_anchored_to_the_archived_run_date(self, dataset: dict[str, Path]) -> None:
        question = self._questions(dataset)[101]
        with bench.anchored_clock(question.forecasting_window, question.today):
            prompt = bench.render_prompt(question.question, question.arms["full"])

        assert WINDOW in prompt
        assert f"as of {RUN_DAY}" in prompt
        assert datetime.now(UTC).strftime("%Y-%m-%d") not in prompt
        assert V1_BODY in prompt
        assert V2_BODY in prompt
        assert "posterior_prob" in prompt

    def test_the_clock_is_restored_after_the_build(self, dataset: dict[str, Path]) -> None:
        original_window, original_today = prompts_module._forecasting_window_str, prompts_module._today_str
        question = self._questions(dataset)[101]
        with bench.anchored_clock(question.forecasting_window, question.today):
            assert prompts_module._today_str() == RUN_DAY
        assert prompts_module._forecasting_window_str is original_window
        assert prompts_module._today_str is original_today

    def test_multiple_choice_and_numeric_pick_their_own_templates(self, dataset: dict[str, Path]) -> None:
        questions = self._questions(dataset)
        with bench.anchored_clock(WINDOW, RUN_DAY):
            mc_prompt = bench.render_prompt(questions[103].question, questions[103].arms["minus_v1"])
            numeric_prompt = bench.render_prompt(questions[104].question, questions[104].arms["minus_both"])
            discrete_prompt = bench.render_prompt(questions[105].question, questions[105].arms["full"])

        assert "option_probs" in mc_prompt
        assert all(option in mc_prompt for option in OPTIONS)
        assert V1_BODY not in mc_prompt
        assert V2_BODY in mc_prompt
        assert "declared_percentiles" in numeric_prompt
        assert "The upper bound is open" in numeric_prompt
        assert "Cases" in numeric_prompt
        assert V1_BODY not in numeric_prompt
        assert V2_BODY not in numeric_prompt
        assert "declared_percentiles" in discrete_prompt

    def test_the_plan_has_one_prompt_per_question_arm_replicate(self, dataset: dict[str, Path]) -> None:
        questions = bench.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"])
        items = bench.build_plan(questions, ["full", "minus_v1"], 2, model="fake/model")

        assert len(items) == 5 * 2 * 2
        first, second = (item for item in items if item.question.question_id == 101 and item.arm == "full")
        assert first.prompt.endswith(bench.replicate_nonce(1))
        assert second.prompt.endswith(bench.replicate_nonce(2))
        assert first.prompt.removesuffix(bench.replicate_nonce(1)) == second.prompt.removesuffix(
            bench.replicate_nonce(2)
        )
        assert all(item.prompt_tokens > 0 for item in items)
        full_tokens = next(i.prompt_tokens for i in items if i.question.question_id == 101 and i.arm == "full")
        minus_tokens = next(i.prompt_tokens for i in items if i.question.question_id == 101 and i.arm == "minus_v1")
        assert full_tokens > minus_tokens


class TestScoringPlumbing:
    def _plan(self, dataset: dict[str, Path], arms: list[str], seeds: int = 1) -> list[plan.PlanItem]:
        questions = bench.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"])
        return bench.build_plan(questions, arms, seeds, model="fake/model")

    def test_canned_replies_score_through_the_ladder_and_the_cdf_build(self, dataset: dict[str, Path]) -> None:
        rows = _run(self._plan(dataset, ["full", "minus_v1"]), FakeModel())

        assert {row.status for row in rows} == {bench.STATUS_SCORED}
        by_key = {(row.question_id, row.arm): row for row in rows}
        # The block rung read the value; the binary clamp (0.22 is inside it) leaves the raw value published.
        yes = by_key[(101, "full")]
        assert yes.score == pytest.approx(binary_log_score(0.22, True))
        assert yes.rung == "block"
        assert yes.block_present is True
        assert yes.forecast == 0.22
        assert by_key[(102, "full")].score == pytest.approx(binary_log_score(0.22, False))
        mc = by_key[(103, "full")]
        assert mc.forecast == pytest.approx([0.1, 0.5, 0.3, 0.1])
        assert mc.score == pytest.approx(mc_log_score([0.1, 0.5, 0.3, 0.1], OPTIONS.index("Arizona")))
        numeric = by_key[(104, "full")]
        assert numeric.score is not None
        assert math.isfinite(numeric.score)
        assert numeric.forecast is not None
        assert len(numeric.forecast) == len(STANDARD_PERCENTILES)
        discrete_score = by_key[(105, "full")].score
        assert discrete_score is not None
        assert math.isfinite(discrete_score)
        # The same reply in two arms scores identically: the arm text changes the prompt, never the scorer.
        assert by_key[(104, "minus_v1")].score == numeric.score
        assert all((row.prompt_tokens, row.completion_tokens, row.reasoning_tokens) == (1000, 300, 200) for row in rows)
        assert all(row.cost_usd == 0.001 and row.rationale for row in rows)

    def test_a_reply_without_a_block_is_an_extraction_failure_row(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _no_salvage(*_args: Any, **_kwargs: Any) -> Any:
            await asyncio.sleep(0)
            raise RuntimeError("no parser LLM in tests")

        monkeypatch.setattr(value_extraction, "parse_structured", _no_salvage)
        items = [item for item in self._plan(dataset, ["full"]) if item.question.question_id == 101]
        rows = _run(items, FakeModel({BINARY_YES["title"]: NO_BLOCK_REPLY}))

        assert [row.status for row in rows] == [bench.STATUS_EXTRACTION_FAILED]
        assert rows[0].score is None
        assert rows[0].error
        assert "no parser LLM" in rows[0].error
        assert rows[0].rationale == NO_BLOCK_REPLY
        assert rows[0].cost_usd == 0.001

    def test_a_scale_error_trips_the_unit_mismatch_guard(self, dataset: dict[str, Path]) -> None:
        items = [item for item in self._plan(dataset, ["full"]) if item.question.question_id == 104]
        tiny = numeric_reply([0.0075 + 0.0005 * i for i in range(len(STANDARD_PERCENTILES))])
        rows = _run(items, FakeModel({NUMERIC["title"]: tiny}))

        assert [row.status for row in rows] == [bench.STATUS_UNIT_MISMATCH]
        assert rows[0].error
        assert "unit mismatch" in rows[0].error

    def test_a_provider_error_is_a_row_not_a_crash(self, dataset: dict[str, Path]) -> None:
        async def _timeout(_prompt: str) -> run.ModelReply:
            await asyncio.sleep(0)
            raise litellm.exceptions.Timeout(message="slow upstream", model="m", llm_provider="openrouter")

        items = [item for item in self._plan(dataset, ["full"]) if item.question.question_id == 101]
        rows = _run(items, _timeout)

        assert [row.status for row in rows] == [bench.STATUS_API_ERROR]
        assert rows[0].error
        assert rows[0].error.startswith("Timeout")
        assert rows[0].cost_usd is None

    def test_peer_scale_factor_per_type(self) -> None:
        assert bench.peer_scale_factor("binary", None) == pytest.approx(math.log(2))
        assert bench.peer_scale_factor("multiple_choice", 4) == pytest.approx(math.log(4))
        assert bench.peer_scale_factor("numeric", None) == 1.0
        assert bench.peer_scale_factor("discrete", None) == 1.0
        with pytest.raises(ValueError, match="option count"):
            bench.peer_scale_factor("multiple_choice", None)

    def test_reply_from_response_reads_openrouter_cost_and_reasoning_tokens(self) -> None:
        response = litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hello"}}],
            usage={
                "prompt_tokens": 12,
                "completion_tokens": 9,
                "total_tokens": 21,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cost": 0.00042,
            },
        )
        reply = bench.reply_from_response(response)

        assert reply == run.ModelReply(
            text="hello", prompt_tokens=12, completion_tokens=9, reasoning_tokens=4, cost_usd=0.00042
        )


class TestSpendGates:
    def test_the_meter_stops_the_run_once_measured_spend_reaches_the_cap(self, dataset: dict[str, Path]) -> None:
        questions = bench.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"])
        items = bench.build_plan(questions, ["full"], 2, model="fake/model")
        model = FakeModel(cost_usd=3.0)
        rows = _run(items, model, cap_usd=5.0, concurrency=1)

        statuses = [row.status for row in rows]
        # Two calls ($6) pass the $5 cap; every later call is skipped before it is made.
        assert statuses.count(bench.STATUS_SCORED) == 2
        assert statuses.count(bench.STATUS_SKIPPED_SPEND_CAP) == len(items) - 2
        assert len(model.prompts) == 2

    def test_the_meter_counts_the_parser_ledger_too(self) -> None:
        meter = run.SpendMeter(cap_usd=1.0, extra_usd=lambda: 0.75)
        meter.add(0.2)
        assert not meter.exhausted
        meter.add(0.1)
        assert meter.exhausted
        assert meter.measured_usd == pytest.approx(1.05)

    def test_a_bare_invocation_prints_the_estimate_and_refuses(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "run_bench", lambda *_a, **_k: pytest.fail("the refusal path started a run"))

        assert cli.main(_dataset_args(dataset)) == 2
        out = capsys.readouterr().out
        assert "Estimated cost of this run" in out
        assert "--i-accept-spend" in out
        assert "calls: 60;" in out  # 5 questions x 4 arms x 3 replicates
        assert not dataset["out"].exists()

    def test_dry_run_prints_the_plan_and_calls_nothing(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "run_bench", lambda *_a, **_k: pytest.fail("a dry run started a run"))

        assert cli.main(["--dry-run", "--seeds", "2", "--arms", "full", "minus_v1", *_dataset_args(dataset)]) == 0
        out = capsys.readouterr().out
        assert "Plan: 5 resolved questions" in out
        assert "calls: 20;" in out
        assert "Estimated cost of this run" in out
        assert "replicates of the full 5 x 2 set" in out

    def test_an_estimate_over_the_cap_aborts_before_the_first_call(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "run_bench", lambda *_a, **_k: pytest.fail("an over-cap estimate started a run"))

        code = cli.main(["--i-accept-spend", "--max-spend-usd", "0.000001", *_dataset_args(dataset)])

        assert code == 2
        assert "Aborting before the first call" in capsys.readouterr().out
        assert not dataset["out"].exists()

    def test_the_estimate_arithmetic(self, dataset: dict[str, Path]) -> None:
        questions = bench.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"])
        items = bench.build_plan(questions, list(bench.ARMS), 3, model="fake/model")
        estimate = bench.estimate_spend(items, output_tokens=800, price_in=0.10, price_out=0.20)

        assert (estimate.n_questions, estimate.n_arms, estimate.n_seeds, estimate.n_calls) == (5, 4, 3, 60)
        assert estimate.prompt_tokens_total == sum(item.prompt_tokens for item in items)
        assert estimate.prompt_usd == pytest.approx(estimate.prompt_tokens_total * 0.10 / 1e6)
        assert estimate.completion_usd == pytest.approx(60 * 800 * 0.20 / 1e6)
        assert estimate.total_usd == pytest.approx(estimate.prompt_usd + estimate.completion_usd)
        assert estimate.calls_under(10.0) == int(10.0 / estimate.usd_per_call)
        assert estimate.seeds_under(10.0) == estimate.calls_under(10.0) // 20

    def test_arms_must_include_full(self, dataset: dict[str, Path]) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args(["--dry-run", "--arms", "minus_v1", *_dataset_args(dataset)])


class TestPersonalKeyOnly:
    def test_the_donated_key_is_unreachable_on_the_paid_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(run, "load_environment", lambda: None)
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "sk-donated")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-personal")
        monkeypatch.delenv("DONATED_OPENROUTER_KEY_ENABLED", raising=False)

        assert bench.personal_key_only_environment() == "sk-personal"
        assert "OAI_ANTH_OPENROUTER_KEY" not in __import__("os").environ
        assert __import__("os").environ["DONATED_OPENROUTER_KEY_ENABLED"] == "false"

    def test_no_personal_key_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(run, "load_environment", lambda: None)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            bench.personal_key_only_environment()


def _row(qid: int, qtype: str, arm: str, seed: int, score: float, *, n_options: int | None = None) -> bench.CallRow:
    return bench.CallRow(
        question_id=qid, qtype=qtype, n_options=n_options, arm=arm, seed=seed, status=bench.STATUS_SCORED, score=score
    )


SUMMARIES = [
    {
        "question_id": 1,
        "qtype": "binary",
        "title": "b",
        "resolution": "True",
        "n_options": None,
        "published_score": 40.0,
        "spot_peer_score": 5.0,
    },
    {
        "question_id": 2,
        "qtype": "multiple_choice",
        "title": "m",
        "resolution": "A",
        "n_options": 3,
        "published_score": 10.0,
        "spot_peer_score": 1.0,
    },
    {
        "question_id": 3,
        "qtype": "numeric",
        "title": "n",
        "resolution": "9.0",
        "n_options": None,
        "published_score": 20.0,
        "spot_peer_score": 2.0,
    },
]


class TestAggregation:
    def test_paired_deltas_average_replicates_then_pair_by_question(self) -> None:
        rows = [
            _row(1, "binary", "full", 1, 60.0),
            _row(1, "binary", "full", 2, 40.0),
            _row(1, "binary", "minus_v1", 1, 30.0),
            _row(2, "multiple_choice", "full", 1, 20.0, n_options=3),
            _row(2, "multiple_choice", "minus_v1", 1, 30.0, n_options=3),
            _row(3, "numeric", "full", 1, 15.0),
            _row(3, "numeric", "minus_v1", 1, 5.0),
            bench.CallRow(
                question_id=3,
                qtype="numeric",
                n_options=None,
                arm="minus_v1",
                seed=2,
                status=bench.STATUS_EXTRACTION_FAILED,
            ),
        ]
        results = bench.aggregate(rows, SUMMARIES, arms=["full", "minus_v1"], bootstrap_seed=7)

        deltas = results["paired_deltas"]["minus_v1"]
        assert deltas["binary"]["mean_delta"] == pytest.approx(50.0 - 30.0)
        assert deltas["multiple_choice"]["mean_delta"] == pytest.approx(-10.0)
        assert deltas["numeric"]["mean_delta"] == pytest.approx(10.0)
        peer = deltas["all_peer_points"]
        assert peer["n"] == 3
        assert peer["mean_delta"] == pytest.approx((20.0 * math.log(2) - 10.0 * math.log(3) + 10.0) / 3)
        assert (peer["n_full_better"], peer["n_arm_better"]) == (2, 1)
        assert 0.0 <= peer["sign_test_p"] <= 1.0
        # Too few pairs for a bootstrap interval, so the CI collapses onto the mean rather than fabricating one.
        assert peer["ci95_low"] == peer["ci95_high"] == peer["mean_delta"]

        summary = results["arm_summary"]
        assert summary["full"]["all"] == {"n": 3, "mean": pytest.approx((50 + 20 + 15) / 3), "median": 20.0}
        assert summary["published"]["binary"]["mean"] == 40.0
        assert results["call_status_by_arm"]["minus_v1"] == {bench.STATUS_SCORED: 3, bench.STATUS_EXTRACTION_FAILED: 1}
        per_question = {row["question_id"]: row for row in results["per_question"]}
        assert per_question[1]["deltas"] == {"minus_v1": pytest.approx(20.0)}
        assert per_question[1]["arm_scores"] == {"full": 50.0, "minus_v1": 30.0}

    def test_the_markdown_leads_with_the_deltas(self) -> None:
        rows = [_row(1, "binary", "full", 1, 60.0), _row(1, "binary", "minus_v1", 1, 30.0)]
        results = bench.aggregate(rows, SUMMARIES[:1], arms=["full", "minus_v1"], bootstrap_seed=0)
        text = bench.render_markdown(results, model="fake/model", seeds=1)

        assert text.startswith("# Section-strip bench")
        assert "## Paired deltas, full minus arm" in text
        assert "| minus_v1 | binary | 1 | +30.00 |" in text
        assert "| published | binary | 1 | 40.00 | 40.00 |" in text

    def test_rescore_rebuilds_the_results_from_the_run_dir(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = tmp_path / "section_strip_bench_x"
        run_dir.mkdir()
        run_meta = {
            "model": "fake/model",
            "seeds": 1,
            "arms": ["full", "minus_v1"],
            "bootstrap_seed": 0,
            "questions": SUMMARIES[:1],
        }
        (run_dir / "run.json").write_text(json.dumps(run_meta), encoding="utf-8")
        rows = [_row(1, "binary", "full", 1, 60.0), _row(1, "binary", "minus_v1", 1, 30.0)]
        (run_dir / "calls.jsonl").write_text("".join(json.dumps(r.__dict__) + "\n" for r in rows), encoding="utf-8")

        assert cli.main(["--rescore", str(run_dir)]) == 0

        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        assert results["paired_deltas"]["minus_v1"]["binary"]["mean_delta"] == 30.0
        assert (run_dir / "SUMMARY.md").exists()
        assert "Rescored 2 calls" in capsys.readouterr().out


class TestPaidPathWithFakes:
    def test_the_whole_run_writes_every_artifact(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = FakeModel()

        async def _drained() -> None:
            await asyncio.sleep(0)

        monkeypatch.setattr(cli, "personal_key_only_environment", lambda: "sk-personal")
        monkeypatch.setattr(cli, "install_role_spend_tracker", lambda: None)
        monkeypatch.setattr(cli, "drain_litellm_callbacks", _drained)
        monkeypatch.setattr(cli, "parser_ledger_usd", lambda: 0.0)
        monkeypatch.setattr(cli, "build_model_call", lambda *_a, **_k: model)
        monkeypatch.setattr(cli, "build_parser_llm", lambda _m: parser_stub())

        code = cli.main(["--i-accept-spend", "--seeds", "1", "--concurrency", "3", *_dataset_args(dataset)])

        assert code == 0
        (run_dir,) = list(dataset["out"].iterdir())
        rows = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 5 * 4 == len(model.prompts)
        assert {row["status"] for row in rows} == {bench.STATUS_SCORED}
        run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run_meta["measured_usd"] == pytest.approx(20 * 0.001)
        assert run_meta["cap_hit"] is False
        assert [q["question_id"] for q in run_meta["questions"]] == RESOLVED_IDS
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        assert set(results["paired_deltas"]) == set(bench.STRIPPED_ARMS)
        assert results["spend"]["forecaster_usd"] == pytest.approx(0.02)
        assert (run_dir / "SUMMARY.md").read_text(encoding="utf-8").startswith("# Section-strip bench")
        assert (run_dir / "bench.log").exists()
