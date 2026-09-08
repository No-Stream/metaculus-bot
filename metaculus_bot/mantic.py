"""Client for Mantic's competitions platform (competitions.mantic.com).

Mantic's "Crucible" tournaments run on a fork of the open-source Metaculus platform, so the
framework's ``MetaculusClient`` is the transport (same endpoints, same ``Token`` auth, same post
JSON) and the repo's class-level fetch and publish hardening apply to this subclass unchanged: it
overrides none of ``_get_questions_from_api``, ``_post_question_prediction`` or
``post_question_comment``. Three differences, verified against the live API on 2026-09-08, need code:

1. The list filter ``forecast_type`` speaks ``quantitative`` where the framework sends
   ``numeric,discrete``, so the framework's default tournament fetch returns ZERO quantitative
   questions. The fetch here omits the parameter (``ApiFilter(allowed_types=[])``) and leaves type
   filtering to the bot's own guard in ``forecaster.forecast_questions``.
2. Series 2 questions may arrive as ``question.type == "quantitative"`` (Mantic merged numeric and
   discrete). forecasting-tools 0.2.92 raises ``ValueError`` on it and ``_get_questions_from_api``
   swallows that as a warning, so the post would be silently dropped. The type is rewritten to
   ``discrete`` before parsing; the ``scaling`` semantics (bins, ``inbound_outcome_count``,
   ``zero_point``) are identical.
3. The framework hardcodes ``https://www.metaculus.com/questions/{post_id}`` as ``page_url``.

Every parsed question emits one ``MANTIC_QUESTION`` line so the fields Mantic added and the framework
does not model (``multi_resolution``, ``date_granularity``, ``precision``), plus the type as it
arrived on the wire, outlive the 90-day GitHub Actions log expiry for residual analysis. The spec
lives in ``scripts/telemetry/markers.py``; the verbatim example lines in
``tests/test_telemetry_markers.py``.

Conditional posts are not modelled: Mantic publishes none, and the bot's type guard drops
``ConditionalQuestion`` anyway, so one reaching this client surfaces as the framework's per-post
"Error processing post" warning rather than as a forecast.
"""

from __future__ import annotations

import asyncio
import logging
import os

from forecasting_tools.data_models.questions import DateQuestion, MetaculusQuestion, NumericQuestion
from forecasting_tools.helpers.metaculus_client import ApiFilter, GroupQuestionMode, MetaculusClient

from metaculus_bot.constants import MANTIC_API_BASE_URL, MANTIC_SITE_URL, MANTIC_TOKEN_ENV

logger = logging.getLogger(__name__)

_QUANTITATIVE_WIRE_TYPE = "quantitative"
_DISCRETE_TYPE = "discrete"
# The registry's None sentinel (scripts/telemetry/markers._NONE_SENTINELS): harvests as None.
_ABSENT = "n/a"


class ManticClient(MetaculusClient):
    """``MetaculusClient`` pointed at Mantic, with the three platform differences absorbed."""

    def __init__(self, *, token: str, timeout: int = 30) -> None:
        super().__init__(base_url=MANTIC_API_BASE_URL, token=token, timeout=timeout)

    def get_all_open_questions_from_tournament(
        self,
        tournament_id: int | str,
        group_question_mode: GroupQuestionMode = "unpack_subquestions",
    ) -> list[MetaculusQuestion]:
        logger.info("Retrieving questions from Mantic tournament %s", tournament_id)
        # Empty allowed_types: no forecast_type param (Mantic's vocabulary differs) and no local type filter.
        api_filter = ApiFilter(
            allowed_tournaments=[tournament_id],
            allowed_statuses=["open"],
            allowed_types=[],
            group_question_mode=group_question_mode,
        )
        questions = asyncio.run(self.get_questions_matching_filter(api_filter))
        logger.info("Retrieved %d questions from Mantic tournament %s", len(questions), tournament_id)
        return questions

    def _post_json_to_questions_while_handling_groups(
        self, post_json_from_api: dict, group_question_mode: GroupQuestionMode
    ) -> list[MetaculusQuestion]:
        post_json, wire_types = _normalize_quantitative_types(post_json_from_api)
        questions = super()._post_json_to_questions_while_handling_groups(post_json, group_question_mode)
        for question in questions:
            question.page_url = f"{MANTIC_SITE_URL}/questions/{question.id_of_post}/"
            _log_mantic_question(question, wire_type=wire_types[question.api_json["question"]["id"]])
        return questions


def build_mantic_client() -> ManticClient:
    """Build the client on the operator's personal Mantic bot token; fail shut before any request."""
    token = os.environ.get(MANTIC_TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"{MANTIC_TOKEN_ENV} is not set; a Mantic run needs the operator's personal Mantic bot token"
        )
    return ManticClient(token=token)


def _normalize_quantitative_types(post_json: dict) -> tuple[dict, dict[int, str]]:
    """Return a copy of the post with ``quantitative`` question types rewritten to ``discrete``.

    Also returns each question's type as it arrived on the wire, keyed by question id, for the
    marker. Only the dicts that get rewritten are copied, so the caller's post JSON is left as it
    was. Mirrors the framework's dispatch: a ``group_of_questions`` post carries its questions
    under the group, anything else under ``question``.
    """
    post = dict(post_json)
    if "group_of_questions" in post:
        group = dict(post["group_of_questions"])
        question_jsons: list[dict] = group["questions"]
        group["questions"] = [_as_discrete_if_quantitative(question) for question in question_jsons]
        post["group_of_questions"] = group
    else:
        question_jsons = [post["question"]]
        post["question"] = _as_discrete_if_quantitative(post["question"])
    return post, {question["id"]: question["type"] for question in question_jsons}


def _as_discrete_if_quantitative(question_json: dict) -> dict:
    if question_json["type"] != _QUANTITATIVE_WIRE_TYPE:
        return question_json
    return {**question_json, "type": _DISCRETE_TYPE}


def _log_mantic_question(question: MetaculusQuestion, *, wire_type: str) -> None:
    """Emit the MANTIC_QUESTION marker for one parsed question.

    The three Mantic-only fields are read with ``.get`` so a post that lacks one still parses and
    renders ``n/a``: telemetry must never turn into a dropped question.
    """
    question_json: dict = question.api_json["question"]
    cdf_size = question.cdf_size if isinstance(question, (NumericQuestion, DateQuestion)) else None
    logger.info(
        "MANTIC_QUESTION: post=%s question=%s type=%s cdf_size=%s multi_resolution=%s date_granularity=%s precision=%s",
        question.id_of_post,
        question.id_of_question,
        wire_type,
        _render(cdf_size),
        _render(question_json.get("multi_resolution")),
        _render(question_json.get("date_granularity")),
        _render(question_json.get("precision")),
    )


def _render(value: object) -> str:
    """Marker field rendering: ``n/a`` for None and the empty string, lowercase booleans, else ``str``."""
    if value is None or value == "":
        return _ABSENT
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
