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

Three robustness rules from the 2026-09-08 readiness review sit beside them:

- The tournament fetch asks for a CEILING of questions (``MANTIC_FETCH_QUESTION_CEILING``) rather
  than the framework's default single page of 100, because Mantic's paginator advertises a ``next``
  link past the last page (offset 600 of a 520-post tournament still carries one), so only walking
  offsets until an empty page can prove there is nothing more.
- A post the framework cannot parse used to be forfeited silently: ``_get_questions_from_api`` logs
  one warning per failed post and continues, nothing counts it, and after the 90-day log expiry it
  is gone. The parse override now counts the drop (``get_post_drop_count``, read into cli's
  alertable arithmetic so the run reddens) and emits one ``MANTIC_POST_DROPPED`` line before
  re-raising. Fail-fast is kept: nothing here swallows the error.
- :func:`preflight_mantic_tournaments` makes two authenticated GETs before any spend, neither
  retried. The tournament list logs ``MANTIC_TOURNAMENTS`` naming every ongoing bots-only tournament,
  at WARNING when one is not the configured slug, so a Series 2 slug is named in the log the run it
  appears. The configured tournament's own route (``/projects/tournaments/<slug>/``) is then read
  and the run refuses to proceed unless the token's ``user_permission`` there allows forecasting (a
  view-only token reads fine and would fail only at the publish POST, after the ensemble had been
  paid for, every hour). The detail route rather than the list row because the list omits an
  ``unlisted`` project, the state a new season sits in before its first question opens, so absence
  from the list proves nothing, while a slug no tournament has 404s there. Every failure shape of
  either GET is one ``ApiIdentityError`` for the operator to grep.

Every parsed question emits one ``MANTIC_QUESTION`` line so the fields Mantic added and the framework
does not model (``multi_resolution``, ``date_granularity``, ``precision``), plus the type as it
arrived on the wire, outlive the 90-day GitHub Actions log expiry for residual analysis. The specs
live in ``scripts/telemetry/markers.py``; the verbatim example lines in
``tests/test_telemetry_markers.py``.

Conditional posts are not modelled: Mantic publishes none, and the bot's type guard drops
``ConditionalQuestion`` anyway, so one reaching this client is counted and logged as a dropped post
and surfaces as the framework's per-post "Error processing post" warning rather than as a forecast.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import requests
from forecasting_tools.data_models.questions import DateQuestion, MetaculusQuestion, NumericQuestion
from forecasting_tools.helpers.metaculus_client import ApiFilter, GroupQuestionMode, MetaculusClient

from metaculus_bot.api_preflight import BODY_PREVIEW_CHARS, ApiIdentityError
from metaculus_bot.constants import (
    MANTIC_API_BASE_URL,
    MANTIC_FETCH_QUESTION_CEILING,
    MANTIC_SITE_URL,
    MANTIC_TOKEN_ENV,
)

logger = logging.getLogger(__name__)

_QUANTITATIVE_WIRE_TYPE = "quantitative"
_DISCRETE_TYPE = "discrete"
# The registry's None sentinel (scripts/telemetry/markers._NONE_SENTINELS): harvests as None.
_ABSENT = "n/a"
# The Metaculus backend's ObjectPermission roles that may forecast (projects/permissions.py); viewer and null only read.
_FORECASTING_PERMISSIONS: frozenset[str] = frozenset({"forecaster", "curator", "admin", "creator"})
_BOTS_ONLY_LEADERBOARD = "bots_only"
# Live, both tournament routes answer a mistyped or revoked token with a 403, not the 401 one expects.
_AUTH_REJECTED_HINT = "A 401 or 403 ('Invalid token.') means MANTIC_TOKEN is not accepted."

_post_drop_count = 0


def get_post_drop_count() -> int:
    """Posts this process failed to parse into questions: the ``MANTIC_POST_DROPPED`` count."""
    return _post_drop_count


def reset_post_drop_count() -> None:
    global _post_drop_count  # noqa: PLW0603  # module-global run counter is the design (AGENTS.md)
    _post_drop_count = 0


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
        # The ceiling makes the framework walk offsets until an EMPTY page; short of it is the normal result.
        questions = asyncio.run(
            self.get_questions_matching_filter(
                api_filter,
                num_questions=MANTIC_FETCH_QUESTION_CEILING,
                error_if_question_target_missed=False,
            )
        )
        logger.info("Retrieved %d questions from Mantic tournament %s", len(questions), tournament_id)
        return questions

    def _post_json_to_questions_while_handling_groups(
        self, post_json_from_api: dict, group_question_mode: GroupQuestionMode
    ) -> list[MetaculusQuestion]:
        try:
            post_json, wire_types = _normalize_quantitative_types(post_json_from_api)
            questions = super()._post_json_to_questions_while_handling_groups(post_json, group_question_mode)
            for question in questions:
                question.page_url = f"{MANTIC_SITE_URL}/questions/{question.id_of_post}/"
                _log_mantic_question(question, wire_type=wire_types.get(question.id_of_question))
        except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # counted and logged, then re-raised into the framework's per-post loop
            _count_dropped_post(post_json_from_api, exc)
            raise
        return questions

    def list_tournaments(self) -> list[dict]:
        """One authenticated GET of ``/projects/tournaments/``; ``ApiIdentityError`` on anything but a 200 list.

        Authenticated on purpose: the platform's token authentication rejects a revoked or mistyped
        token here, which the unauthenticated identity preflight cannot see, and ``user_permission``
        in the payload is the caller's own. Not retried, like the identity preflight: a transient
        failure stops the run before any spend and the next cron retries.
        """
        url = f"{self.base_url}/projects/tournaments/"
        tournaments, body_preview = self._get_json(url, what="tournament list", status_hint=_AUTH_REJECTED_HINT)
        if not isinstance(tournaments, list):
            raise ApiIdentityError(
                f"Mantic tournament list GET {url!r} answered 200 but not with a JSON list "
                f"(first {BODY_PREVIEW_CHARS} chars: {body_preview!r}); stopping before any spend."
            )
        return tournaments

    def get_tournament(self, tournament_id: str) -> dict:
        """One authenticated GET of ``/projects/tournaments/<slug>/``; ``ApiIdentityError`` unless a 200 object.

        The route that is authoritative for one project: slug and numeric id both resolve, a slug no
        tournament has 404s, and the payload carries the caller's own ``user_permission``. Unretried,
        like :meth:`list_tournaments`.
        """
        url = f"{self.base_url}/projects/tournaments/{tournament_id}/"
        tournament, body_preview = self._get_json(
            url,
            what=f"tournament {tournament_id!r}",
            status_hint=(
                f"A 404 means no tournament has the slug {tournament_id!r}: re-point MANTIC_TOURNAMENT_ID in "
                f"constants.py. {_AUTH_REJECTED_HINT}"
            ),
        )
        if not isinstance(tournament, dict):
            raise ApiIdentityError(
                f"Mantic tournament {tournament_id!r} GET {url!r} answered 200 but not with a JSON object "
                f"(first {BODY_PREVIEW_CHARS} chars: {body_preview!r}); stopping before any spend."
            )
        return tournament

    def _get_json(self, url: str, *, what: str, status_hint: str) -> tuple[Any, str]:
        """One authenticated, unretried GET decoded as JSON, plus the body preview for diagnostics.

        Every failure shape is ``ApiIdentityError``: a transport failure (DNS, TLS, connect,
        timeout), a non-200 status, and a 200 whose body is not JSON (the captive-portal shape).
        The run stops before any spend either way; wrapping makes the stop one greppable exception
        carrying the URL, status and body preview instead of a requests traceback.
        """
        try:
            response = requests.get(url, headers=self._get_auth_headers()["headers"], timeout=self.timeout)
        except requests.RequestException as exc:
            raise ApiIdentityError(
                f"Mantic {what} GET {url!r} failed before any response ({type(exc).__name__}: {exc}); the token's "
                "forecast permission cannot be confirmed, so the run stops before any spend."
            ) from exc
        body_preview = response.text[:BODY_PREVIEW_CHARS]
        if response.status_code != 200:
            raise ApiIdentityError(
                f"Mantic {what} GET {url!r} answered status={response.status_code} "
                f"(first {BODY_PREVIEW_CHARS} chars: {body_preview!r}); the token's forecast permission cannot "
                f"be confirmed, so the run stops before any spend. {status_hint}"
            )
        try:
            return response.json(), body_preview
        except requests.exceptions.JSONDecodeError as exc:
            raise ApiIdentityError(
                f"Mantic {what} GET {url!r} answered 200 but not with JSON "
                f"(first {BODY_PREVIEW_CHARS} chars: {body_preview!r}); stopping before any spend."
            ) from exc


def build_mantic_client() -> ManticClient:
    """Build the client on the operator's personal Mantic bot token; fail shut before any request."""
    token = os.environ.get(MANTIC_TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"{MANTIC_TOKEN_ENV} is not set; a Mantic run needs the operator's personal Mantic bot token"
        )
    return ManticClient(token=token)


def _normalize_quantitative_types(post_json: dict) -> tuple[dict, dict[int | None, str | None]]:
    """Return a copy of the post with ``quantitative`` question types rewritten to ``discrete``.

    Also returns each question's type as it arrived on the wire, keyed by question id, for the
    marker. Only the dicts that get rewritten are copied, so the caller's post JSON is left as it
    was. Mirrors the framework's dispatch: a ``group_of_questions`` post carries its questions
    under the group, anything else under ``question``; a post with neither key passes through
    untouched for the framework's own parser to reject. The telemetry-only reads (``id``, ``type``)
    use ``.get``, so the marker can never be what drops a question.
    """
    post = dict(post_json)
    if "group_of_questions" in post:
        group = dict(post["group_of_questions"])
        question_jsons: list[dict] = group["questions"]
        group["questions"] = [_as_discrete_if_quantitative(question) for question in question_jsons]
        post["group_of_questions"] = group
    elif "question" in post:
        question_jsons = [post["question"]]
        post["question"] = _as_discrete_if_quantitative(post["question"])
    else:
        question_jsons = []
    return post, {question.get("id"): question.get("type") for question in question_jsons}


def _as_discrete_if_quantitative(question_json: dict) -> dict:
    if question_json.get("type") != _QUANTITATIVE_WIRE_TYPE:
        return question_json
    return {**question_json, "type": _DISCRETE_TYPE}


def _count_dropped_post(post_json: dict, exc: BaseException) -> None:
    """Bump the process counter and emit ``MANTIC_POST_DROPPED`` for a post that failed to parse.

    The framework's ``_get_questions_from_api`` catches the re-raised error per post, logs one
    warning and continues, which forfeited the post silently on every run. The counter reaches
    cli's alertable arithmetic so the run reddens; the marker outlives the log expiry. Every read
    here is ``.get``: telemetry about a broken post must not raise on the same broken post.
    """
    global _post_drop_count  # noqa: PLW0603  # module-global run counter is the design (AGENTS.md)
    _post_drop_count += 1
    question_json = post_json.get("question") or {}
    logger.error(
        "MANTIC_POST_DROPPED: post=%s type=%s error=%s",
        _render(post_json.get("id")),
        _render(question_json.get("type")),
        type(exc).__name__,
    )


def preflight_mantic_tournaments(client: ManticClient, tournament_id: str) -> None:
    """Two authenticated GETs, two checks, before any spend.

    Series 2 discovery first, off the tournament LIST: ``MANTIC_TOURNAMENTS`` names every ongoing
    tournament, the configured slug, and the ongoing bots-only tournaments that are NOT the
    configured one, at WARNING when that last set is non-empty. A zero-question run is green, so
    without this line a Series 2 slug could open and every hourly run would keep fetching the ended
    preseason silently.

    Then the permission check, off the configured tournament's own route, which fails shut:
    ``ApiIdentityError`` unless ``/projects/tournaments/<slug>/`` answers 200 with a
    ``user_permission`` that allows forecasting. The detail route rather than the list row because
    the list omits an ``unlisted`` project, the state a new season sits in before its first question
    opens, so absence from the list is not evidence the slug is wrong, while a slug no tournament
    has 404s there. A token that authenticates but may only view reads the tournament fine and
    fails only at the publish POST, after the whole ensemble has been paid for, on every cron until
    somebody reads the red runs.
    """
    tournaments = client.list_tournaments()
    by_slug = {tournament.get("slug"): tournament for tournament in tournaments}
    ongoing = sorted(slug for slug, tournament in by_slug.items() if slug and tournament.get("is_ongoing") is True)
    new = [
        slug
        for slug in ongoing
        if slug != tournament_id and by_slug[slug].get("bot_leaderboard_status") == _BOTS_ONLY_LEADERBOARD
    ]
    logger.log(
        logging.WARNING if new else logging.INFO,
        "MANTIC_TOURNAMENTS: ongoing=%s configured=%s new=%s",
        _slugs_csv(ongoing),
        tournament_id,
        _slugs_csv(new),
    )

    permission = client.get_tournament(tournament_id).get("user_permission")
    if permission not in _FORECASTING_PERMISSIONS:
        raise ApiIdentityError(
            f"MANTIC_TOKEN holds user_permission={permission!r} on tournament {tournament_id!r}, which does not "
            f"allow forecasting (one of {sorted(_FORECASTING_PERMISSIONS)} does). Every question would be researched "
            "and forecast and then fail at the publish POST, so the run stops before any spend."
        )


def _slugs_csv(slugs: list[str]) -> str:
    """Comma-joined slugs for the ``MANTIC_TOURNAMENTS`` marker; ``none`` for an empty list."""
    return ",".join(slugs) or "none"


def _log_mantic_question(question: MetaculusQuestion, *, wire_type: str | None) -> None:
    """Emit the MANTIC_QUESTION marker for one parsed question.

    The three Mantic-only fields and the wire type are read with ``.get`` so a post that lacks one
    still parses and renders ``n/a``: telemetry must never turn into a dropped question.
    """
    question_json: dict = question.api_json["question"]
    cdf_size = question.cdf_size if isinstance(question, (NumericQuestion, DateQuestion)) else None
    logger.info(
        "MANTIC_QUESTION: post=%s question=%s type=%s cdf_size=%s multi_resolution=%s date_granularity=%s precision=%s",
        question.id_of_post,
        question.id_of_question,
        _render(wire_type),
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
