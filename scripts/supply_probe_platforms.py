"""The two question platforms the supply probe walks, and how each answers "did the bot forecast this".

``with_cp=true`` rides every list GET on both platforms. It is what puts the token's ``my_forecasts``
on a posts list page (so the forfeit sweep's per-post detail GETs are only a fallback), and on
Mantic's Crucible (competitions.mantic.com, a Metaculus fork) it is also what puts the public
spot-time snapshot of a RESOLVED question there. Metaculus reads are authenticated, so its token is
required; Mantic's reads are public, so its token is optional and only lets a closed-but-unresolved
question classify. Everything else the probe does (paging, backlog, forfeits, the per-release-hour
table, rendering) is shared and lives in ``scripts/supply_probe.py``; the design notes are in
``docs/supply_probe.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from forecasting_tools import MetaculusApi
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot.constants import (
    FALL_CUP_SLUG,
    MANTIC_API_BASE_URL,
    MANTIC_BOT_USER_ID,
    MANTIC_TOKEN_ENV,
    MANTIC_TOURNAMENT_ID,
    METACULUS_CUP_ID,
    METACULUS_TOKEN_ENV,
    PLATFORM_MANTIC,
    PLATFORM_METACULUS,
    TOURNAMENT_ID,
)

# Deduplicated so re-pointing METACULUS_CUP_ID at the dated fall slug probes it once (docs/supply_probe.md).
DEFAULT_SLUGS: tuple[str, ...] = tuple(
    dict.fromkeys([TOURNAMENT_ID, METACULUS_CUP_ID, FALL_CUP_SLUG, MetaculusApi.CURRENT_MINIBENCH_ID])
)

# What a forecast-state read can answer. UNKNOWN is a measurement failure, not a forfeit.
FORECAST_PRESENT = "forecast"
FORECAST_ABSENT = "no_forecast"
FORECAST_UNKNOWN = "unknown"

# Puts my_forecasts (under a token) and Mantic's public spot-time snapshot on the list page; free on both.
WITH_CP_LIST_PARAMS: Mapping[str, str] = {"with_cp": "true"}


@dataclass(frozen=True)
class PlatformProbe:
    """The seams where the two platforms' APIs differ; paging, backlog and forfeit logic are shared.

    ``base_url`` is the API root the identity preflight vets, and ``posts_url`` hangs off it, so the
    token can only go to the vetted host. ``list_params`` ride every list GET, token or not.
    ``sweep_needs_detail_gets`` is False where the list page already answers, so no per-post GET is
    ever issued there. ``bot_user_id`` is the id the public snapshot is read against, None where
    classification is token-only. The three prose fields are the report's header note on how
    forecast state was read, its line for a slug where every state is unknown, and its check for a
    slug where nothing carries a bot forecast.
    """

    name: str
    base_url: str
    token_env: str
    token_required: bool
    default_slugs: tuple[str, ...]
    forecast_state: Callable[[Mapping[str, Any]], str]
    list_params: Mapping[str, str]
    sweep_needs_detail_gets: bool
    bot_user_id: int | None
    classification_note: str
    all_unknown_hint: str
    identity_hint: str

    @property
    def posts_url(self) -> str:
        return f"{self.base_url}/posts/"


def bot_forecast_state(question: Mapping[str, Any]) -> str:
    """Whether the token's own user forecast THIS question, per its ``my_forecasts`` block.

    Three answers, because "the payload says we did not forecast it" and "the payload does
    not say" are different facts and only the first is a forfeit. A question dict from a list
    page read without ``with_cp=true`` (or without a token) carries no ``my_forecasts`` at all,
    so it answers UNKNOWN until the sweep enriches it from a per-post detail GET.

    ``history`` is the authoritative emptiness test (the operator's own read of the API), but
    a non-empty ``latest`` also counts as present: this must never call a real forecast a
    forfeit, and the scoring collector keys on ``latest``.
    """
    if "my_forecasts" not in question:
        return FORECAST_UNKNOWN
    my_forecasts = question.get("my_forecasts")
    if not isinstance(my_forecasts, Mapping):
        # Present but null/scalar: the block carried no answer, so neither do we.
        return FORECAST_UNKNOWN
    if my_forecasts.get("history") or my_forecasts.get("latest"):
        return FORECAST_PRESENT
    return FORECAST_ABSENT


def _public_snapshot_authors(question: Mapping[str, Any]) -> list[Any] | None:
    """Author ids in Mantic's public spot-time snapshot, or None before it exists.

    Three wire shapes under ``with_cp=true``, all verified on the 2026-09-08 corpus: an open
    question has ``score_data: {}``, a closed-but-unresolved one has ``disagreement_forecasts:
    null``, and a resolved one carries the list. An empty list is the resolved-but-not-yet-scored
    state and also answers None, since it is not evidence of anyone's absence. Without the flag
    every question on the list page has ``score_data: {}`` whatever its status (the two recorded
    Series 1 pages under ``tests/data/`` are the same posts read both ways), so nothing classifies.
    """
    aggregations = question.get("aggregations") or {}
    score_data = (aggregations.get("recency_weighted") or {}).get("score_data") or {}
    snapshot = score_data.get("disagreement_forecasts") or {}
    forecasts = snapshot.get("forecasts")
    if not forecasts:
        return None
    return [entry.get("author_id") for entry in forecasts]


def mantic_forecast_state(question: Mapping[str, Any], *, bot_user_id: int) -> str:
    """Mantic's read: the token's own block when it answers, else the public spot-time snapshot.

    The snapshot (``aggregations.recency_weighted.score_data.disagreement_forecasts.forecasts``,
    one entry per competitor with an ``author_id``) is the platform's own spot-time record, which
    is what is scored, so a non-empty one that does not name ``bot_user_id`` is a forfeit. It is
    also why the token is optional on Mantic: every eventually-resolved question becomes
    measurable without a secret, as long as ``with_cp=true`` is on the list GET.

    Its caveat, stated in the report header rather than modelled: the snapshot names one competitor
    fewer than the post's ``nr_forecasters`` on nearly every resolved question (507 of 524 in the
    2026-09-08 corpus) for a cause not established, so it can read ``no_forecast`` for an account
    that did forecast, and the PRESENT branch has never been exercised live. Numbers and the
    provisional-reading rule: ``docs/supply_probe.md`` "The Mantic mode".
    """
    state = bot_forecast_state(question)
    if state != FORECAST_UNKNOWN:
        return state
    authors = _public_snapshot_authors(question)
    if authors is None:
        return FORECAST_UNKNOWN
    return FORECAST_PRESENT if bot_user_id in authors else FORECAST_ABSENT


METACULUS_PROBE = PlatformProbe(
    name=PLATFORM_METACULUS,
    # Read off the client so the token goes to the host the preflight vets (docs/supply_probe.md "API facts").
    base_url=MetaculusClient().base_url,
    token_env=METACULUS_TOKEN_ENV,
    token_required=True,
    default_slugs=DEFAULT_SLUGS,
    forecast_state=bot_forecast_state,
    list_params=WITH_CP_LIST_PARAMS,
    sweep_needs_detail_gets=True,
    bot_user_id=None,
    classification_note=(
        "Forecast state: the token's own my_forecasts block, fetched per post only where the list page lacks it."
    ),
    all_unknown_hint="my_forecasts was unreadable on every one — run without --no-forfeits to resolve it",
    identity_hint=f"Check that {METACULUS_TOKEN_ENV} is the bot's own token",
)
MANTIC_PROBE = PlatformProbe(
    name=PLATFORM_MANTIC,
    base_url=MANTIC_API_BASE_URL,
    token_env=MANTIC_TOKEN_ENV,
    token_required=False,
    default_slugs=(MANTIC_TOURNAMENT_ID,),
    forecast_state=partial(mantic_forecast_state, bot_user_id=MANTIC_BOT_USER_ID),
    list_params=WITH_CP_LIST_PARAMS,
    sweep_needs_detail_gets=False,
    bot_user_id=MANTIC_BOT_USER_ID,
    classification_note=(
        f"Forecast state: my_forecasts under {MANTIC_TOKEN_ENV} when set, else the public spot-time snapshot of a "
        "resolved question; a closed-but-unresolved question with neither reads unknown. The snapshot names one "
        "competitor fewer than nr_forecasters on nearly every resolved question (cause not established), so a "
        "no_forecast read from it is provisional until the first question the bot forecast resolves and its "
        "snapshot names the bot."
    ),
    all_unknown_hint=(
        f"no question could be classified — set {MANTIC_TOKEN_ENV} so closed-but-unresolved questions carry "
        "my_forecasts; resolved ones classify from the public snapshot without it"
    ),
    identity_hint=(
        f"Check that {MANTIC_TOKEN_ENV} is the bot's own token and MANTIC_BOT_USER_ID ({MANTIC_BOT_USER_ID}) "
        "its user id"
    ),
)
PLATFORM_PROBES: dict[str, PlatformProbe] = {probe.name: probe for probe in (METACULUS_PROBE, MANTIC_PROBE)}

# The Metaculus posts endpoint; the forfeit sweep's per-post detail GETs hang off it too.
POSTS_URL = METACULUS_PROBE.posts_url
