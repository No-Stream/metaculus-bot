"""The two question platforms the supply probe walks, and how each answers "did the bot forecast this".

Metaculus reads are authenticated and its list pages lack ``my_forecasts``, so the token is required
and the forfeit sweep issues per-post detail GETs. Mantic's Crucible (competitions.mantic.com, a
Metaculus fork) has public reads, so its token is optional: under one every list GET carries
``with_cp=true`` (which puts ``my_forecasts`` on the list page), and without one a RESOLVED question
is classified from the platform's public spot-time snapshot. Everything else the probe does
(paging, backlog, forfeits, the per-release-hour table, rendering) is shared and lives in
``scripts/supply_probe.py``; the design notes are in ``docs/supply_probe.md``.
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

# Read off the client so the token goes to the host the preflight vetted (docs/supply_probe.md "API facts").
POSTS_URL = f"{MetaculusClient().base_url}/posts/"

# Deduplicated so re-pointing METACULUS_CUP_ID at the dated fall slug probes it once (docs/supply_probe.md).
DEFAULT_SLUGS: tuple[str, ...] = tuple(
    dict.fromkeys([TOURNAMENT_ID, METACULUS_CUP_ID, FALL_CUP_SLUG, MetaculusApi.CURRENT_MINIBENCH_ID])
)

# What a forecast-state read can answer. UNKNOWN is a measurement failure, not a forfeit.
FORECAST_PRESENT = "forecast"
FORECAST_ABSENT = "no_forecast"
FORECAST_UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlatformProbe:
    """The seams where the two platforms' APIs differ; paging, backlog and forfeit logic are shared.

    ``authenticated_list_params`` ride every list GET when a token is present.
    ``sweep_needs_detail_gets`` is False where the list page already answers, so no per-post GET is
    ever issued there. ``bot_user_id`` is the id the public snapshot is read against, None where
    classification is token-only. The three prose fields are the report's header note on how
    forecast state was read, its line for a slug where every state is unknown, and its check for a
    slug where nothing carries a bot forecast.
    """

    name: str
    posts_url: str
    token_env: str
    token_required: bool
    default_slugs: tuple[str, ...]
    forecast_state: Callable[[Mapping[str, Any]], str]
    authenticated_list_params: Mapping[str, str]
    sweep_needs_detail_gets: bool
    bot_user_id: int | None
    classification_note: str
    all_unknown_hint: str
    identity_hint: str


def bot_forecast_state(question: Mapping[str, Any]) -> str:
    """Whether the token's own user forecast THIS question, per its ``my_forecasts`` block.

    Three answers, because "the payload says we did not forecast it" and "the payload does
    not say" are different facts and only the first is a forfeit. A list-page question dict
    carries no ``my_forecasts`` at all, so it answers UNKNOWN until the sweep enriches it
    from a per-post detail GET.

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

    Three wire shapes, all verified on the 2026-09-08 corpus: an open question has
    ``score_data: {}``, a closed-but-unresolved one has ``disagreement_forecasts: null``, and a
    resolved one carries the list. An empty list is the resolved-but-not-yet-scored state and
    also answers None, since it is not evidence of anyone's absence.
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
    is exactly what is scored, so a non-empty one that does not name ``bot_user_id`` is a forfeit.
    It is also why the token is optional on Mantic: every eventually-resolved question becomes
    measurable without a secret. Its caveat is that a forecast withdrawn before spot time reads
    as ``no_forecast`` (post 500 holds 8 entries against ``nr_forecasters`` 9), which the report
    header states rather than models.
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
    posts_url=POSTS_URL,
    token_env=METACULUS_TOKEN_ENV,
    token_required=True,
    default_slugs=DEFAULT_SLUGS,
    forecast_state=bot_forecast_state,
    authenticated_list_params={},
    sweep_needs_detail_gets=True,
    bot_user_id=None,
    classification_note=(
        "Forecast state: the token's own my_forecasts block, fetched per post where the list page lacks it."
    ),
    all_unknown_hint="my_forecasts was unreadable on every one — run without --no-forfeits to resolve it",
    identity_hint=f"Check that {METACULUS_TOKEN_ENV} is the bot's own token",
)
MANTIC_PROBE = PlatformProbe(
    name=PLATFORM_MANTIC,
    posts_url=f"{MANTIC_API_BASE_URL}/posts/",
    token_env=MANTIC_TOKEN_ENV,
    token_required=False,
    default_slugs=(MANTIC_TOURNAMENT_ID,),
    forecast_state=partial(mantic_forecast_state, bot_user_id=MANTIC_BOT_USER_ID),
    authenticated_list_params={"with_cp": "true"},
    sweep_needs_detail_gets=False,
    bot_user_id=MANTIC_BOT_USER_ID,
    classification_note=(
        f"Forecast state: my_forecasts under {MANTIC_TOKEN_ENV} when set, else the public spot-time snapshot of a "
        "resolved question (a forecast withdrawn before spot time reads as no_forecast); a closed-but-unresolved "
        "question with neither reads unknown."
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
