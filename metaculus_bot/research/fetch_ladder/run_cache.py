"""Bounded process-run cache of complete reads, before caller verdict and presentation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from metaculus_bot.research import document_cache, resolution_presentation
from metaculus_bot.research.fetch_ladder.policy import LadderPolicy
from metaculus_bot.research.fetch_ladder.throttle import matched_throttle_phrase
from metaculus_bot.research.fetch_ladder.verdict import PageExtraction
from metaculus_bot.research.http_fetch import DatawrapperChartRef
from metaculus_bot.research.resolution_body_text import _truncate_with_marker
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute
from metaculus_bot.research.wayback import WaybackSnapshot, snapshot_age_days, wayback_lead

_MAX_ENTRIES = 50


class ReadArtifact(Protocol):
    @property
    def url(self) -> str: ...

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None: ...


@dataclass(frozen=True, slots=True)
class HtmlRead:
    url: str
    http_status: int | None
    content_type: str | None
    extraction: PageExtraction
    chart_block: str
    datawrapper_charts: tuple[DatawrapperChartRef, ...]
    unreadable_embeds: tuple[str, ...]
    links: tuple[str, ...]
    routing_body: bytes

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None:
        del query, now
        if policy.verdict.body_route(self.content_type or "", self.routing_body) != "html":
            return None
        read = policy.verdict.html(
            self.extraction, chart_block=self.chart_block, unreadable_embeds=list(self.unreadable_embeds)
        )
        floor = policy.thin_content_escalation_chars
        escalate = bool(
            floor is not None
            and not self.chart_block
            and (read.status != "success" or len(read.published_text.strip()) < floor)
        )
        text = ""
        if read.status == "success":
            text = resolution_presentation._page_text_with_leads(
                read.published_text,
                self.url,
                list(self.unreadable_embeds) if policy.disclose_unreadable_embeds else [],
                self.chart_block,
                cap=policy.per_url_max_chars,
            )
        return FetchResult(
            url=self.url,
            status=read.status,
            text=text,
            http_status=self.http_status,
            content_type=self.content_type,
            datawrapper_charts=list(self.datawrapper_charts),
            unreadable_embeds=list(self.unreadable_embeds),
            status_reason=read.status_reason,
            route=route,
            chrome_metric_withheld=self.extraction.chrome_metric_withheld,
            precision_rescued=self.extraction.precision_rescued,
            links=list(self.links) if policy.collect_links else [],
            escalate_rendered=escalate,
        )


@dataclass(frozen=True, slots=True)
class TextRead:
    url: str
    text: str
    http_status: int | None
    content_type: str | None
    lead: str = ""
    routing_body: bytes = b""

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None:
        del query, now
        if policy.verdict.body_route(self.content_type or "", self.routing_body) != "text":
            return None
        if self.lead:
            body = resolution_presentation._lead_then_capped_body(
                self.lead, self.text, self.url, cap=policy.per_url_max_chars
            )
        elif policy.per_url_max_chars is None:
            body = self.text
        else:
            body = _truncate_with_marker(self.text, policy.per_url_max_chars, self.url)
        floor = policy.thin_content_escalation_chars
        return FetchResult(
            url=self.url,
            status="success",
            text=body,
            http_status=self.http_status,
            content_type=self.content_type,
            route=route,
            escalate_rendered=floor is not None and len(self.text) < floor,
        )


@dataclass(frozen=True, slots=True)
class PdfRead:
    url: str
    http_status: int
    content_type: str | None

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None:
        del now
        if policy.verdict.body_route(self.content_type or "", b"%PDF-") != "document":
            return None
        pdf = document_cache.cached_document(self.url)
        if pdf is None:
            return None
        read = policy.verdict.document(pdf, query=query, max_chars=policy.per_url_max_chars, source_url=self.url)
        return FetchResult(
            url=self.url,
            status=read.status,
            text=read.text,
            http_status=self.http_status,
            content_type=self.content_type,
            status_reason=read.status_reason,
            route=route,
        )


@dataclass(frozen=True, slots=True)
class WaybackRead:
    """A dated archive capture plus the complete read made from its bytes."""

    url: str
    snapshot: WaybackSnapshot
    artifact: ReadArtifact
    live_status: str
    live_http_status: int | None
    live_content_type: str | None
    live_failure_class: str | None
    live_exc: str | None
    live_server: str | None

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None:
        uncapped_policy = replace(policy, per_url_max_chars=None)
        snapshot_read = self.artifact.present(uncapped_policy, query=query, route=route, now=now)
        if snapshot_read is None:
            return None
        if snapshot_read.status != "success":
            return replace(snapshot_read, url=self.url, route="wayback")
        age_days = snapshot_age_days(self.snapshot, now)
        max_age_days = policy.wayback_max_age_days
        if age_days is None or (max_age_days is not None and age_days > max_age_days):
            return FetchResult(
                url=self.url,
                status="stale_data",
                text="",
                http_status=self.live_http_status,
                content_type=self.live_content_type,
                failure_class=self.live_failure_class,
                exc=self.live_exc,
                server=self.live_server,
                route="wayback",
            )
        lead = wayback_lead(self.snapshot, age_days, self.live_status)
        return replace(
            snapshot_read,
            url=self.url,
            text=resolution_presentation._lead_then_capped_body(
                lead, snapshot_read.text, self.url, cap=policy.per_url_max_chars
            ),
            route="wayback",
        )


@dataclass(frozen=True, slots=True)
class _Entry:
    artifact: ReadArtifact
    route: FetchRoute


_CACHE: OrderedDict[str, _Entry] = OrderedDict()


async def get(url: str, *, policy: LadderPolicy, query: str, now: datetime, budget_s: float) -> FetchResult | None:
    entry = _CACHE.get(url)
    if entry is None:
        return None
    result = await asyncio.wait_for(
        asyncio.to_thread(entry.artifact.present, policy, query=query, route=entry.route, now=now),
        timeout=max(0.0, budget_s),
    )
    if result is None:
        if _CACHE.get(url) is entry:
            _CACHE.pop(url)
        return None
    if _CACHE.get(url) is entry:
        _CACHE.move_to_end(url)
    return replace(result, cache_hit=True)


def put(requested_url: str, artifact: ReadArtifact, *, route: FetchRoute) -> None:
    entry = _Entry(artifact=artifact, route=route)
    for key in dict.fromkeys((requested_url, artifact.url)):
        _CACHE[key] = entry
        _CACHE.move_to_end(key)
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.popitem(last=False)


def with_lead(artifact: ReadArtifact, lead: str, *, url: str) -> ReadArtifact:
    if not isinstance(artifact, TextRead):
        raise TypeError("a provenance lead can only decorate a text read")
    return replace(artifact, url=url, lead=lead)


def clear() -> None:
    _CACHE.clear()


def cacheable(artifact: ReadArtifact) -> bool:
    """Whether the successful artifact is reusable; throttle interstitials stay retryable."""
    if isinstance(artifact, HtmlRead):
        candidate = artifact.extraction.text or artifact.chart_block
        return matched_throttle_phrase(candidate) is None
    if isinstance(artifact, TextRead):
        return matched_throttle_phrase(artifact.text) is None
    if isinstance(artifact, WaybackRead):
        return cacheable(artifact.artifact)
    return True
