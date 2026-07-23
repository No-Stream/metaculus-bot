"""Seam-pinning tests for the forecasting-tools integration points that the
0.2.54 -> 0.2.92+ upgrade is known to silently break (FUTURE.md Workstream B).

Seam 1: our ``PchipNumericDistribution`` overrides only the ``.cdf`` property,
which the 0.2.54 publish path reads directly — at ft HEAD that internal read
becomes ``get_cdf()``, which we don't override, so the base-class CDF builder
would silently take over and publish the wrong distribution. Seam 2:
``fetch_hardening`` patches ``MetaculusApi._get_questions_from_api``, the 0.2.54
chokepoint for every question-list GET — at ft HEAD the real fetch moves to
``MetaculusClient``, so the patch could become a silent no-op. Both tests are
green on 0.2.54 and are expected to go red at HEAD until (1) ``get_cdf()`` is
overridden on our distribution subclasses and (2) the fetch-hardening patches
are repointed at ``MetaculusClient``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests
from forecasting_tools import MetaculusApi, NumericDistribution, NumericQuestion
from forecasting_tools.data_models.numeric_report import NumericReport, Percentile
from forecasting_tools.helpers import metaculus_api as ft_api
from forecasting_tools.helpers.metaculus_api import ApiFilter

from metaculus_bot import fetch_hardening
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles

# In-range 13-percentile forecast on closed [0, 100] bounds. PCHIP (monotone
# cubic + uniform mixture) and the base builder (piecewise-linear) both produce
# a valid 201-point CDF here, but with materially different interior probabilities
# — that divergence is what makes the "publish reads our CDF, not the base one"
# assertion meaningful.
_DECLARED_VALUES = [5, 8, 12, 18, 28, 42, 50, 58, 72, 82, 88, 92, 96]


class TestPchipCdfIsWhatGetsPublished:
    """Seam 1: the CDF submitted to Metaculus must be our PCHIP output, not the base builder's."""

    @pytest.fixture
    def question(self) -> NumericQuestion:
        # cdf_size=201 is load-bearing: publish_report_to_metaculus rebuilds a
        # plain base NumericDistribution (dropping our subclass) when cdf_size is None.
        return NumericQuestion(
            question_text="What will the value be?",
            id_of_question=4242,
            id_of_post=4242,
            page_url="https://www.metaculus.com/questions/4242/",
            background_info="",
            resolution_criteria="",
            fine_print="",
            lower_bound=0.0,
            upper_bound=100.0,
            open_lower_bound=False,
            open_upper_bound=False,
            zero_point=None,
            unit_of_measure="units",
            cdf_size=201,
        )

    @pytest.fixture
    def pchip_distribution(self, question: NumericQuestion) -> NumericDistribution:
        declared = [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, _DECLARED_VALUES)]
        sanitized, zero_point = sanitize_percentiles(declared, question)
        return build_numeric_distribution(sanitized, question, zero_point)

    def test_publish_submits_our_pchip_cdf_not_the_base_builder(
        self, question: NumericQuestion, pchip_distribution: NumericDistribution, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert type(pchip_distribution).__name__ == "PchipNumericDistribution"
        assert pchip_distribution.cdf_size == 201, "cdf_size must be set or publish drops the subclass"

        our_cdf = [p.percentile for p in pchip_distribution.cdf]
        base_cdf = [
            p.percentile
            for p in NumericDistribution.from_question(pchip_distribution.declared_percentiles, question).cdf
        ]
        # The submitted == our_cdf assertion below only proves the seam if the base
        # builder would have produced something different. Confirm that first.
        max_divergence = float(np.max(np.abs(np.array(our_cdf) - np.array(base_cdf))))
        assert our_cdf != base_cdf, f"PCHIP and base CDFs identical ({max_divergence=}); test can't detect a swap"

        post_numeric = MagicMock()
        post_comment = MagicMock()
        monkeypatch.setattr(MetaculusApi, "post_numeric_question_prediction", post_numeric)
        monkeypatch.setattr(MetaculusApi, "post_question_comment", post_comment)

        report = NumericReport(question=question, prediction=pchip_distribution, explanation="# Seam test")
        # Pydantic must not coerce our subclass away to a plain NumericDistribution,
        # or the .cdf override would be lost before publish ever runs.
        assert report.prediction is pchip_distribution

        asyncio.run(report.publish_report_to_metaculus())

        post_numeric.assert_called_once()
        submitted_question_id, submitted_cdf = post_numeric.call_args.args
        assert submitted_question_id == question.id_of_question
        assert list(submitted_cdf) == our_cdf, "publish must submit our PCHIP CDF, not the base builder's"


def _reset_fetch_hardening_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the patched MetaculusApi methods, sentinel, and global requests.get.

    ``apply_fetch_hardening`` mutates the MetaculusApi class in place and is
    idempotent via a sentinel. Snapshot the current descriptors through
    monkeypatch (auto-restored on teardown) and clear the sentinel so each test
    exercises a fresh install. Mirrors the helper in test_fetch_hardening.py;
    re-implemented here to avoid importing across test modules.
    """

    for name in fetch_hardening._PATCHED_METHODS:
        monkeypatch.setattr(MetaculusApi, name, MetaculusApi.__dict__[name])

    monkeypatch.setattr(ft_api.requests, "get", ft_api.requests.get)

    if hasattr(MetaculusApi, fetch_hardening._SENTINEL):
        monkeypatch.setattr(MetaculusApi, fetch_hardening._SENTINEL, False)
        delattr(MetaculusApi, fetch_hardening._SENTINEL)
    else:
        monkeypatch.setattr(MetaculusApi, fetch_hardening._SENTINEL, False, raising=False)
        delattr(MetaculusApi, fetch_hardening._SENTINEL)


class TestPublicFetchRoutesThroughHardenedChokepoint:
    """Seam 2: the public question-list fetch must route through the hardened chokepoint.

    fetch_hardening wraps ``MetaculusApi._get_questions_from_api`` with bounded
    retry. This drives the framework's public entry points (the ones the bot's
    ``forecast_on_tournament`` actually calls) with the HTTP transport failing
    once then succeeding, and asserts the retry fired — i.e. the public path
    still funnels through the chokepoint we patched. If a future ft version
    reroutes the public fetch away from ``_get_questions_from_api``, the first
    403 propagates and the fetch raises (transport called once, not twice).
    """

    @pytest.fixture(autouse=True)
    def _fast_and_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Zero backoff, no random pre-request sleep, a token for _get_auth_headers,
        # two retries, and a clean hardening install for this test.
        monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_BACKOFF_BASE", 0.0)
        monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_BACKOFF_JITTER", 0.0)
        monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_BACKOFF_BASE", 0.0)
        monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_BACKOFF_JITTER", 0.0)
        monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
        monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)
        monkeypatch.setattr("forecasting_tools.helpers.metaculus_api.time.sleep", lambda *_: None)
        monkeypatch.setenv("METACULUS_TOKEN", "test-token")
        _reset_fetch_hardening_state(monkeypatch)

    def _install_403_then_ok_transport(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """Stage forecasting-tools' ``requests.get``: 403 on the first call, empty page after.

        Installed BEFORE apply_fetch_hardening so the global GET-timeout wrapper
        closes over this fake, mirroring the real install over ft's requests.get.
        Returns a mutable call counter.
        """

        n_calls = {"n": 0}

        def fake_get(*args, **kwargs):
            n_calls["n"] += 1
            response = MagicMock()
            if n_calls["n"] == 1:
                # The observed 2026-05-19 incident: a CDN/WAF-style 403.
                response.status_code = 403
                response.url = "https://www.metaculus.com/api/posts/?dummy"
                response.reason = "Forbidden"
                response.text = "API only available to authenticated users"
                response.json.return_value = None
                response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden", response=response)
            else:
                response.status_code = 200
                response.content = b'{"results": []}'
                response.raise_for_status.return_value = None
            return response

        monkeypatch.setattr(ft_api.requests, "get", fake_get)
        return n_calls

    def test_get_all_open_questions_from_tournament_retries_through_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bot's actual fetch (forecast_on_tournament -> this) must absorb a transient 403."""
        n_calls = self._install_403_then_ok_transport(monkeypatch)
        fetch_hardening.apply_fetch_hardening()

        result = MetaculusApi.get_all_open_questions_from_tournament("ft-seam-test")

        assert result == []  # fetch succeeded — the retry absorbed the 403
        assert n_calls["n"] == 2  # transport called twice: 403 then success, through the chokepoint

    def test_get_questions_matching_filter_retries_through_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other documented public entry point routes through the same chokepoint."""
        n_calls = self._install_403_then_ok_transport(monkeypatch)
        fetch_hardening.apply_fetch_hardening()

        api_filter = ApiFilter(allowed_tournaments=["ft-seam-test"], allowed_statuses=["open"])
        result = asyncio.run(MetaculusApi.get_questions_matching_filter(api_filter))

        assert result == []
        assert n_calls["n"] == 2
