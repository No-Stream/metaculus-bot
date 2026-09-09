"""``question_platform`` reads the platform off ``page_url`` and defaults to Metaculus."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from metaculus_bot.constants import MANTIC_HOST, PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.question_platform import question_platform


def _question_with_url(page_url: str | None) -> MagicMock:
    q = MagicMock()
    q.page_url = page_url
    return q


class TestQuestionPlatform:
    @pytest.mark.parametrize(
        "page_url",
        [
            f"https://{MANTIC_HOST}/questions/650/",
            f"https://api.{MANTIC_HOST}/questions/650/",
            f"http://{MANTIC_HOST}/questions/650",
        ],
    )
    def test_the_crucible_host_and_its_subdomains_are_mantic(self, page_url: str) -> None:
        assert question_platform(_question_with_url(page_url)) == PLATFORM_MANTIC

    @pytest.mark.parametrize(
        "page_url",
        [
            "https://www.metaculus.com/questions/12345/",
            "https://example.com/q/1",
            "example",
            None,
            # The company's marketing site and blog are outside sources, not the competition.
            "https://www.mantic.com/blog/post",
            "https://notcompetitions.mantic.com.evil.example/",
        ],
    )
    def test_everything_else_is_metaculus(self, page_url: str | None) -> None:
        assert question_platform(_question_with_url(page_url)) == PLATFORM_METACULUS

    def test_the_vocabulary_is_the_research_archives(self) -> None:
        """The two tokens are the archive's ``platform`` field, so their spelling is a data contract."""
        assert PLATFORM_MANTIC == "mantic"
        assert PLATFORM_METACULUS == "metaculus"
