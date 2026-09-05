"""Question stubs and prompt renderers shared by the ``tests/prompts/`` modules.

Every module in that package renders the same three base prompts over the same throwaway
question stubs, so the builders live here once. A plain module rather than a ``conftest.py``,
following ``tests/resolution_source_fakes.py``: none of these is a pytest fixture, nothing in
the prompt tests needs autouse setup, and an ordinary import is easier to follow than fixture
injection for a function that takes arguments.

Scope: these are the helpers used by MORE THAN ONE module. A one-off render stays inline in the
test that needs it, and the roughly twenty readable one-line ``binary_prompt(_binary_q(),
research="r")`` calls elsewhere are deliberately left as they are.
"""

import re
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from metaculus_bot.prompts import (
    MARKET_SNAPSHOT_SECTION_HEADER,
    asknews_summarizer_prompt,
    binary_prompt,
    multiple_choice_prompt,
    numeric_prompt,
    stacking_binary_prompt,
    stacking_multiple_choice_prompt,
    stacking_numeric_prompt,
)

# Research that carries a rendered prediction-market section. The market clause in the three
# forecaster prompts is gated on this header (see ``tests/prompts/test_research_clauses.py``),
# so tests that assert on the clause must hand the prompt research that would actually have carried a snapshot.
_RESEARCH_WITH_MARKETS = f"Some news.\n\n{MARKET_SNAPSHOT_SECTION_HEADER}\n| venue | market | prob |\n| k | m | 0.4 |"


def _binary_q(
    open_time: datetime | None = None,
    resolve_time: datetime | None = None,
) -> MagicMock:
    """Minimal question stub with the attributes the prompts read."""
    q = MagicMock()
    q.question_text = "Will X occur by 2030?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = "fp"
    q.open_time = open_time if open_time is not None else datetime.now() - timedelta(days=30)
    q.scheduled_resolution_time = resolve_time if resolve_time is not None else datetime.now() + timedelta(days=365)
    return q


def _mc_q(**kwargs) -> MagicMock:
    q = _binary_q(**kwargs)
    q.options = ["A", "B", "C"]
    return q


def _numeric_q(**kwargs) -> MagicMock:
    q = _binary_q(**kwargs)
    q.unit_of_measure = "widgets"
    q.lower_bound = 0
    q.upper_bound = 1000
    return q


def _flat(text: str) -> str:
    """Lowercase and collapse all whitespace.

    The prompt constants are pre-indented for ``clean_indents``, so an assertion on their
    wording must not depend on where the lines happen to wrap.
    """
    return " ".join(text.lower().split())


def _binary_prompt_text() -> str:
    return binary_prompt(_binary_q(), research="r")


def _mc_prompt_text() -> str:
    return multiple_choice_prompt(_mc_q(), research="r")


def _numeric_prompt_text() -> str:
    return numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")


def _stacked_prompt_texts() -> list[str]:
    """All three stacking prompts, for the scope guard every base-prompt rule carries."""
    return [
        stacking_binary_prompt(_binary_q(), research="r", base_predictions=["a1", "a2"]),
        stacking_multiple_choice_prompt(_mc_q(), research="r", base_predictions=["a1", "a2"]),
        stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        ),
    ]


def _extract_last_json_block(prompt: str) -> str:
    """Return the body of the LAST fenced ```json block in a prompt.

    Every forecasting prompt ends on its STRUCTURED FORECAST example, and the ladder
    (``value_extraction``) reads that block as the authoritative forecast, so the example
    is what teaches the model the schema. Shared because two modules read it: the schema-block
    suite parses the example as real JSON, and the base-rule suite asserts that neither Phase 1
    rule added a field to it.
    """
    blocks = re.findall(r"```json\s*\n(.*?)\n\s*```", prompt, re.DOTALL)
    assert blocks, "no fenced json block found in prompt"
    return blocks[-1]


def _summarizer_prompt(**overrides) -> str:
    """Build the AskNews summarizer prompt with representative defaults."""
    kwargs = {
        "question_text": "Will X happen by 2027?",
        "resolution_criteria": "Resolves YES if X happens",
        "fine_print": "fp",
        "open_date": "2026-03-15",
        "research": "raw asknews articles",
    }
    kwargs.update(overrides)
    return asknews_summarizer_prompt(**kwargs)
