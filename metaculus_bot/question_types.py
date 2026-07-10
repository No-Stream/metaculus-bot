"""Shared question-type dispatch, decoupled from the tool-runner import graph.

Hoisted out of ``tool_runner`` (F3/F11) so light observability modules
(e.g. ``shadow_divergence``) can map a ``MetaculusQuestion`` to its
structured-block question-type string without importing ``tool_runner``,
which drags the full ``probabilistic_tools`` package into the import
graph. Keep this module's imports minimal — nothing here may import
``tool_runner`` or ``probabilistic_tools``. (``forecasting_tools`` is
unavoidable: the isinstance targets live there.)
"""

from __future__ import annotations

from typing import Literal

from forecasting_tools import BinaryQuestion, MetaculusQuestion, MultipleChoiceQuestion, NumericQuestion

QuestionType = Literal["binary", "numeric", "multiple_choice"]


def question_type_of(question: MetaculusQuestion) -> QuestionType | None:
    """Map a question instance to its structured-block type string (None if unsupported)."""
    if isinstance(question, BinaryQuestion):
        return "binary"
    if isinstance(question, NumericQuestion):
        return "numeric"
    if isinstance(question, MultipleChoiceQuestion):
        return "multiple_choice"
    return None
