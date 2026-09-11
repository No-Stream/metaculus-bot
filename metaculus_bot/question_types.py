"""Shared question-type dispatch, decoupled from the tool-runner import graph.

Hoisted out of ``tool_runner`` (F3/F11) so light observability and
value-extraction consumers can map a ``MetaculusQuestion`` to its
structured-block question-type string without importing ``tool_runner``,
which drags the full ``probabilistic_tools`` package into the import
graph. Keep this module's imports minimal — nothing here may import
``tool_runner``, ``probabilistic_tools`` or ``numeric``. (``forecasting_tools`` is
unavoidable: the isinstance targets live there.)

``date`` is the type of a ``DateQuestion`` as the platform hands it over. The numeric
pipeline forecasts it through ``numeric.date_axis.EpochDateQuestion``, a
``NumericQuestion`` subclass this leaf cannot import, so that adapter reads as
``numeric`` here and its emitters ask ``numeric.date_axis.numeric_qtype`` instead.
"""

from __future__ import annotations

from typing import Literal

from forecasting_tools import BinaryQuestion, MetaculusQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import DateQuestion

QuestionType = Literal["binary", "numeric", "multiple_choice", "date"]


def question_type_of(question: MetaculusQuestion) -> QuestionType | None:
    """Map a question instance to its structured-block type string (None if unsupported)."""
    if isinstance(question, BinaryQuestion):
        return "binary"
    if isinstance(question, DateQuestion):
        return "date"
    if isinstance(question, NumericQuestion):
        return "numeric"
    if isinstance(question, MultipleChoiceQuestion):
        return "multiple_choice"
    return None
