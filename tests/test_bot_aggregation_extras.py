from unittest.mock import MagicMock

import pytest
from forecasting_tools import BinaryQuestion
from forecasting_tools.data_models.data_organizer import PredictionTypes

from main import TemplateForecaster


@pytest.mark.asyncio
async def test_bot_binary_aggregate_rounding():
    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "parser": "mock",
            "researcher": "mock",
            "summarizer": "mock",
        }
    )
    preds: list[PredictionTypes] = [0.3331, 0.3332]
    q = MagicMock(spec=BinaryQuestion)
    agg = await bot._aggregate_predictions(preds, q)
    assert agg == 0.333
