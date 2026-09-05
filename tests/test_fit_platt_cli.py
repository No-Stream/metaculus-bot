from metaculus_bot.calibration.fit_platt_cli import _extract_mc_pairs


def test_extract_mc_pairs_discards_entire_record_when_one_probability_is_invalid() -> None:
    records = [
        {
            "type": "multiple_choice",
            "question_id": 123,
            "resolution_parsed": "B",
            "options": ["A", "B", "C"],
            "our_forecast_values": [0.4, "bad", 0.3],
        },
        {
            "type": "multiple_choice",
            "question_id": 456,
            "resolution_parsed": "B",
            "options": ["A", "B", "C"],
            "our_forecast_values": [0.2, 0.5, 0.3],
        },
    ]

    assert _extract_mc_pairs(records) == ([0.2, 0.5, 0.3], [False, True, False])
