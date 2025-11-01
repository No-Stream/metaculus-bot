import math

from metaculus_bot.comment_trimming import TRIM_NOTICE, trim_comment, trim_section
from metaculus_bot.constants import COMMENT_CHAR_LIMIT, REPORT_SECTION_CHAR_LIMIT


def test_trim_section_preserves_header_and_tail() -> None:
    header = "## Report 1 Summary"
    body_length = REPORT_SECTION_CHAR_LIMIT + 512
    body = "A" * body_length
    original = f"{header}\n{body}"

    trimmed = trim_section(original, "unit-test-section")

    assert trimmed.splitlines()[0] == header
    assert TRIM_NOTICE in trimmed.splitlines()[1]
    assert len(trimmed) == REPORT_SECTION_CHAR_LIMIT

    available = REPORT_SECTION_CHAR_LIMIT - len(header) - len(TRIM_NOTICE) - 2
    assert available > 0
    expected_tail = body[-available:]
    assert trimmed.endswith(expected_tail)


def test_trim_section_noop_when_within_limit() -> None:
    original = "## Report 1 Summary\nshort body"
    trimmed = trim_section(original, "unit-test-noop")
    assert trimmed == original


def test_trim_final_comment_uses_tail() -> None:
    payload = "PREFIX\n" + ("0123456789" * math.ceil((COMMENT_CHAR_LIMIT + 250) / 10))
    trimmed = trim_comment(payload)

    assert trimmed.startswith(TRIM_NOTICE)
    tail_length = COMMENT_CHAR_LIMIT - len(TRIM_NOTICE) - 1
    assert tail_length > 0
    assert len(trimmed) == COMMENT_CHAR_LIMIT
    assert trimmed.endswith(payload[-tail_length:])


def test_trim_final_comment_noop_when_short() -> None:
    payload = "Concise explanation"
    trimmed = trim_comment(payload)
    assert trimmed == payload
