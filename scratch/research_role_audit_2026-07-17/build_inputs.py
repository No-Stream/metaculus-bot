"""Build the fixed inputs for the research-role model audit (Zambia Q44229).

Three roles are audited (AskNews summarizer, native-search provider,
disagreement-crux analyzer); every candidate model must see byte-identical
inputs, so this script freezes them to disk once:

1. ``inputs/question_meta.json`` — question text / resolution criteria /
   fine print / open date from the Metaculus API (free).
2. ``inputs/asknews_raw.md`` — ONE fresh AskNews pull via the real
   ``_asknews_provider`` (the raw pre-summarization article dump from the
   2026-07-17 smoke run was never logged, so this is the replacement fixed
   input; operator-approved single paid/rate-gated call).
3. ``inputs/crux_base_texts.json`` — the six forecaster rationales recovered
   verbatim from /tmp/v2-smoke.log (free). These are exactly what
   ``extract_disagreement_crux`` receives in prod, and the smoke run's numeric
   spread (0.343) genuinely exceeded the stacking threshold (0.15), so this is
   a real disagreement case, not a fabricated one.

Usage:
    uv run python scratch/research_role_audit_2026-07-17/build_inputs.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from metaculus_bot.config import load_environment

load_environment()

from forecasting_tools import MetaculusApi  # noqa: E402

from metaculus_bot.research.providers import _asknews_provider  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_inputs")

BASE_DIR = Path(__file__).parent
INPUTS_DIR = BASE_DIR / "inputs"
QUESTION_URL = "https://www.metaculus.com/questions/44229/"
SMOKE_LOG = Path("/tmp/v2-smoke.log")

# Matches the forecaster_runners LLM-output framing in the smoke log.
_BLOCK_RE = re.compile(
    r"^={40}\nLLM OUTPUT \| Model: (?P<model>\S+) \| Question: \d+ \| Length: \d+ chars\n={40}\n"
    r"(?P<body>.*?)"
    r"\n={40}\nEND LLM OUTPUT \| (?P=model)\n={40}",
    re.DOTALL | re.MULTILINE,
)


def build_question_meta() -> dict:
    path = INPUTS_DIR / "question_meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    question = MetaculusApi.get_question_by_url(QUESTION_URL)
    assert question.open_time is not None
    meta = {
        "question_text": question.question_text,
        "resolution_criteria": question.resolution_criteria or "",
        "fine_print": question.fine_print or "",
        "open_date": question.open_time.strftime("%Y-%m-%d"),
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote question_meta.json")
    return meta


def build_crux_base_texts() -> None:
    path = INPUTS_DIR / "crux_base_texts.json"
    if path.exists():
        logger.info("crux_base_texts.json already exists; skipping")
        return
    log_text = SMOKE_LOG.read_text(encoding="utf-8")
    blocks = [(m.group("model"), m.group("body").strip()) for m in _BLOCK_RE.finditer(log_text)]
    assert len(blocks) == 6, f"expected 6 forecaster rationales in smoke log, found {len(blocks)}"
    # Store model names for OUR bookkeeping only; the crux prompt receives just
    # the texts (prod strips model tags before the analyzer sees them too).
    payload = [{"source_model": model, "reasoning": body} for model, body in blocks]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote crux_base_texts.json (%d rationales)", len(blocks))


async def build_asknews_raw() -> None:
    path = INPUTS_DIR / "asknews_raw.md"
    if path.exists():
        logger.info("asknews_raw.md already exists (%d chars); skipping pull", len(path.read_text(encoding="utf-8")))
        return
    question = MetaculusApi.get_question_by_url(QUESTION_URL)
    provider = _asknews_provider()
    raw = await provider(question)
    assert raw.strip() and "No articles were found" not in raw, "AskNews pull returned no articles"
    path.write_text(raw, encoding="utf-8")
    logger.info("Wrote asknews_raw.md (%d chars)", len(raw))


async def main() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    build_question_meta()
    build_crux_base_texts()
    await build_asknews_raw()
    logger.info("All fixed inputs ready in %s", INPUTS_DIR)


if __name__ == "__main__":
    asyncio.run(main())
