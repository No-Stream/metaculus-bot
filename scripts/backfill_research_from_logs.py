"""Extract research text from GitHub Actions logs and save as JSONL for backtest use.

Parses GHA log output to find "Found Research for URL" blocks logged by the bot,
extracts the research text, and writes structured JSONL records suitable for
offline backtest replay.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Regex patterns for parsing GHA log lines ---

RESEARCH_START = re.compile(
    r"^forecast_job\t[^\t]*\t"
    r"(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+"
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "
    r"(?:main|metaculus_bot\.research_orchestrator) - INFO - "
    r"Found Research for URL (https://www\.metaculus\.com/(?:questions|c)/[^:]+):"
)

APP_LOG_LINE = re.compile(
    r"^forecast_job\t[^\t]*\t"
    r"\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+"
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \S+ - (?:INFO|WARNING|ERROR|DEBUG) - "
)

GHA_PREFIX = re.compile(r"^forecast_job\t[^\t]*\t\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*")

# --- Provider header mapping ---

PROVIDER_HEADERS = {
    "## News Articles (AskNews)": "asknews",
    "## Web Research (Native Search)": "native_search",
    "## Web Research (Google Search via Gemini)": "gemini_search",
    "## Financial & Economic Data": "financial_data",
    "## Web Research (Exa)": "exa",
    "## Web Research (Perplexity)": "perplexity",
    "## Web Research (OpenRouter)": "openrouter",
    "## Research (Custom)": "custom",
}

# --- QID extraction from Metaculus URLs ---

QID_PATTERN = re.compile(r"metaculus\.com/(?:questions|c/[^/]+)/(\d+)")


def extract_qid(url: str) -> int | None:
    match = QID_PATTERN.search(url)
    if match:
        return int(match.group(1))
    return None


def normalize_timestamp(gha_timestamp: str) -> str:
    """Strip fractional seconds from a GHA ISO timestamp: 2026-05-20T15:48:05.5335094Z -> 2026-05-20T15:48:05Z"""
    idx = gha_timestamp.find(".")
    if idx == -1:
        return gha_timestamp
    return gha_timestamp[:idx] + "Z"


def detect_providers(research_text: str) -> list[str]:
    """Detect which research providers contributed based on ## headers in the text."""
    providers = []
    for header, provider_name in PROVIDER_HEADERS.items():
        if header in research_text:
            providers.append(provider_name)
    return providers


def detect_gap_fill(research_text: str) -> bool:
    """Detect if gap-fill was used based on presence of the gap-fill header."""
    return "## Targeted Gap-Fill" in research_text


def strip_gha_prefix(line: str) -> str:
    """Remove the GHA log prefix (job name, step, timestamp) to get raw content."""
    return GHA_PREFIX.sub("", line)


def parse_research_blocks(log_text: str, run_id: str) -> list[dict]:
    """Parse a full GHA log output and extract all research blocks as structured records."""
    records = []
    lines = log_text.split("\n")

    i = 0
    while i < len(lines):
        match = RESEARCH_START.match(lines[i])
        if match:
            gha_timestamp = match.group(1)
            url = match.group(2)
            qid = extract_qid(url)

            # Collect continuation lines until next app-level log line
            research_lines = []
            i += 1
            while i < len(lines):
                if APP_LOG_LINE.match(lines[i]):
                    break
                raw_content = strip_gha_prefix(lines[i])
                research_lines.append(raw_content)
                i += 1

            research_text = "\n".join(research_lines).rstrip("\n")

            records.append(
                {
                    "schema_version": 1,
                    "qid": qid,
                    "page_url": url,
                    "question_text": "",
                    "research_text": research_text,
                    "providers_used": detect_providers(research_text),
                    "run_mode": "tournament",
                    "tournament_id": "",
                    "timestamp": normalize_timestamp(gha_timestamp),
                    "run_id": run_id,
                    "research_chars": len(research_text),
                    "gap_fill_used": detect_gap_fill(research_text),
                }
            )
        else:
            i += 1

    return records


def list_qualifying_runs(workflow: str, limit: int, status: str, since: str, repo: str) -> list[dict]:
    """List qualifying GHA runs via the gh CLI."""
    cmd = [
        "gh",
        "run",
        "list",
        "--repo",
        repo,
        "--workflow",
        workflow,
        "--status",
        status,
        "--limit",
        str(limit),
        "--json",
        "databaseId,createdAt,conclusion,status",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"gh run list failed: {result.stderr}")
        sys.exit(1)

    runs = json.loads(result.stdout)

    if since:
        runs = [r for r in runs if r["createdAt"] >= since]

    return runs


def download_run_log(run_id: int, repo: str) -> str | None:
    """Download the full log for a GHA run. Returns None on failure."""
    cmd = ["gh", "run", "view", str(run_id), "--repo", repo, "--log"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logger.warning(f"Failed to download log for run {run_id}: {result.stderr.strip()}")
        return None
    return result.stdout


def existing_records(output_dir: Path) -> set[tuple[int, str]]:
    """Load existing (qid, run_id) pairs from all JSONL files in the output dir."""
    seen = set()
    if not output_dir.exists():
        return seen
    for jsonl_file in output_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                seen.add((record["qid"], record["run_id"]))
    return seen


def main():
    parser = argparse.ArgumentParser(
        description="Extract research text from GitHub Actions logs into JSONL for backtests."
    )
    parser.add_argument("--workflow", default="run_bot_on_tournament.yaml", help="Workflow filename")
    parser.add_argument("--limit", type=int, default=500, help="Max runs to scan")
    parser.add_argument("--status", default="failure", help="Run status filter: failure, success, completed")
    parser.add_argument("--since", default="2025-11-01", help="Only runs after this date (ISO format)")
    parser.add_argument(
        "--output-dir",
        default="backtests/research_archive/backfill",
        help="Where to write JSONL output",
    )
    parser.add_argument("--dry-run", action="store_true", help="Just list qualifying runs without downloading logs")
    parser.add_argument("--repo", default="No-Stream/metaculus-bot", help="GitHub repo")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Verify gh CLI is available
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        logger.error("gh CLI not found. Install from https://cli.github.com/")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logger.error("gh CLI is installed but returned an error. Check authentication with 'gh auth status'.")
        sys.exit(1)

    runs = list_qualifying_runs(args.workflow, args.limit, args.status, args.since, args.repo)
    logger.info(f"Found {len(runs)} qualifying runs (status={args.status}, since={args.since})")

    if args.dry_run:
        for run in runs:
            print(f"  Run {run['databaseId']} created={run['createdAt']} conclusion={run.get('conclusion', '?')}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = existing_records(output_dir)
    output_file = output_dir / "backfill.jsonl"

    total_new_records = 0
    for idx, run in enumerate(runs, 1):
        run_id = str(run["databaseId"])
        print(f"Processing run {run_id} ({idx}/{len(runs)})...", end=" ", flush=True)

        log_text = download_run_log(run["databaseId"], args.repo)
        if log_text is None:
            print("FAILED to download log")
            continue

        records = parse_research_blocks(log_text, run_id=run_id)

        new_records = [r for r in records if (r["qid"], r["run_id"]) not in seen]
        if not new_records:
            print(f"found {len(records)} blocks, 0 new")
            continue

        with open(output_file, "a") as f:
            for record in new_records:
                f.write(json.dumps(record) + "\n")
                seen.add((record["qid"], record["run_id"]))

        total_new_records += len(new_records)
        print(f"found {len(records)} blocks, {len(new_records)} new")

    logger.info(f"Done. Wrote {total_new_records} new records to {output_file}")


if __name__ == "__main__":
    main()
