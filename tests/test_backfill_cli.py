"""Smoke tests for the offline backfill script entrypoints."""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKFILL_SCRIPTS = (
    ("backfill_research_from_comments.py", "Backfill research archive from Metaculus bot comments."),
    ("backfill_research_from_logs.py", "Extract research text from GitHub Actions logs into JSONL for backtests."),
)


@pytest.mark.parametrize(("script_name", "description"), _BACKFILL_SCRIPTS)
def test_backfill_help_starts_from_unrelated_cwd_without_credentials(
    tmp_path: Path, script_name: str, description: str
) -> None:
    """Help must stop before token/API or ``gh`` startup work, from outside the checkout."""
    environment = {"HOME": str(tmp_path)}
    # The log backfill's first post-parse operation is ``gh --version``. An empty PATH makes
    # an accidental startup call fail loudly while direct execution of this interpreter works.
    environment["PATH"] = ""

    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / script_name), "--help"],
        capture_output=True,
        cwd=tmp_path,
        env=environment,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        f"{script_name} --help failed from {tmp_path} (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert description in result.stdout
    assert "usage:" in result.stdout
