import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the tested wrapper is Linux-specific")


def _make_test_with_fake_uv(fake_uv_directory: Path, child_exit_status: int) -> subprocess.CompletedProcess[str]:
    environment = os.environ | {
        "FAKE_UV_EXIT_STATUS": str(child_exit_status),
        "PATH": f"{fake_uv_directory}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(
        ["make", "--no-print-directory", "test"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )


@pytest.fixture
def fake_uv_directory(tmp_path: Path) -> Path:
    fake_uv_directory = tmp_path / "bin"
    fake_uv_directory.mkdir()
    fake_uv = fake_uv_directory / "uv"
    fake_uv.write_text(
        """#!/bin/sh
if [ "$1" != "run" ] || [ "$2" != "python" ] || [ "$3" != "-u" ] || [ "$4" != "-m" ] || [ "$5" != "pytest" ]; then
    exit 64
fi
exit "$FAKE_UV_EXIT_STATUS"
"""
    )
    fake_uv.chmod(0o755)
    return fake_uv_directory


def test_make_test_fails_when_wrapped_command_fails(fake_uv_directory: Path) -> None:
    result = _make_test_with_fake_uv(fake_uv_directory, child_exit_status=7)

    assert result.returncode != 0


def test_make_test_succeeds_when_wrapped_command_succeeds(fake_uv_directory: Path) -> None:
    result = _make_test_with_fake_uv(fake_uv_directory, child_exit_status=0)

    assert result.returncode == 0
