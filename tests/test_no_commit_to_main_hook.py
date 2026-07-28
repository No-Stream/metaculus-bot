"""Guards on the pre-commit hook that refuses a commit whose HEAD is ``main``.

``main`` is ruleset-protected on GitHub (PR required, ``lint`` + ``test`` required), so a
direct push is rejected — but only at PUSH time, once the commits already sit on local
``main`` and have to be replayed onto a branch. That happened on 2026-07-27. The hook
moves the refusal to commit time; these tests guard the three ways it silently stops
working:

1. The exec bit is lost, so ``language: script`` fails with an exec error instead of the
   message the script writes. Asserted against the git INDEX mode, which is the copy a
   fresh clone gets — a local ``chmod`` that git never recorded helps nobody else.
2. ``.pre-commit-config.yaml`` stops wiring it at the commit stage, or its ``entry`` path
   drifts from the script on disk. A typo'd ``entry`` fails only at commit time, on
   somebody else's machine.
3. The branch logic regresses — most likely by treating a detached HEAD (rebase, bisect)
   as ``main`` and blocking it.

Class 3 runs the script for real in a throwaway repo built under ``tmp_path``, which is
the only assertion here that proves behavior rather than text. Those repos are
self-contained on purpose: they pin ``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_SYSTEM`` to
``os.devnull`` and set an identity per-invocation, so the operator's global git config
(``init.defaultBranch``, ``commit.gpgsign``, hooks) can neither leak in nor be needed.
No assertion mentions an absolute path: the checkout location differs between this
machine and CI, and asserting on one is how a sibling test shipped red to CI the same day
(see the comment on ``test_plist_points_at_the_wrapper_this_suite_checks`` in
``tests/test_research_sync_job.py``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HOOK_RELPATH = "scripts/hooks/no_commit_to_main.sh"
_HOOK = _REPO_ROOT / _HOOK_RELPATH
_PRECOMMIT_CONFIG = _REPO_ROOT / ".pre-commit-config.yaml"
_HOOK_ID = "no-commit-to-main"


def _local_hook_entry() -> dict[str, object]:
    """The ``no-commit-to-main`` hook mapping from ``.pre-commit-config.yaml``."""
    config = yaml.safe_load(_PRECOMMIT_CONFIG.read_text())
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"] if hook.get("id") == _HOOK_ID]
    assert len(hooks) == 1, f"expected exactly one `{_HOOK_ID}` hook, got {hooks}"
    return hooks[0]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in ``repo``, isolated from the operator's global and system git config.

    Identity is passed per-invocation so a commit works on a machine with no
    ``user.email`` configured (CI), and ``core.hooksPath`` is neutralized so a hook
    configured anywhere on the host cannot run inside the fixture repo.
    """
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    return subprocess.run(
        ["git", "-c", "user.email=test@example.invalid", "-c", "user.name=Test", "-c", "core.hooksPath=", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _repo_on_branch(tmp_path: Path, branch: str) -> Path:
    """A throwaway repo with one commit, checked out on ``branch``."""
    repo = tmp_path / f"repo_{branch.replace('/', '_')}"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", branch, ".")
    (repo / "seed.txt").write_text("seed\n")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _run_hook(repo: Path) -> subprocess.CompletedProcess[str]:
    """Execute the hook script with ``repo`` as the working directory.

    The script is located via ``_HOOK`` (derived from ``__file__``) because invocation
    needs a real path; nothing about that path is asserted on.
    """
    return subprocess.run(
        [str(_HOOK)],
        cwd=repo,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
        capture_output=True,
        text=True,
        check=False,
    )


class TestHookScriptIsExecutableAndShellShaped:
    def test_script_exists_at_the_wired_path(self) -> None:
        assert _HOOK.is_file(), f"{_HOOK_RELPATH} must exist — `language: script` execs it directly"

    def test_git_records_the_executable_bit(self) -> None:
        # The index mode, not the filesystem mode: a `language: script` hook that is not
        # executable in the committed tree fails on a fresh clone with a confusing exec
        # error rather than the script's own message, and a local-only chmod hides that.
        recorded = subprocess.run(
            ["git", "ls-files", "-s", "--", _HOOK_RELPATH],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert recorded.strip(), f"{_HOOK_RELPATH} must be tracked by git"
        assert recorded.startswith("100755 "), (
            f"git must record mode 100755 for {_HOOK_RELPATH} (got: {recorded.split()[0]!r}). "
            "Fix with `chmod +x` followed by `git add`."
        )

    def test_bash_shebang(self) -> None:
        first_line = _HOOK.read_text().splitlines()[0]
        assert first_line == "#!/usr/bin/env bash", f"expected a bash shebang, got {first_line!r}"


class TestPreCommitConfigWiresTheGuardAtCommitTime:
    def test_entry_points_at_a_script_that_exists(self) -> None:
        # A typo here fails only at commit time, on somebody else's machine.
        entry = _local_hook_entry()["entry"]
        assert entry == _HOOK_RELPATH, f"entry must be {_HOOK_RELPATH!r}, got {entry!r}"
        assert (_REPO_ROOT / str(entry)).is_file()

    def test_runs_at_the_commit_stage_only(self) -> None:
        # Explicit, not defaulted: an entry with no `stages` inherits `default_stages`,
        # which is EVERY stage, so the guard would also fire on push and print a
        # "commit refused" message at a push it is not guarding.
        assert _local_hook_entry().get("stages") == ["pre-commit"]

    def test_language_script_and_no_filenames(self) -> None:
        entry = _local_hook_entry()
        # `language: script` is what execs the file directly (hence the exec-bit test
        # above); `always_run` + `pass_filenames: false` make the guard branch-scoped
        # rather than dependent on which files a commit happens to touch.
        assert entry["language"] == "script"
        assert entry["pass_filenames"] is False
        assert entry["always_run"] is True


class TestHookBehaviorInARealRepo:
    """The behavioral proof: run the script against each of the three HEAD states."""

    def test_refuses_a_commit_on_main(self, tmp_path: Path) -> None:
        result = _run_hook(_repo_on_branch(tmp_path, "main"))
        assert result.returncode != 0, f"the hook must refuse on `main`; stdout={result.stdout!r}"

    def test_allows_a_feature_branch(self, tmp_path: Path) -> None:
        result = _run_hook(_repo_on_branch(tmp_path, "feature/some-work"))
        assert result.returncode == 0, f"the hook must allow a feature branch; stderr={result.stderr!r}"

    def test_allows_a_detached_head(self, tmp_path: Path) -> None:
        # A rebase or bisect leaves HEAD detached, where `git branch --show-current`
        # prints empty. Empty must never be read as `main`, or the guard blocks a rebase.
        repo = _repo_on_branch(tmp_path, "main")
        _git(repo, "switch", "-q", "--detach", "HEAD")
        assert _git(repo, "branch", "--show-current").stdout.strip() == "", "expected a detached HEAD"
        result = _run_hook(repo)
        assert result.returncode == 0, f"a detached HEAD must not be blocked; stderr={result.stderr!r}"

    def test_refusal_message_names_the_recovery_path(self, tmp_path: Path) -> None:
        # Without this the message can rot into a bare non-zero exit, which is the
        # failure mode that gets a guard worked around destructively instead of used.
        result = _run_hook(_repo_on_branch(tmp_path, "main"))
        message = result.stdout + result.stderr
        assert "git switch -c" in message, f"the refusal must give the branch-creation command; got {message!r}"
        assert "--no-verify" in message, f"the refusal must document the bypass; got {message!r}"
