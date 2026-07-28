#!/usr/bin/env bash
#
# pre-commit hook: refuse a commit whose HEAD is `main`.
#
# WHY: `main` is ruleset-protected on GitHub (PR required, `lint` + `test` must pass), so
# a direct push is rejected — but only at push time, after the commits already exist on
# local `main` and have to be replayed onto a branch. This moves the refusal to commit
# time, where the fix is one `git switch -c`.
#
# `git branch --show-current` rather than `git rev-parse --abbrev-ref HEAD`: it prints
# empty on a detached HEAD (rebase, bisect — must not be blocked) and still prints the
# branch name on an unborn branch, where rev-parse fatals and `set -e` would abort with
# that error instead of the message below.

set -euo pipefail

branch="$(git branch --show-current)"

if [[ "${branch}" != "main" ]]; then
  exit 0
fi

cat >&2 <<'EOF'
Commit REFUSED: HEAD is `main`, and work in this repo goes on a feature branch.

Move the staged changes onto a branch and commit there — staged changes survive a
branch switch, so nothing is lost:

    git switch -c <branch-name>
    git commit -m "..."

If this commit genuinely belongs on `main`, bypass the guard:

    git commit --no-verify
EOF

exit 1
