#!/usr/bin/env bash
# Refuse a commit whose git identity is not a GitHub noreply address.
# GitHub's "block command line pushes that expose my email" setting only protects
# emails registered on the pushing account — a stray identity inherited from a
# global gitconfig (the usual way a wrong email reaches a public repo) is invisible
# to it. This guard travels with the repo instead: wherever it is cloned, a commit
# must carry a pseudonymous identity before it can land.
# One-off override: SKIP=require-noreply-email git commit ...
set -euo pipefail

email="$(git config user.email || true)"
case "${email}" in
  *@users.noreply.github.com) exit 0 ;;
  *)
    echo "commit refused: git user.email is '${email:-<unset>}', not a GitHub noreply address." >&2
    echo "Fix with: git config user.email '<username>@users.noreply.github.com'" >&2
    echo "(one-off override: SKIP=require-noreply-email git commit ...)" >&2
    exit 1
    ;;
esac
