"""The registrable domain of a host, from the vendored public-suffix list.

A leaf: it imports nothing from this package, and it exists so that two unrelated callers can
share one answer to "are these two hosts the same publisher?" without one reaching into the
other. The settlement-source join (``market_retrieval.settlement_join``) collapses a question's
cited hosts and Kalshi's settlement URLs onto their publisher, and the rendered-fetch JSON
harvest (``rendered_fetch``) decides whether a response the page fetched came from the page's
own publisher. The helper lived inside the join first, and the harvest reached it through a
function-scoped import because the join imports ``resolution_source``, which imports the
harvest's module: a real cycle, and one that existed only because a generic string utility sat
inside a feature package.

Why the public-suffix list and not "the last two labels": ``data.bls.gov`` and ``www.bls.gov``
are the same publisher and meet at ``bls.gov``, but the naive rule collapses ``abs.gov.au`` to
``gov.au``, which would make every Australian agency one publisher. The PSL is the published
answer to exactly that question.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Frozen 2026-08-04 from https://publicsuffix.org/list/public_suffix_list.dat — PSL version
# 2026-07-25_14-20-03_UTC, 10,239 rules (281 wildcard, 8 exception), byte-identical to the copy
# the market-retrieval bake-off measured on. Vendored rather than fetched: a `tldextract`-style
# dependency downloads the list at runtime, which the tests' egress guard blocks and which would
# be a new prod failure mode. Refreshing it is a deliberate commit, so a suffix-list change can
# never silently move a measured pool.
#
# `uv_build` ships the `.dat` with NO packaging configuration — MEASURED 2026-08-04 (uv 0.9.18),
# not assumed, when the file lived under `market_retrieval/data/`. `uv build` put it in both
# artifacts (wheel and sdist), and installing the wheel `--no-deps` into a bare venv resolved the
# module-relative path to a real 332,855-byte file parsing to the same 10,239 rules, SHA-256
# identical to the checked-in copy. The file moved here with the helper on 2026-09-04, still
# module-relative and still inside the package, so the same property holds: the backend includes
# every file under the package directory. If a future `[tool.uv.build-backend]` block adds an
# include/exclude list, this file has to be on it — a dropped `.dat` fails at first PSL use with
# FileNotFoundError, and the suite would not catch it because tests run from the checkout where
# the file is always there.
_PUBLIC_SUFFIX_LIST_PATH = Path(__file__).parent / "data/public_suffix_list.dat"


@lru_cache(maxsize=1)
def _public_suffix_rules() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """The PSL split into (exact, wildcard-parent, exception) rule sets.

    The three kinds are the whole algorithm: `gov.au` is exact, `*.ck` means every direct
    child of `ck` is itself a suffix, and `!www.ck` carves one back out. Comments (`//`) and
    blank lines are dropped; the ICANN/PRIVATE section split is deliberately ignored, because
    a private registry (`s3.amazonaws.com`) is still not a publisher boundary we want to
    collapse across.
    """
    exact: set[str] = set()
    wildcard_parents: set[str] = set()
    exceptions: set[str] = set()
    for raw in _PUBLIC_SUFFIX_LIST_PATH.read_text(encoding="utf-8").splitlines():
        rule = raw.strip()
        if not rule or rule.startswith("//"):
            continue
        if rule.startswith("!"):
            exceptions.add(rule[1:].lower())
        elif rule.startswith("*."):
            wildcard_parents.add(rule[2:].lower())
        else:
            exact.add(rule.lower())
    return frozenset(exact), frozenset(wildcard_parents), frozenset(exceptions)


def registrable_domain(host: str) -> str | None:
    """Collapse a host to its registrable domain: the public suffix plus one more label.

    `data.bls.gov` -> `bls.gov`, `abs.gov.au` -> `abs.gov.au` (because `gov.au` is itself a
    public suffix, so the registrable domain is the whole three-label host, NOT `gov.au`).
    Returns None for a host that IS a bare public suffix with nothing registered under it —
    there is no publisher there to join on. The list is the unicode form throughout (`公司.cn`,
    zero `xn--` rules), so a punycode host has to be decoded before it is passed in
    (`settlement_join.normalize_host` does that for its callers).
    """
    exact, wildcard_parents, exceptions = _public_suffix_rules()
    labels = host.split(".")

    # Longest matching rule wins, per the PSL algorithm. An exception rule (`!www.ck`) means
    # the matched suffix is one label SHORTER than the wildcard would make it.
    suffix_len = 1  # An unknown TLD is treated as a suffix of one label, per the PSL default.
    for start in range(len(labels)):
        candidate = ".".join(labels[start:])
        if candidate in exceptions:
            suffix_len = len(labels) - start - 1
            break
        if candidate in exact:
            suffix_len = max(suffix_len, len(labels) - start)
        parent = ".".join(labels[start + 1 :])
        if parent and parent in wildcard_parents:
            suffix_len = max(suffix_len, len(labels) - start)

    if len(labels) <= suffix_len:
        return None
    return ".".join(labels[-(suffix_len + 1) :])
