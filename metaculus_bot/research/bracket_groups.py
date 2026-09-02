"""The bracket-group grammar both Gemini post-passes read the same text through.

``gemini_search._strip_model_citation_indices`` (drop the model's own ``[1.2.3]``
citation indices) and ``gemini_attribution.rewrite_unsupported_attributions``
(replace tier-tag attributions the grounding record cannot back) run back to back
over one response body, and each one has to answer the same three questions: what
counts as a bracket group, where its items end, and how a rewritten group is put
back together. Written twice, the two passes could come to disagree about the same
string while still each looking right in isolation, so the grammar lives here once
and both import it. Deliberately a leaf: it holds no notion of citations, tiers or
grounding, so nothing in it can pull either consumer's judgment into the other's.

A group is a single-line bracketed run with no nested bracket (``[^\\[\\]\\n]*``),
because both passes only ever mean to touch a citation-shaped bracket and a
multi-line or nested one is markdown structure. Items split on ``,`` and ``;`` with
the separator CAPTURED, so a rewritten group keeps the model's own ``;`` where it
wrote one (``[A: ILA; C: Sea news]``) instead of normalizing to ``,``.

What is deliberately NOT shared: which items each pass keeps. The strip drops any
item left with no alphanumeric character after its index is removed; the
attribution check drops only items that are empty once stripped. Those filters are
each pass's own reading of "this item still says something", not grammar, and
folding them together would silently change both.
"""

import re
from collections.abc import Iterator, Sequence

__all__ = ["BRACKET_GROUP_RE", "iter_group_items", "join_group_items", "rebuild_group"]

# ``pre`` captures the single space in front of the group, when the group has a
# non-space character before that space. It exists for the one consumer that can
# delete a group outright (the citation-index strip, on a group that was nothing but
# indices): deleting the group alone leaves ``"office . He"``, so the space goes with
# it. A consumer that only rewrites a group's inner text must re-emit ``pre``, which
# is what :func:`rebuild_group` is for.
BRACKET_GROUP_RE = re.compile(r"(?P<pre>(?<=\S) )?\[(?P<inner>[^\[\]\n]*)\]")

_GROUP_ITEM_SPLIT_RE = re.compile(r"([,;])")


def iter_group_items(inner: str) -> Iterator[tuple[str, str]]:
    """Yield ``(introducing separator, raw item text)`` per comma/semicolon item.

    The separator is ``""`` for the first item and the literal ``,`` or ``;`` that
    introduced each later one. Item text comes through RAW — unstripped and
    unfiltered — because the two passes strip and filter differently (see the module
    docstring), and only one of them can tidy a given item correctly.
    """
    parts = _GROUP_ITEM_SPLIT_RE.split(inner)
    for index in range(0, len(parts), 2):
        yield (parts[index - 1] if index else ""), parts[index]


def join_group_items(items: Sequence[tuple[str, str]]) -> str:
    """Rejoin surviving ``(separator, item)`` pairs into one group's inner text.

    The first surviving item drops whatever separator introduced it — it now leads
    the group — and every later one keeps its own separator plus one space. So a
    group whose first item was removed reads ``[B: Reuters]`` rather than
    ``[, B: Reuters]``.
    """
    return "".join(item if index == 0 else f"{separator} {item}" for index, (separator, item) in enumerate(items))


def rebuild_group(match: re.Match[str], inner: str) -> str:
    """Re-render a matched group with new ``inner``, keeping the space it consumed.

    Every rewrite goes through here so no consumer can drop the ``pre`` space the
    regex swallowed and turn ``"the office [A: FDA]"`` into ``"the office[...]"``.
    """
    return f"{match.group('pre') or ''}[{inner}]"
