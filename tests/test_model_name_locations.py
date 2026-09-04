"""Model-id string literals may live only in the files pinned below.

AGENTS.md "Config and models" says where a model id is allowed to live, and until this test
existed that rule was enforced only by a reviewer noticing a new literal in a diff. It is
load-bearing twice over: a roster change is a config-era boundary that has to ship in one merge,
and a model id sitting in a module nobody thinks to grep survives a season rotation and quietly
bills the wrong key.

The pin is a set of FILES rather than a set of ids, so a value bump (a Gemini rotation, a roster
swap) stays a one-file edit while a genuinely new location reddens CI. Comments are invisible to
``ast`` and docstrings are skipped, so prose naming a model for its receipt does not trip the
check.
"""

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "metaculus_bot"

# Paths are repo-relative POSIX strings, never absolute: an absolute path derived from the
# developer's checkout passes locally by construction and can only fail in CI.
_ALLOWED_FILES: frozenset[str] = frozenset(
    {
        "metaculus_bot/llm_configs.py",  # forecaster roster + every support-model GeneralLlm
        "metaculus_bot/constants.py",  # bare ids for native-SDK callers that build no GeneralLlm
        "metaculus_bot/fallback_openrouter.py",  # DONATED_KEY_BLOCKED_GOOGLE_MODELS routing blocklist
        "metaculus_bot/ablation/cli_args.py",  # --gemini-model default, offline harness only
        "metaculus_bot/ablation/forecaster_lineup.py",  # prod-mirror + free-tier ablation lineups
        "metaculus_bot/ablation/leakage_screen.py",  # DEFAULT_DETECTOR_MODEL
        "metaculus_bot/ablation/run_stacker.py",  # ablation stacker/parser defaults
        "metaculus_bot/benchmark/bot_factory.py",  # MODEL_CATALOG for the deprecated benchmark harness
        "metaculus_bot/ensemble_analysis/ensemble_simulator.py",  # per-model cost-correction heuristic
    }
)

# Shapes present today, plus the vendor families a future roster could plausibly reach for: an
# OpenRouter slug (``openrouter/<vendor>/<model>``), a vendor-prefixed slug
# (``openai/gpt-5.6-terra``, ``perplexity/sonar-reasoning-pro``), a bare id
# (``gemini-3.8-flash``, ``qwen3-235b``), and the legacy OpenAI o-series as an exact match. Bare
# family tokens ("deepseek", "qwen3") deliberately do NOT match, because
# ensemble_analysis/benchmark_identity.py matches on those to name an ensemble and they are not
# model ids. A bare ``openrouter/`` prefix used for startswith/removeprefix is excluded the same way.
_MODEL_ID_PATTERN = re.compile(
    r"""
    ^openrouter/[a-z0-9._-]+/[a-z0-9]
  | ^o[0-9](?:-mini|-pro)?$
  | (?:^|/)(?:
        gpt-[0-9o]
      | claude-[a-z]+-[0-9]
      | gemini-[0-9]
      | gemma-[0-9]
      | grok-[0-9]
      | sonar(?:-[a-z]|$)
      | deepseek-[a-z0-9]
      | qwen[0-9]+-
      | glm-[0-9]
      | kimi-k[0-9]
      | minimax-m[0-9]
      | nemotron-[0-9]
      | llama-[0-9]
      | mistral-[a-z0-9]
      | command-r
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_RULE_POINTER = 'AGENTS.md "Config and models" — add the id to an allowlisted file, or pin the new location here'


def _docstring_node_ids(tree: ast.Module) -> set[int]:
    """Identity of every ``ast.Constant`` that is a module, class or function docstring."""
    scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, scopes):
            continue
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            ids.add(id(first.value))
    return ids


def _model_id_literals(path: Path) -> list[tuple[int, str]]:
    """Every non-docstring string literal in ``path`` that looks like a model id, with its line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_node_ids(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or id(node) in docstrings:
            continue
        text = node.value
        if text.startswith("http") or "://" in text:  # a URL is not a model id
            continue
        if _MODEL_ID_PATTERN.search(text):
            hits.append((node.lineno, text))
    return hits


def _files_holding_a_model_id() -> dict[str, list[tuple[int, str]]]:
    found: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        hits = _model_id_literals(path)
        if hits:
            found[path.relative_to(_REPO_ROOT).as_posix()] = sorted(hits)
    return found


def test_no_model_id_literal_outside_the_allowlist() -> None:
    """A model id in an unpinned module is the regression this test exists to catch."""
    found = _files_holding_a_model_id()
    offenders = {name: hits for name, hits in found.items() if name not in _ALLOWED_FILES}
    detail = "\n".join(f"  {name}:{line} {text!r}" for name, hits in offenders.items() for line, text in hits)
    assert not offenders, f"model-id string literal in an unpinned file:\n{detail}\n{_RULE_POINTER}"


def test_every_allowlisted_file_still_holds_a_model_id() -> None:
    """A pin that no longer matches anything is dead weight that hides the next real hardcode."""
    missing_paths = sorted(name for name in _ALLOWED_FILES if not (_REPO_ROOT / name).is_file())
    assert not missing_paths, f"allowlisted file does not exist (renamed or deleted?): {missing_paths}"

    stale_pins = sorted(_ALLOWED_FILES - _files_holding_a_model_id().keys())
    assert not stale_pins, f"allowlisted file holds no model-id literal any more; drop the pin: {stale_pins}"
