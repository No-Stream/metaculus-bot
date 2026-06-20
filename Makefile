.PHONY: install lock test test_verbose all lint format typecheck typecheck_ty cov audit run benchmark precommit precommit_all precommit_install analyze_correlations analyze_correlations_latest backtest_smoke_test backtest_small backtest_medium backtest_large ablation_qa_research ablation_smoke ablation_small ablation_medium ablation_score test_e2e test_live test_fast check_credits sync_research backfill_research download_research backfill_comments backtest_with_cache

# Stream logs live from recipes; avoid per-target buffering
MAKEFLAGS += --output-sync=none

# OS detection for cross-platform unbuffered output with PTY
# - Linux: stdbuf + script -c "cmd" /dev/null
# - macOS: script -q /dev/null cmd (no stdbuf needed, different script syntax)
# `uv run` executes inside the in-project .venv that `uv sync` manages.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS: script allocates PTY; PYTHONUNBUFFERED handles Python buffering
    define RUN_UNBUFFERED
        PYTHONUNBUFFERED=1 script -q /dev/null uv run python -u $(1)
    endef
else
    # Linux: stdbuf for system-level line buffering + script for PTY
    define RUN_UNBUFFERED
        PYTHONUNBUFFERED=1 stdbuf -oL -eL script -q -c "uv run python -u $(1)" /dev/null
    endef
endif

install:
	uv sync --dev

lock:
	uv lock

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check . --fix

typecheck:
	uv run basedpyright

typecheck_ty:
	uv run ty check

cov:
	$(call RUN_UNBUFFERED,-m pytest --cov=metaculus_bot --cov-report=term-missing)

# Scan uv.lock for known vulnerabilities. osv-scanner is a Go binary (not a
# PyPI package), so it can't be run via uvx — install it with
# `brew install osv-scanner` (see https://google.github.io/osv-scanner/installation/).
# CI runs the equivalent google/osv-scanner-action.
audit:
	osv-scanner scan --lockfile=uv.lock

# Pre-commit helpers (use local cache to avoid readonly home cache)
precommit_install:
	PRE_COMMIT_HOME=.pre-commit-cache uv run pre-commit install

precommit:
	PRE_COMMIT_HOME=.pre-commit-cache uv run pre-commit run

precommit_all:
	PRE_COMMIT_HOME=.pre-commit-cache uv run pre-commit run -a

test:
	$(call RUN_UNBUFFERED,-m pytest)

# Verbose test run: shows which tests are running/failing and where, with
# short tracebacks. Useful when debugging a regression.
test_verbose:
	$(call RUN_UNBUFFERED,-m pytest -v --tb=short)

# One-stop pre-merge check: format, lint, then run tests with verbose output.
# Lint output is informative on violations; tests use -v + short tracebacks so
# you can see which test failed and why without wading through bare pass/fail.
# Recipes run sequentially (default make behavior); if any step fails the
# subsequent steps don't run, so failures surface immediately.
all: format lint typecheck test_verbose

run:
	$(call RUN_UNBUFFERED,main.py)

# DEPRECATED: Community benchmark baseline scoring is broken because Metaculus removed
# the aggregations field from their list API. Use backtest_* targets instead.
# These targets still work for fetching/running questions, but expected_baseline_score is unreliable.
benchmark_run_smoke_test_binary:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_smoke_test'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode run --num-questions 1)

benchmark_run_smoke_test:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_smoke_test'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode custom --num-questions 4 --mixed)

benchmark_run_binary_only:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_small'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode run --num-questions 30)

benchmark_run_small:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_small'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode custom --num-questions 12 --mixed)

benchmark_run_medium:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_medium'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode custom --num-questions 32 --mixed)

benchmark_run_large:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_large'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode custom --num-questions 100 --mixed)

benchmark_display:
	$(call RUN_UNBUFFERED,community_benchmark.py --mode display)

analyze_correlations:
	$(call RUN_UNBUFFERED,analyze_correlations.py $(if $(FILE),$(FILE),benchmarks/))

analyze_correlations_latest:
	$(call RUN_UNBUFFERED,analyze_correlations.py $$(ls -t benchmarks/benchmarks_*.jsonl | head -1))

analyze_correlations_latest_excluding:
	$(call RUN_UNBUFFERED,analyze_correlations.py $$(ls -t benchmarks/benchmarks_*.jsonl | head -1) --exclude-models grok-4 gemini-2.5-pro)

backtest_smoke_test:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 4)

backtest_small:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 12)

backtest_medium:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 32)

backtest_large:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 100)

# Probabilistic-tools ablation benchmark.
# CLI: metaculus_bot/ablation/cli.py (entry point: python -m metaculus_bot.ablation.cli).
# Tournaments default in the CLI (spring-aib-2026 + other 2026 slugs); not pinned here.
ablation_qa_research:
	# Runs only fetch + research + leakage screen + QA dump, then halts.
	# Uses 3/3/3 question mix. No forecasting, no stacking.
	# Config-in-code: --no-gap-fill (gap-fill amplifies leakage on resolved Qs);
	# --gemini-model gemini-2.5-flash (free tier, no Tier 1 billing required).
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --num-binary 3 --num-multiple-choice 3 --num-numeric 3 --resolved-after 2026-01-01 --no-gap-fill --gemini-model gemini-2.5-flash --qa-research)

ablation_smoke:
	# 9 questions: 3 binary, 3 MC, 3 numeric. Full pipeline through scoring.
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --num-binary 3 --num-multiple-choice 3 --num-numeric 3 --resolved-after 2026-01-01 --no-gap-fill --gemini-model gemini-2.5-flash)

ablation_small:
	# 15 questions: 5/5/5.
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --num-binary 5 --num-multiple-choice 5 --num-numeric 5 --resolved-after 2026-01-01 --no-gap-fill --gemini-model gemini-2.5-flash)

ablation_medium:
	# 60 questions: 20/20/20. PENDING USER SIGN-OFF — do not run without explicit go-ahead.
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --num-binary 20 --num-multiple-choice 20 --num-numeric 20 --resolved-after 2026-01-01 --no-gap-fill --gemini-model gemini-2.5-flash)

ablation_score:
	# Re-runs scoring against existing caches (no API spend).
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --stages score)

test_e2e:
	$(call RUN_UNBUFFERED,-m pytest -m e2e -v --tb=short)

test_live:
	$(call RUN_UNBUFFERED,-m pytest -m live -v --tb=short --timeout=300)

test_fast:
	$(call RUN_UNBUFFERED,-m pytest -m "not live and not e2e" --tb=short)

# --- Research persistence (backtest replay) ---

# Sync research archive: download GHA artifacts (source of truth) + backfill
# from Metaculus comments for anything missing. Run before backtests.
sync_research:
	@echo "=== Downloading GHA artifacts ==="
	uv run python scripts/download_research.py $(ARGS)
	@echo ""
	@echo "=== Backfilling from Metaculus comments (historical) ==="
	uv run python scripts/backfill_research_from_comments.py
	@echo ""
	@echo "=== Rebuilding archive ==="
	uv run python scripts/download_research.py --skip-download
	@echo ""
	@echo "Archive ready at backtests/research_archive/latest/"

# Backfill research from existing GitHub Actions logs (Nov 2025 onward).
# Pass ARGS="--limit 100 --status completed" to customize.
backfill_research:
	uv run python scripts/backfill_research_from_logs.py $(ARGS)

# Download research artifacts from recent GHA runs into local archive.
download_research:
	uv run python scripts/download_research.py $(ARGS)

# Backfill from Metaculus bot comments (historical, covers full tournament).
backfill_comments:
	uv run python scripts/backfill_research_from_comments.py $(ARGS)

# Run backtest using cached (non-leaky) research from the archive.
backtest_with_cache:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 20 --research-dir backtests/research_archive/latest $(ARGS))

# Check OpenRouter key balances. Pass ARGS="--key donated" or ARGS="--key personal"
# to limit which key is queried (default: both).
check_credits:
	@uv run python -m metaculus_bot.check_openrouter_credits $(ARGS)
