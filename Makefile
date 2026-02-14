.PHONY: conda_env install test run benchmark lint format precommit precommit_all precommit_install analyze_correlations analyze_correlations_latest backtest_smoke_test backtest_small backtest_medium backtest_large

# Stream logs live from recipes; avoid per-target buffering
MAKEFLAGS += --output-sync=none

# Absolute Python in conda env (use tilde to avoid hardcoding username)
PY_ABS := ~/miniconda3/envs/metaculus-bot/bin/python

# OS detection for cross-platform unbuffered output with PTY
# - Linux: stdbuf + script -c "cmd" /dev/null
# - macOS: script -q /dev/null cmd (no stdbuf needed, different script syntax)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS: script allocates PTY; PYTHONUNBUFFERED handles Python buffering
    define RUN_UNBUFFERED
        PYTHONUNBUFFERED=1 PYTHONPATH=. script -q /dev/null $(PY_ABS) -u $(1)
    endef
else
    # Linux: stdbuf for system-level line buffering + script for PTY
    define RUN_UNBUFFERED
        PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u $(1)" /dev/null
    endef
endif

# for reference, won't actually persist to the shell
# conda_env:
# 	conda activate metaculus-bot

install:
	conda run -n metaculus-bot poetry install

lock:
	conda run -n metaculus-bot poetry lock

lint:
	conda run -n metaculus-bot poetry run ruff check .

format:
	conda run -n metaculus-bot poetry run ruff format .
	conda run -n metaculus-bot poetry run ruff check . --fix

# Pre-commit helpers (use local cache to avoid readonly home cache)
precommit_install:
	PRE_COMMIT_HOME=.pre-commit-cache conda run -n metaculus-bot poetry run pre-commit install

precommit:
	PRE_COMMIT_HOME=.pre-commit-cache conda run -n metaculus-bot poetry run pre-commit run

precommit_all:
	PRE_COMMIT_HOME=.pre-commit-cache conda run -n metaculus-bot poetry run pre-commit run -a

test:
	$(call RUN_UNBUFFERED,-m pytest)

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
