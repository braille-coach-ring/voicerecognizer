.PHONY: install fmt lint check type test ci check-quality sync-baseline clean

install:
	uv sync

fmt:
	uv run ruff format .

lint:
	uv run python script/check_quality_gate.py --step ruff

type:
	uv run python script/check_quality_gate.py --step mypy

test:
	uv run python script/check_quality_gate.py --step pytest

check:
	uv run ruff format . --check
	uv run python script/check_quality_gate.py

ci: check

check-quality:
	uv run python script/check_quality_gate.py

sync-baseline:
	uv run python script/check_quality_gate.py --sync

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +