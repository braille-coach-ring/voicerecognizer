.PHONY: install dev format lint check type clean

install:
	uv sync

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

fix:
	uv run ruff check . --fixclear

check:
	uv run ruff format . --check
	uv run ruff check .

type:
	uv run mypy main.py

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +