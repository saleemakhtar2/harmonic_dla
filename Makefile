.PHONY: sync sync-locked lock format lint type test test-all check benchmark build clean

sync:
	uv sync

sync-locked:
	uv sync --locked

lock:
	uv lock

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run ty check

test:
	uv run pytest -m "not slow" --cov=harmonic_dla --cov-report=term-missing

test-all:
	uv run pytest --cov=harmonic_dla --cov-report=term-missing

check: lint type test

benchmark:
	uv run pytest benchmarks -m benchmark --benchmark-only

build:
	uv build

clean:
	rm -rf .pytest_cache .ruff_cache .ty_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
