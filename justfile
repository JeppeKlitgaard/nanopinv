install-cpu:
    uv sync --all-extras --no-extra cuda

install-gpu:
    uv sync --all-extras --no-extra cpu

format:
    uv run ruff check --select I --fix .
    uv run ruff format .

check:
    uv run ruff check .
