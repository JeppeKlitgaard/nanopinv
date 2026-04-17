install:
    uv sync --all-groups

format:
    uv run ruff check --select I --fix .
    uv run ruff format .

check:
    uv run ruff check .
