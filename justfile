install-cpu:
    uv sync --all-groups --no-group gpu

install-gpu:
    uv sync --all-groups --no-group cpu

format:
    uv run ruff check --select I --fix .
    uv run ruff format .

check:
    uv run ruff check .
