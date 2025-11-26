@default:
    just --list

# Update dependency lockfile
upgrade:
    uv sync --upgrade

# Lint the code with ruff
lint:
    ruff check --fix; ruff check --select I --fix

format:
    ruff format