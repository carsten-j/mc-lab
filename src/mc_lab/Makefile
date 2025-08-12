DIR := .

test:
	uv run pytest

format-fix:
	uv run ruff format $(DIR)
	uv run ruff check --select I --fix

lint-fix:
	uv run ruff check --fix