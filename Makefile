.PHONY: test perftest format-fix lint-fix

# Run tests, formatting, and linting from the repository root
DIR := .

# Run all non-performance tests
test:
	uv run pytest -m 'not performance'

# Run only performance tests and print their output (-s disables capture)
perftest:
	uv run pytest -m performance -s

# Auto-format and sort imports
format-fix:
	uv run ruff format $(DIR)
	uv run ruff check --select I --fix $(DIR)

# Lint and auto-fix issues
lint-fix:
	uv run ruff check --fix $(DIR)
