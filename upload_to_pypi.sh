rm -rf dist/
uv run python -m build
uv run twine check dist/*
uv run twine upload dist/*
