.PHONY: sphinx test lint build publish clean

sphinx:
	uv run sphinx-build -b html docs docs/_build/html

test:
	uv run pytest

lint:
	uv run ruff check .

build: clean
	uv build

# Needs a PyPI API token: UV_PUBLISH_TOKEN=pypi-... make publish
publish: build
	uv publish

clean:
	rm -rf dist
