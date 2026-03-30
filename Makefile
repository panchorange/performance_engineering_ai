.PHONY: setup
setup:
	uv add --dev ruff mypy pre-commit
	uv run pre-commit install

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .

.PHONY: typecheck
typecheck:
	uv run mypy .

.PHONY: check
check: lint typecheck

.PHONY: precommit-install
precommit-install:
	uv run pre-commit install

.PHONY: precommit-run
precommit-run:
	uv run pre-commit run --all-files
