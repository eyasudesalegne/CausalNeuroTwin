.PHONY: install lint format-check type test build doctor verify clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

format-check:
	ruff format --check .

type:
	mypy src scripts

test:
	pytest --cov=causalneurotwin --cov-report=term-missing

build:
	python -m build

doctor:
	causalneurotwin doctor

verify:
	python scripts/verify_repository.py
	ruff check .
	ruff format --check .
	mypy src scripts
	pytest --cov=causalneurotwin --cov-report=term-missing
	python -m build
	causalneurotwin doctor

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
