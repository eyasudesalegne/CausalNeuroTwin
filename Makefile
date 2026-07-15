.PHONY: install lint format-check type test build doctor run-contract verify clean

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

run-contract:
	rm -rf .phase02-validation
	causalneurotwin run-contract --config configs/run_contract.example.yaml --output-root .phase02-validation --run-id phase02-make-validation

verify:
	rm -rf .phase02-validation
	python scripts/verify_repository.py
	ruff check .
	ruff format --check .
	mypy src scripts
	pytest --cov=causalneurotwin --cov-report=term-missing
	python -m build
	causalneurotwin doctor
	causalneurotwin run-contract --config configs/run_contract.example.yaml --output-root .phase02-validation --run-id verify
	rm -rf .phase02-validation

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage runs .phase02-validation
