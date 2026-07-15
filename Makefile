.PHONY: install lint format-check type test build doctor run-contract dataset-fixture dataset-validate verify clean

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

dataset-fixture:
	rm -rf .ci-dataset-root
	python scripts/create_phase03_fixture.py --output .ci-dataset-root/openneuro/ds004024-v1.0.1

dataset-validate: dataset-fixture
	rm -rf .phase03-validation
	causalneurotwin dataset validate --registry configs/data/openneuro_ds004024.yaml --dataset-root .ci-dataset-root/openneuro/ds004024-v1.0.1 --output-dir .phase03-validation

verify:
	rm -rf .phase02-validation .phase03-validation .ci-dataset-root
	python scripts/verify_repository.py
	ruff check .
	ruff format --check .
	mypy src scripts
	pytest --cov=causalneurotwin --cov-report=term-missing
	python -m build
	causalneurotwin doctor
	causalneurotwin run-contract --config configs/run_contract.example.yaml --output-root .phase02-validation --run-id verify
	python scripts/create_phase03_fixture.py --output .ci-dataset-root/openneuro/ds004024-v1.0.1
	causalneurotwin dataset validate --registry configs/data/openneuro_ds004024.yaml --dataset-root .ci-dataset-root/openneuro/ds004024-v1.0.1 --output-dir .phase03-validation
	rm -rf .phase02-validation .phase03-validation .ci-dataset-root

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage runs .phase02-validation .phase03-validation .ci-dataset-root .ci-dataset-report
