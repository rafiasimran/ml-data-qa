.PHONY: help install data check experiment test lint all clean

help:
	@echo "make data       generate the dataset with planted defects"
	@echo "make check      run the QA suite (exit 1 when not fit to train)"
	@echo "make experiment train the models that show what the defects cost"
	@echo "make all        all three, in order"
	@echo "make test       run the test suite"
	@echo "make lint       run ruff"
	@echo "make clean      remove generated data and reports"

install:
	pip install -r requirements-dev.txt

data:
	python data/generate_dataset.py

check:
	PYTHONPATH=src python -m dataqa check

experiment:
	python experiments/train_model.py

all: data
	-PYTHONPATH=src python -m dataqa check
	python experiments/train_model.py

test:
	PYTHONPATH=src python -m pytest -v

lint:
	ruff check src tests data experiments

clean:
	rm -f data/train.csv data/test.csv data/planted_defects.json
	rm -f reports/*.json reports/*.html
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
