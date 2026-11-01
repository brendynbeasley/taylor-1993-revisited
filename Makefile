PY ?= .venv/bin/python

.PHONY: env data analysis figures test check all

env:
	python3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) src/01_data_pipeline.py

analysis:
	$(PY) src/02_econometric_suite.py
	$(PY) src/03_structural_and_montecarlo.py

figures:
	$(PY) src/04_visualization.py

test:
	$(PY) -m pytest -q tests

check:
	$(PY) src/05_audit.py

all: data analysis figures test check
