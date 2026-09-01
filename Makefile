.PHONY: install test download prepare train-cross-encoder benchmark-full acceptance serve smoke demo-test airflow-smoke

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev,training]"

test:
	$(PYTHON) scripts/tasks.py test

download:
	$(PYTHON) scripts/tasks.py download

prepare:
	$(PYTHON) scripts/tasks.py prepare

train-cross-encoder:
	$(PYTHON) scripts/tasks.py train-cross-encoder

benchmark-full:
	$(PYTHON) scripts/tasks.py benchmark-full

acceptance:
	$(PYTHON) scripts/tasks.py acceptance

serve:
	$(PYTHON) scripts/tasks.py serve

smoke:
	$(PYTHON) scripts/tasks.py smoke

demo-test:
	$(PYTHON) scripts/tasks.py demo-test

airflow-smoke:
	$(PYTHON) scripts/tasks.py airflow-smoke
