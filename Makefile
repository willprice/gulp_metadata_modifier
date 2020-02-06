PYTHON ?= python

.PHONY: build
build:
	$(PYTHON) setup.py sdist

.PHONY: docs
docs:
	$(MAKE) -C docs html

.PHONY: mypy
mypy:
	mypy gulp_metadata_modifier.py
