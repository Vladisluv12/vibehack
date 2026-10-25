PY ?= python3

.PHONY: test compile demo serve

test:
	$(PY) -m unittest discover -s tests

compile:
	$(PY) -m compileall -q second_brain tests

demo:
	$(PY) -m second_brain --vault var/demo-vault --state var/demo-state demo
	$(PY) -m second_brain --vault var/demo-vault --state var/demo-state serve
