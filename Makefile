ifeq ($(OS),Windows_NT)
PY ?= python
else
PY ?= python3
endif

.PHONY: test compile demo showcase serve

test:
	$(PY) -m unittest discover -s tests

compile:
	$(PY) -m compileall -q second_brain tests

demo:
	$(PY) -m second_brain --vault var/demo-vault --state var/demo-state demo
	$(PY) -m second_brain --vault var/demo-vault --state var/demo-state serve

showcase:
	$(PY) -m second_brain --vault var/showcase-vault --state var/showcase-state demo --showcase
	$(PY) -m second_brain --vault var/showcase-vault --state var/showcase-state serve
