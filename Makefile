PYTHON ?= $(shell if command -v python >/dev/null 2>&1; then echo python; else echo python3; fi)

.PHONY: stage0-test stage1-test stage1-rtl-sim stage1-lint stage1-synth stage1-sta

stage0-test:
	$(PYTHON) -m pytest tb/model/test_reference_attention.py

stage1-test:
	$(PYTHON) scripts/sim/run_stage1_tests.py

stage1-rtl-sim:
	bash scripts/sim/run_stage1_vcs.sh

stage1-lint:
	$(PYTHON) scripts/lint/run_stage1_lint.py

stage1-synth:
	$(PYTHON) scripts/synth/run_stage1_synth_check.py

stage1-sta:
	$(PYTHON) scripts/synth/run_stage1_synth_check.py
