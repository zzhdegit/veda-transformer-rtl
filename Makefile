PYTHON ?= $(shell if command -v python >/dev/null 2>&1; then echo python; else echo python3; fi)

.PHONY: stage0-test stage1-test stage1-rtl-sim stage1-lint stage1-synth stage1-sta stage1b-test stage1b-rtl-sim stage1b-lint stage1b-synth stage2-test stage2-rtl-sim stage2-lint stage2-synth stage3-test stage3-rtl-sim stage3-lint stage3-synth stage4-test stage4-rtl-sim stage4-lint stage4-synth stage4p1-test stage4p1-rtl-sim stage4p1-lint stage4p1-synth stage5-test stage5-rtl-sim stage5-lint stage5-synth stage6a-test stage6b-test stage6b-rtl-sim stage6b-lint stage6b-synth stage6c-test stage6c-rtl-sim stage6c-lint stage6c-synth stage6d-test stage6d-rtl-sim stage6d-lint stage6d-synth stage6e-test stage6e-rtl-sim stage6e-lint stage6e-synth stage6-test stage6-rtl-sim stage6-lint stage6-synth

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

stage1b-test:
	$(PYTHON) scripts/sim/run_stage1b_tests.py

stage1b-rtl-sim:
	bash scripts/sim/run_stage1b_vcs.sh

stage1b-lint:
	$(PYTHON) scripts/lint/run_stage1b_lint.py

stage1b-synth:
	$(PYTHON) scripts/synth/run_stage1b_synth_check.py

stage2-test:
	$(PYTHON) scripts/sim/run_stage2_tests.py

stage2-rtl-sim:
	bash scripts/sim/run_stage2_vcs.sh

stage2-lint:
	$(PYTHON) scripts/lint/run_stage2_lint.py

stage2-synth:
	$(PYTHON) scripts/synth/run_stage2_synth_check.py

stage3-test:
	$(PYTHON) scripts/sim/run_stage3_tests.py

stage3-rtl-sim:
	bash scripts/sim/run_stage3_vcs.sh

stage3-lint:
	$(PYTHON) scripts/lint/run_stage3_lint.py

stage3-synth:
	$(PYTHON) scripts/synth/run_stage3_synth_check.py

stage4-test:
	$(PYTHON) scripts/sim/run_stage4_tests.py

stage4-rtl-sim:
	bash scripts/sim/run_stage4_vcs.sh

stage4-lint:
	$(PYTHON) scripts/lint/run_stage4_lint.py

stage4-synth:
	$(PYTHON) scripts/synth/run_stage4_synth_check.py

stage4p1-test: stage4-test

stage4p1-rtl-sim: stage4-rtl-sim

stage4p1-lint: stage4-lint

stage4p1-synth: stage4-synth

stage5-test:
	$(PYTHON) scripts/sim/run_stage5_tests.py

stage5-rtl-sim:
	bash scripts/sim/run_stage5_vcs.sh

stage5-lint:
	$(PYTHON) scripts/lint/run_stage5_lint.py

stage5-synth:
	$(PYTHON) scripts/synth/run_stage5_synth_check.py

stage6a-test:
	$(PYTHON) scripts/sim/run_stage6a_tests.py

stage6b-test:
	$(PYTHON) scripts/sim/run_stage6b_tests.py

stage6b-rtl-sim:
	bash scripts/sim/run_stage6b_vcs.sh

stage6b-lint:
	$(PYTHON) scripts/lint/run_stage6b_lint.py

stage6b-synth:
	$(PYTHON) scripts/synth/run_stage6b_synth_check.py

stage6c-test:
	$(PYTHON) scripts/sim/run_stage6c_tests.py

stage6c-rtl-sim:
	bash scripts/sim/run_stage6c_vcs.sh

stage6c-lint:
	$(PYTHON) scripts/lint/run_stage6c_lint.py

stage6c-synth:
	$(PYTHON) scripts/synth/run_stage6c_synth_check.py

stage6d-test:
	$(PYTHON) scripts/sim/run_stage6d_tests.py

stage6d-rtl-sim:
	bash scripts/sim/run_stage6d_vcs.sh

stage6d-lint:
	$(PYTHON) scripts/lint/run_stage6d_lint.py

stage6d-synth:
	$(PYTHON) scripts/synth/run_stage6d_synth_check.py

stage6e-test:
	$(PYTHON) scripts/sim/run_stage6e_tests.py

stage6e-rtl-sim:
	bash scripts/sim/run_stage6e_vcs.sh

stage6e-lint:
	$(PYTHON) scripts/lint/run_stage6e_lint.py

stage6e-synth:
	$(PYTHON) scripts/synth/run_stage6e_synth_check.py

stage6-test:
	$(PYTHON) scripts/sim/run_stage6_tests.py

stage6-rtl-sim:
	bash scripts/sim/run_stage6_vcs.sh

stage6-lint:
	$(PYTHON) scripts/lint/run_stage6_lint.py

stage6-synth:
	$(PYTHON) scripts/synth/run_stage6_synth_check.py
