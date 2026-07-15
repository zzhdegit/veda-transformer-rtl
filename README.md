# VEDA Transformer RTL Accelerator

This repository is for a Transformer RTL accelerator based on the VEDA dataflow
ideas, with the current project line focused on generation attention and a
single accepted Pre-Norm Transformer layer.

## Current Stage

Stage 7: Pre-Norm Transformer Layer.

Status: STAGE 7 ACCEPTANCE AUDIT PASS.

The accepted top is:

- `rtl/transformer/transformer_layer.sv`

The frozen Stage 7 structure is:

```text
n1 = RMSNorm(x)
a  = MHA(n1)
r1 = x + a
n2 = RMSNorm(r1)
h1 = W1(n2)
h  = ReLU(h1)
f  = W2(h)
y  = r1 + f
```

Stage 7 wraps exactly one frozen Stage 6 `projection_integrated_mha` child.
Throughput, physical memory, timing pipeline, STA, layout, and PPA remain
provisional.

Authoritative current inputs are:

- `AGENTS.md`
- `PROJECT_STATE.md`
- `HANDOFF.md`
- `docs/stage_07/spec.md`
- `reports/stage_07/summary.md`
- `reports/stage_07/acceptance_audit.md`

Legacy backend notes in this repository are not VEDA specifications unless an
authoritative stage file explicitly names them.

## Directory Structure

```text
AGENTS.md
PROJECT_STATE.md
HANDOFF.md
docs/
model/
rtl/
tb/
scripts/
reports/
build/
transformer_rtl_plan_md/
```

## Run Stage 7 Tests

Host:

```bash
python scripts/sim/run_stage7a_tests.py
python scripts/sim/run_stage6_tests.py
python scripts/sim/run_stage5_tests.py
```

Docker EDA environment:

```bash
make stage7a-test
make stage7b-test
make stage7b-rtl-sim
make stage7b-lint
make stage7b-synth
make stage7c-test
make stage7c-rtl-sim
make stage7c-lint
make stage7c-synth
make stage7d-test
make stage7d-rtl-sim
make stage7d-lint
make stage7d-synth
make stage6-test
make stage6-rtl-sim
make stage6-lint
make stage6-synth
make stage5-test
make stage5-rtl-sim
make stage5-lint
make stage5-synth
```

There is currently no unified `stage7-test`, `stage7-rtl-sim`, `stage7-lint`,
or `stage7-synth` make target. Use the Stage 7A/7B/7C/7D phase targets.

The Stage 7 no-stall cycle model can be run directly:

```bash
python model/transformer/transformer_layer_cycle_model.py
```

## PDK Policy

PDKs, standard-cell libraries, memory compiler outputs, and EDA installation
directories are not stored in this repository. Later synthesis and physical
implementation stages must receive those paths through local environment
variables.

## Dual-Track Development

Hardware and model work are tracked separately after Model Stage M1.

- Hardware state remains in `PROJECT_STATE.md` and `HANDOFF.md`.
- Model state is tracked in `ML_PROJECT_STATE.md` and `ML_HANDOFF.md`.
- Model Stage M2 uses `ml/`, `scripts/ml/`, `docs/ml/`, and
  `reports/ml_m2/`.
- Hardware Stage H8 remains isolated from ML-M2.
- Model Stage M3 uses Q2 real weights and model-line testbenches for
  PyTorch/bit-model/RTL co-simulation. Current M3 status is blocked at a
  one-token RTL-vs-bit-model mismatch; see `reports/ml_m3/summary.md`.

## Next Step

Stage 7 is closed for single-layer Pre-Norm Transformer correctness. Future
optimization or physical implementation work must preserve the frozen Stage 7
numeric/interface contract and must not claim SRAM macro binding, STA, layout,
area, power, frequency, WNS, or PPA until a later stage adds the required
technology collateral and reports.
