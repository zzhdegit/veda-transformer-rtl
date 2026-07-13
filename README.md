# VEDA Transformer RTL Accelerator

This repository is for a Transformer RTL accelerator based on the VEDA dataflow ideas, with the current project line focused on generation attention rather than voting-based KV cache eviction.

## Current Stage

Hardware Stage H9 closure checkpoint: Full-Array Attention Mapping and SFU-PE
Element-Serial Interleaving.

Status: HW-H9 IN PROGRESS, NOT ACCEPTED.

The repository now contains a Hardware Stage H9 checkpoint for paper-native
Attention mapping, bounded score/probability stream buffers, and single-head
SFU/PE interleaving smoke coverage. It also contains a matched single-head RTL
A/B baseline for `PAPER_ARRAY+STAGED` versus `PAPER_ARRAY+INTERLEAVED`; the
matched RTL data shows H9 interleaved faster than paper staged at seq16 and
seq32 for D_HEAD=8, 16, and 64. The accepted hardware baseline remains Hardware
Stage H8 / Stage 8: paper-structured 8x8x2 PE array correctness and Attention
QK/sV mapping correctness.

Hardware Stage H9 is not accepted yet. Multi-head/full-layer H9 RTL coverage,
exhaustive reset/random-backpressure/cache-full coverage, long-sequence H9
coverage, broad assertion execution evidence, and exact cycle-model-to-RTL
calibration remain open. Global array sharing, physical memory, timing closure,
and PPA remain provisional.

Authoritative current inputs are:

- `AGENTS.md`
- `PROJECT_STATE.md`
- `HANDOFF.md`
- `docs/hw_h9/spec.md`
- `docs/hw_h9/paper_schedule_evidence.md`
- `reports/hw_h9/summary.md`
- `reports/hw_h9/acceptance_audit.md`
- `docs/stage_08/spec.md`
- `docs/stage_08/paper_evidence.md`
- `reports/stage_08/summary.md`

Legacy backend notes in this repository are not VEDA specifications unless an authoritative stage file explicitly names them.

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

## Run Stage 8 Tests

```bash
python scripts/sim/run_stage8_tests.py
```

In the Docker EDA environment:

```bash
make stage8-test
make stage8-rtl-sim
make stage8-lint
make stage8-synth
```

## Run Hardware Stage H9 Checkpoint Tests

```bash
python scripts/sim/run_hw_h9_tests.py
```

In the Docker EDA environment:

```bash
make hw-h9-test
make hw-h9-rtl-sim
make hw-h9-lint
make hw-h9-synth
```

Stage 8 replaces only the Attention QK and sV PE path. Projection WQ/WK/WV/WO
and FFN W1/W2 still use the legacy PE path. The Stage 7 Pre-Norm Transformer
math, FP16/FP32 boundaries, RMSNorm, Residual, FFN, Softmax, KV cache layout,
and commit semantics are unchanged.

The Hardware Stage H9 RTL A/B baseline is reported in:

```bash
reports/hw_h9/matched_rtl_ab_baseline.md
reports/hw_h9/cycle_model_calibration.md
```

The cycle model can be run directly:

```bash
python model/pe_array/paper_array_cycle_model.py
python model/attention/paper_attention_cycle_model.py
python model/attention/paper_interleaved_cycle_model.py
```

## PDK Policy

PDKs, standard-cell libraries, memory compiler outputs, and EDA installation directories are not stored in this repository. Later synthesis and physical implementation stages must receive those paths through local environment variables.

## Next Step

Continue Hardware Stage H9 only after addressing
`reports/hw_h9/acceptance_audit.md`. Do not enter Hardware Stage H10 or create
an H9 accepted tag until all HW-H9 exit conditions are closed.
