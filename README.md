# VEDA Transformer RTL Accelerator

This repository is for a Transformer RTL accelerator based on the VEDA dataflow ideas, with the current project line focused on generation attention rather than voting-based KV cache eviction.

## Current Stage

Stage 8: Paper-Structured 8x8x2 Reconfigurable PE Array.

Status: STAGE 8 PASS.

Paper-structured 8x8x2 PE array correctness accepted for the repository
mapping. Attention QK and sV mapping correctness accepted.

SFU-PE interleaving, global array sharing, physical memory, timing closure, and
PPA remain provisional.

Authoritative current inputs are:

- `AGENTS.md`
- `PROJECT_STATE.md`
- `HANDOFF.md`
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

Stage 8 replaces only the Attention QK and sV PE path. Projection WQ/WK/WV/WO
and FFN W1/W2 still use the legacy PE path. The Stage 7 Pre-Norm Transformer
math, FP16/FP32 boundaries, RMSNorm, Residual, FFN, Softmax, KV cache layout,
and commit semantics are unchanged.

The cycle model can be run directly:

```bash
python model/pe_array/paper_array_cycle_model.py
python model/attention/paper_attention_cycle_model.py
```

## PDK Policy

PDKs, standard-cell libraries, memory compiler outputs, and EDA installation directories are not stored in this repository. Later synthesis and physical implementation stages must receive those paths through local environment variables.

## Next Step

Stage 8 is closed for paper-structured PE array and Attention QK/sV mapping
correctness. A later independent stage may investigate SFU-PE interleaving, but
this repository does not yet implement that overlap.
