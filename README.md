# VEDA Transformer RTL Accelerator

This repository is for a Transformer RTL accelerator based on the VEDA dataflow ideas, with the current project line focused on generation attention rather than voting-based KV cache eviction.

## Current Stage

Stage 6: Projection-Integrated Multi-Head Attention.

Status: STAGE 6 PASS.

projection-integrated multi-head attention correctness accepted.

throughput, physical memory, and timing pipeline provisional.

Authoritative current inputs are:

- `AGENTS.md`
- `PROJECT_STATE.md`
- `HANDOFF.md`
- `docs/stage_06/spec.md`
- `reports/stage_06/summary.md`

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

## Run Stage 6 Tests

```bash
python scripts/sim/run_stage6_tests.py
```

In the Docker EDA environment:

```bash
make stage6-test
make stage6-rtl-sim
make stage6-lint
make stage6-synth
```

Stage 6 includes Q/K/V projection, Stage 5 multi-head attention, streamed concat
quantization, and W_O output projection. It does not include Norm, Residual, or
FFN.

The cycle model can be run directly:

```bash
python model/projection/projection_mha_cycle_model.py
```

## PDK Policy

PDKs, standard-cell libraries, memory compiler outputs, and EDA installation directories are not stored in this repository. Later synthesis and physical implementation stages must receive those paths through local environment variables.

## Next Step

Stage 6 is closed for projection-integrated MHA correctness. Do not infer that a complete Transformer layer is finished.
