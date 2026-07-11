# VEDA Transformer RTL Accelerator

This repository is for a Transformer RTL accelerator based on the VEDA dataflow ideas, with the current project line focused on generation attention rather than voting-based KV cache eviction.

## Current Stage

Stage 0: project constraints, executable single-head attention specification, and a floating-point software reference model.

Authoritative Stage 0 inputs are:

- `AGENTS.md`
- `PROJECT_STATE.md`
- `HANDOFF.md`
- `docs/stage_00_spec.md`
- `transformer_rtl_plan_md/00_spec_and_reference_model.md`

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

## Run Stage 0 Tests

```bash
python -m pytest tb/model
```

Fallback if pytest is unavailable:

```bash
python tb/model/test_reference_attention.py
```

The reference model can also be run directly:

```bash
python model/reference_attention.py --d-head 8 --seq-len 32 --seed 1 --check
```

## PDK Policy

PDKs, standard-cell libraries, memory compiler outputs, and EDA installation directories are not stored in this repository. Later synthesis and physical implementation stages must receive those paths through local environment variables.

## Next Step

Stage 1 implements arithmetic primitives, FIFO/skid-buffer logic, SRAM wrappers, and the first bit-accurate numeric model.
