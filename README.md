# VEDA Transformer RTL Accelerator

This repository is for a Transformer RTL accelerator based on the VEDA dataflow ideas, with the current project line focused on generation attention rather than voting-based KV cache eviction.

## Current Stage

HW-H9-N1 post-acceptance real-weight numeric repair is the current hardware
baseline for ML-M3 real-weight validation.

Historical H9 thesis tag:

```text
hw-h9-sfu-pe-interleaving-thesis-accepted
9e0b4c9ba42356ee68e489e99cc5cf64e94f607e
```

Repair branch/tag:

```text
hw/h9-real-weight-numeric-repair
hw-h9-real-weight-numeric-repair-accepted
```

ML-M3 real Q2 length1 found a common H8/H9 FFN W2 reduction numeric mismatch
that old artificial vectors did not cover. The root cause was the shared
`fp32_add_wrapper` DesignWare rounding-mode encoding for the project RNE
contract. HW-H9-N1 sets the wrapper to `rnd=3'b000`, adds reduction
association assertions, and adds real-weight and randomized numeric
regressions.

Repair validation entry points:

```bash
make hw-h9-numeric-repair
make hw-h9-q2-length1
make hw-h9-thesis-acceptance
```

The historical H9 thesis architecture acceptance remains historical. For ML-M3
real Q2 deployment validation, use the HW-H9-N1 repair tag after it is pushed.

Hardware Stage H9 undergraduate-thesis accepted baseline: Full-Array Attention
Mapping and SFU-PE Element-Serial Interleaving.

Status: HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE.

Strict verification status: STRICT IP-GRADE H9 VERIFICATION NOT CLOSED.

The repository now contains the Hardware Stage H9 thesis baseline for
paper-native Attention mapping, bounded score/probability stream buffers, and
SFU/PE interleaving. It also contains a matched single-head RTL A/B baseline for
`PAPER_ARRAY+STAGED` versus `PAPER_ARRAY+INTERLEAVED`; the matched RTL data
shows H9 interleaved faster than paper staged at seq16 and seq32 for
D_HEAD=8, 16, and 64. The cycle model is calibrated to that matched RTL
interval for D_HEAD=8, 16, and 64 at seq 1/2/8/16/32/64.

Multi-head/full-layer H9 RTL entries, long-sequence/cache-full entries,
lint/vlogan, DC structural checks, direct H9 datapath reset stress, direct H9
datapath 20-seed random backpressure stress, assertion positive/negative
execution, independent multi-head reset, 24-run multi-head random
backpressure, and Stage5/6/7/8 regressions have executed and passed in the
Docker EDA environment `nailong`.

The full internal transformer-layer reset and multi-endpoint
random-backpressure matrix remains a deferred IP-grade verification
enhancement. Global array sharing, physical memory, timing closure, and PPA
remain provisional.

Authoritative current inputs are:

- `AGENTS.md`
- `PROJECT_STATE.md`
- `HANDOFF.md`
- `docs/hw_h9/spec.md`
- `docs/hw_h9/paper_schedule_evidence.md`
- `reports/hw_h9/summary.md`
- `reports/hw_h9/acceptance_audit.md`
- `docs/hw_h9/thesis_acceptance_scope.md`
- `reports/hw_h9/deferred_ip_verification.md`
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

For the undergraduate thesis acceptance bundle:

```bash
make PYTHON=python3 hw-h9-thesis-acceptance
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

Hardware Stage H9 is accepted for undergraduate thesis scope. Model Stage M3
may use this hardware baseline only after a separate user-approved task. Do not
enter Hardware Stage H10 or claim full IP-grade verification closure from the
current H9 thesis baseline.
