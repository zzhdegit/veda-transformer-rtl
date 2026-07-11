# Stage 0 Specification

This file is the Stage 0 implementation contract for the first project line. It is intentionally compact and only records decisions that affect implementation, verification, or later PPA work.

## 1. Project Scope

- Build a Transformer RTL accelerator for generation-first inference.
- First executable target is single-head generation attention.
- The attention datapath follows the VEDA flexible-product dataflow idea:
  - `qK^T`: inner-product dataflow.
  - softmax: element-serial reduction and normalization.
  - `s'V`: outer-product dataflow.
- K/V cache storage is token-major:
  - `K_cache[token][dimension]`.
  - `V_cache[token][dimension]`.
- Future expansion order is dynamic KV cache, multi-head attention, QKV and output projection, full Transformer layer, then synthesis/STA/power/layout.

## 2. First Version Exclusions

The first implementation does not include:

- Voting or voting-based KV cache eviction.
- Multi-token cross-task overlap.
- Multiple parallel attention engines.
- RoPE.
- GQA or MQA.
- Full HBM controller.
- Full PDK layout.
- Full IEEE NaN/Inf/denormal exception handling.
- Synthesis, STA, power, or layout flows requiring process libraries during Stage 0.

## 3. Mathematical Definition

Single-head generation attention uses:

```text
q: [D_HEAD]
K: [seq_len][D_HEAD]
V: [seq_len][D_HEAD]
score_i        = dot(q, K_i)
scaled_score_i = score_i / sqrt(D_HEAD)
max_score      = max_i(scaled_score_i)
exp_sum        = sum_i exp(scaled_score_i - max_score)
p_i            = exp(scaled_score_i - max_score) / exp_sum
o_j            = sum_i p_i * V_i,j
```

For generation with a newly produced `q/k/v`, the first-version cache order is:

```text
append current k/v to cache
run current q over updated K/V cache
produce current output
```

The reference model stores scaled scores in the score buffer because normalization needs a second pass after final `max_score` and `exp_sum` are known.

## 4. Target And Small Verification Configurations

Target configuration:

```text
D_HEAD        = 128
MAX_SEQ_LEN   = 4096
PE_NUM_TARGET = 128
```

Small verification configuration:

```text
D_HEAD      = 8
MAX_SEQ_LEN = 32
NUM_HEADS   = 4
D_MODEL     = 32
FFN_DIM     = 128
```

Stage 0 only exercises the single-head subset. `NUM_HEADS`, `D_MODEL`, and `FFN_DIM` are recorded for later staged integration.

## 5. Numeric Format

Frozen Stage 0 numeric rules:

- The floating model is the algorithmic reference only; it is not bit-accurate.
- Finite inputs are assumed for the first version.
- RTL numeric behavior must not be tuned before Stage 1 selects arithmetic wrappers and creates a bit-accurate model.
- Any change to data format, accumulator precision, rounding, saturation, or exception behavior must be documented before implementation.

Provisional first-version numeric targets:

| Item | Provisional target |
|---|---|
| Input activation | FP16 |
| K/V data | FP16 |
| MAC product | At least FP16 multiply semantics; exact implementation TBD |
| MAC accumulator | FP32 or equivalent higher precision |
| Scaled score | FP32-equivalent internal representation |
| Softmax max score | FP32-equivalent internal representation |
| Softmax exp result | FP32-equivalent internal representation |
| Softmax exp sum | FP32-equivalent internal representation |
| Softmax probability | FP32-equivalent internal representation before `s'V` |
| `s'V` accumulator | FP32 or equivalent higher precision |
| Output | FP16 |
| FP16 output rounding | Round-to-nearest-even unless selected IP requires a documented alternative |
| Saturation/clamp | TBD in bit-accurate model; not silently changeable |
| NaN/Inf/denormal | Out of first-version scope |

## 6. Module Boundaries

Initial module boundaries:

```text
shared_gemv_core
softmax_reduction
softmax_normalization
score_buffer
kv_cache_interface
single_head_attention_controller
performance_counter
```

Required ownership:

- `shared_gemv_core`: runtime mode support for inner-product `qK^T` and outer-product `s'V` in later RTL.
- `softmax_reduction`: online `max_score` and `exp_sum` over serial scaled scores.
- `softmax_normalization`: reads buffered scaled scores and emits serial probabilities.
- `score_buffer`: stores one active token's scaled score stream for the first version.
- `kv_cache_interface`: exposes token-major K/V streams without physically transposing K.
- `single_head_attention_controller`: sequences phases using real valid/ready and drain status.
- `performance_counter`: counts cycles, stalls, utilization, and buffer occupancy.

## 7. Dataflow And Macro Pipeline

Phase A:

```text
read q and K_i
shared_gemv_core in QK inner-product mode
scale by 1/sqrt(D_HEAD)
write score_buffer
update softmax_reduction
```

Phase B:

```text
read score_buffer
softmax_normalization computes p_i
read aligned V_i
shared_gemv_core in SV outer-product mode
write head output
```

Allowed first-version overlap:

- `qK^T` PE computation overlaps softmax reduction.
- Softmax normalization overlaps `s'V` outer-product computation.

Disallowed first-version overlap:

- The same PE array does not compute `qK^T` and `s'V` at the same time.
- No overlap across multiple tokens.
- No double-buffered score window in Stage 0/first single-head version.

## 8. Module Interface Principles

All long datapath boundaries use ready/valid. Required rules:

- `valid` stays asserted until a transfer occurs.
- Payload and metadata stay stable while `valid=1` and `ready=0`.
- `ready` may deassert for backpressure.
- `last` marks the final element of the active sequence or vector stream, as defined by the local interface.
- Metadata such as `token_id`, `head_id`, `dimension_id`, and `operation_id` must remain aligned with data through every pipeline.
- Reset returns controllers to idle and clears in-flight valid state.
- Flush behavior must be explicit before any long-latency RTL block is accepted.
- No long combinational ready chain across major modules; use skid buffers or registered ready where needed.

Command-style control must eventually define:

```text
op_mode
active_seq_len
base addresses
stride or bank mapping
start
done
error_flag
```

## 9. Storage Layout

K/V layout is fixed:

```text
K_cache[token][dimension]
V_cache[token][dimension]
```

First-version properties:

- Logical token index equals physical row index.
- No eviction.
- No circular overwrite.
- Cache full behavior for later dynamic KV stage is stop-and-report-error.
- No separate physical transposed K copy.
- Dimension banking is allowed later, but must preserve token-major architectural layout.

Score buffer:

- Stores scaled scores in token order.
- First version depth is at least `active_seq_len`.
- Target depth is `MAX_SEQ_LEN` or a documented tile depth if tiling is introduced.
- Read order matches write order for normalization and V alignment.

## 10. Functional Verification Plan

Stage 0 software tests cover:

- `seq_len = 1`.
- All scores equal.
- One score clearly larger than others.
- Random input cases.
- Online softmax reduction matches direct softmax.
- Probability sum is close to 1.

Later RTL/block tests must add:

- Token lengths around PE and tile boundaries.
- Backpressure on K, V, score, SFU, and output streams.
- Reset and flush during active work.
- Metadata alignment checks.
- Score/V token-index alignment.
- X propagation checks when valid is asserted.

## 11. Performance Counters

The single-head attention block must eventually expose:

```text
qk_cycles
qk_drain_cycles
softmax_finalize_cycles
sv_cycles
sv_drain_cycles
total_cycles
pe_active_cycles
sfu_active_cycles
pe_idle_cycles
sfu_idle_cycles
score_buffer_peak_occupancy
stall_input_cycles
stall_output_cycles
valid_mac_operations
available_mac_slots
```

PE utilization is:

```text
valid_mac_operations / available_mac_slots
```

Stage 0 does not claim any cycle, utilization, timing, area, or power result.

## 12. PPA Pre-Control Principles

- PDK, standard-cell libraries, SRAM macros, and EDA tool installs stay outside the repository.
- No Stage 0 synthesis, STA, power, or layout flow is run.
- PPA, technology, and tool details reported by the reference paper are background only; they are not project claims until reproduced by this repository's scripts and real local tool output.
- SRAM wrappers must separate simulation behavior from physical macro binding in later stages.
- Do not claim valid area from large flip-flop arrays replacing SRAM.
- Keep PE mode control, broadcast scalar, lane masks, clear signals, and reset from becoming unmanaged high-fanout nets.
- Pipeline adder trees and long arithmetic units; do not build single-cycle deep adder trees by default.
- Record all future PPA data in reports only when generated by reproducible scripts and real tool output.

## 13. Stage 1 Input Conditions

Stage 1 may start from:

- This Stage 0 spec.
- `model/reference_attention.py` as the non-bit-accurate algorithm reference.
- The frozen dataflow, storage layout, and ready/valid principles above.

Stage 1 must not assume:

- Arithmetic IP latency or exact port names.
- SRAM macro shape or port count.
- Final FP exception behavior.
- Any PDK path, library file, timing result, power result, or layout result.

Stage 1 must produce:

- Project-owned arithmetic wrappers.
- FIFO/skid-buffer/SRAM wrapper RTL.
- A bit-accurate numeric model aligned to selected arithmetic behavior.
- Unit tests and assertions for valid/ready, metadata stability, overflow/underflow, and reset/flush behavior.
