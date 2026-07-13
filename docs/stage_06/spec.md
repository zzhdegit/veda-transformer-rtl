# Stage 6 Specification: Projection-Integrated Multi-Head Attention

## Scope

Stage 6 implements projection-integrated multi-head attention only:

```text
hidden state x_t
-> Q/K/V projection
-> Stage 5 shared multi-head causal attention
-> head concat
-> output projection
-> FP32 MHA output
```

Stage 6 does not implement RMSNorm, LayerNorm, residual paths, FFN, GELU, SiLU,
SwiGLU, multiple Transformer layers, voting, eviction, SRAM macro binding, STA,
P&R, or PPA.

The old `transformer_rtl_plan_md/06_full_transformer_layer.md` describes a
larger Transformer layer and physical implementation scope. That content is
deferred: Stage 7 is Norm + Residual + FFN full Transformer layer integration,
and Stage 8 is real PDK, SRAM, STA, P&R, and PPA.

## Parameters

The first accepted relation is:

```text
D_MODEL = N_HEAD * D_HEAD
```

Primary small verification configuration:

```text
N_HEAD = 2
D_HEAD = 8
D_MODEL = 16
PE_NUM = 8
MAX_SEQ_LEN = 8
```

Additional checked configurations:

```text
N_HEAD=1, D_HEAD=8,  D_MODEL=8
N_HEAD=4, D_HEAD=8,  D_MODEL=32
N_HEAD=2, D_HEAD=16, D_MODEL=32
```

## Matrix Layout

All projection weights are FP16 and have logical shape:

```text
W_Q[D_MODEL][D_MODEL]
W_K[D_MODEL][D_MODEL]
W_V[D_MODEL][D_MODEL]
W_O[D_MODEL][D_MODEL]
```

Weights use output-row-major logical layout:

```text
weight[matrix_kind][output_index][input_index]
```

One GEMV output element is:

```text
y[output_index] =
  sum(input_index = 0..input_length-1)
    input[input_index] * W[output_index][input_index]
```

The reduction order is frozen to the Stage 2 PE core order:

1. FP16 operands are exactly extended to FP32.
2. Products are formed per active PE lane.
3. A balanced FP32 reduction tree produces each tile sum.
4. Tile sums are accumulated sequentially in tile arrival order.

Python `sum`, host float reduction, and NumPy reduction are not the golden
order.

## Numeric Policy

Inputs and weights:

- hidden state `x`: FP16;
- `W_Q`, `W_K`, `W_V`, `W_O`: FP16;
- GEMV operands: exact FP16-to-FP32 extension;
- GEMV accumulation: FP32 through Stage 2 reduction and tile accumulation.

Projection boundaries:

- Q/K/V raw projection results: FP32;
- Q/K/V sent into Stage 5: FP16 after explicit FP32-to-FP16 conversion;
- Stage 5 per-head outputs: FP32;
- logical head concat comparison tensor: FP32;
- RTL head concat storage: FP16 only, written by streaming FP32-to-FP16
  conversion as each Stage 5 head output element arrives;
- concat sent into `W_O` GEMV: the FP16 concat buffer contents;
- final Stage 6 output: FP32.

FP32-to-FP16 conversion policy:

- finite FP32 normal values are supported;
- signed zero is supported and sign-preserving;
- FP32 subnormal inputs flush to signed zero and set `underflow_or_ftz`;
- FP16 subnormal results flush to signed zero and set `underflow_or_ftz`;
- rounding is round-to-nearest-even for supported finite normal results;
- mantissa rounding carry and exponent carry are supported;
- overflow saturates to signed maximum finite FP16 (`0x7BFF` or `0xFBFF`) and
  sets `overflow` and `inexact`;
- NaN/Inf inputs are illegal, produce sign-preserving zero, set `invalid`, and
  are assertion failures in RTL simulation.

## Mapping

Projection output to head/dimension:

```text
projection_output_index = head * D_HEAD + dim
```

Q/K/V projection order:

1. all Q output rows;
2. all K output rows;
3. all V output rows;
4. stream aligned `(Q,K,V)` tuples to Stage 5 in head/dimension order.

Concat address:

```text
concat_index = head * D_HEAD + dim
```

For tiled Stage 5 output this is equivalently:

```text
concat_index = output_head * D_HEAD + output_base_dim + lane_index
```

The concat order is:

```text
[head_0, head_1, ..., head_(N_HEAD-1)]
```

The bit model keeps both the logical FP32 concat tensor and the quantized FP16
concat tensor for node-by-node comparison. RTL does not store a complete FP32
concat buffer. It may quantize each active lane as it arrives and write the
FP16 result to `concat_fp16[concat_index]`. This streaming implementation must
be bit-exact equivalent to forming the full logical FP32 concat tensor and then
quantizing each element in concat-index order.

`W_O` computes:

```text
output[o] = sum(i = 0..D_MODEL-1) concat_fp16[i] * W_O[o][i]
```

`W_O` uses output-row-major weight layout and the same Stage 2 reduction order.
No bias is present in Stage 6.

## Shared Projection Datapath

Q, K, V, and W_O all use one shared projection datapath:

- `projection_input_buffer`
- `projection_weight_buffer`
- `shared_gemv_projection_core`
- `reconfigurable_pe_core`

The final top `projection_integrated_mha` directly arbitrates ownership of this
single `projection_controller`. `qkv_projection_engine` remains available for
Stage 6C/6D regressions, but the final Stage 6 top does not instantiate it and
does not instantiate another PE through an output projection engine.

The Stage 5 attention datapath is unchanged from the accepted Stage 5
architecture.

## Transaction Semantics

Stage 5 semantics remain authoritative:

- current-token causal attention is preserved;
- all heads share one committed `valid_seq_len`;
- K/V for the current token are provisional until every head completes;
- the token commits only after all heads finish successfully;
- cache-full behavior produces an invalid done status without K/V write,
  output, or commit.

Stage 6 output projection runs after Stage 5 has committed K/V. If legal finite
inputs are used, output projection invalid must not occur. If output projection
does report invalid in the first implementation, Stage 6 reports invalid but
does not roll back already committed Stage 5 K/V.

Final Stage 6 token ordering is:

1. hidden-state dimensions load;
2. Q projection;
3. K projection;
4. V projection;
5. Q/K/V quantized stream into Stage 5;
6. Stage 5 attention;
7. Stage 5 all-head atomic K/V commit;
8. streamed head concat quantization into the FP16 concat buffer;
9. W_O output projection through the shared projection GEMV;
10. final FP32 output stream;
11. `done_valid`;
12. only after final done handshake may the next hidden-state token begin.

The implementation distinguishes `attention_done`, cache commit completion via
Stage 5 done, `concat_complete`, `output_projection_done`, and final top done.

If QKV projection or Stage 5 attention fails before commit, `valid_seq_len`
does not increase, Stage 5 abort semantics are preserved, and W_O does not
start. If Stage 5 has already committed and a later concat/W_O error occurs,
the K/V commit is not rolled back; final done reports invalid/status, and the
next token still waits for final done or reset.

Active token transactions block weight writes and block the next hidden-state
token. Weight loading is allowed only while no hidden token transaction is
active.

## Final Top Interface

`rtl/attention/projection_integrated_mha.sv` is the final Stage 6 top. Its
external interface includes:

- clock/reset: `clk`, `rst_n`;
- weight load: `weight_valid`, `weight_ready`, `weight_kind`,
  `weight_output_index`, `weight_input_index`, `weight_data_fp16`,
  `weight_last`, `weight_commit`;
- hidden input: `token_valid`, `token_ready`, `token_dim`,
  `token_hidden_fp16`, `token_last_dim`, `token_meta`;
- final tiled output: `output_valid`, `output_ready`, `output_base_dim`,
  `output_vector_fp32`, `output_lane_mask`, `output_status`,
  `output_invalid`, `output_meta`, `output_last`;
- final done/state: `done_valid`, `done_ready`, `done_status`,
  `done_invalid`, `done_meta`, `done_valid_seq_len`,
  `current_valid_seq_len`;
- performance counters listed in the Stage 6 summary and handoff.

`weight_kind` encodes `WQ`, `WK`, `WV`, and `WO`.

## Performance Counters

Counters are cumulative from reset unless a testbench records deltas:

- `perf_generation_steps`: successful Stage 5 commits; cache-full does not
  increment it;
- `perf_total_cycles`: top transaction-active or final-done-valid cycles;
- `perf_q_projection_cycles`, `perf_k_projection_cycles`,
  `perf_v_projection_cycles`: Q/K/V phase residency cycles;
- `perf_qkv_quantization_cycles`: Q/K/V FP32-to-FP16 output fire cycles;
- `perf_attention_cycles`: Stage 5 per-head attention cycles as reported by
  Stage 5;
- `perf_concat_quantization_cycles`: concat quantizer busy/start cycles;
- `perf_output_projection_cycles`: W_O controller active/start cycles;
- `perf_projection_pe_stall_cycles`: shared projection PE stall cycles;
- `perf_attention_pe_stall_cycles`: Stage 5 attention PE stall cycles;
- `perf_sfu_stall_cycles`: Stage 5 SFU stall cycles;
- `perf_weight_stall_cycles`: projection input/weight backpressure cycles;
- `perf_buffer_stall_cycles`: Stage 5 cache/buffer stall cycles;
- `perf_output_stall_cycles`: final output backpressure cycles;
- `perf_peak_valid_seq_len`: largest committed sequence length observed.

## First Implementation Limits

- Schedule may be fully serial.
- Projection weights, hidden buffers, concat buffers, and K/V cache may remain
  behavioral memories for correctness closure.
- No area, power, WNS, frequency, post-route, or paper PPA comparison is
  produced in Stage 6.
