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
- head concat buffer: FP32;
- concat sent into `W_O` GEMV: FP16 after explicit FP32-to-FP16 conversion;
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

The concat order is:

```text
[head_0, head_1, ..., head_(N_HEAD-1)]
```

`W_O` quantization order is exactly `concat_index = 0..D_MODEL-1`. `W_O` then
uses the same output-row-major GEMV layout and the same Stage 2 reduction order.

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

## First Implementation Limits

- Schedule may be fully serial.
- Projection weights, hidden buffers, concat buffers, and K/V cache may remain
  behavioral memories for correctness closure.
- No area, power, WNS, frequency, post-route, or paper PPA comparison is
  produced in Stage 6.
