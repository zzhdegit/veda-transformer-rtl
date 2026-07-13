# Stage 8 Specification

Stage 8 implements a Paper-Structured 8x8x2 PE Array and maps the current
Attention QK and sV compute paths onto it. This stage does not change the
Stage 7 Transformer math, FP16/FP32 boundaries, RMSNorm, Residual, FFN,
Softmax algorithm, SFU scheduling, KV cache layout, cache commit semantics,
cache eviction, model structure, multilayer execution, or other throughput
optimizations.

## Functional Boundary

The accepted Stage 7 layer remains:

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

Stage 8 changes only the Attention PE architecture used for:

- `Q x K^T` score generation.
- `P x V` softmax-value accumulation.

The Stage 6 projection PE for WQ/WK/WV/WO and the Stage 7 FFN PE for W1/W2
remain legacy `reconfigurable_pe_core` users.

Projection PE and FFN PE remain legacy.

## Physical Array

The Stage 8 paper-structured array has:

```text
8 rows x 8 columns x 2 PE groups = 128 physical PE cells
```

The RTL hierarchy must preserve row index, column index, group index, PE type,
and partial-sum connectivity. A flat parameterized lane array is not sufficient
for acceptance.

## PE Groups

Each PE group is an independent 8x8 array. The paper does not publish full
group scheduling details, so this repository freezes the following rule:

- Group 0 and group 1 share one command stream and have independent group masks.
- A transaction may use group 0 only, group 1 only, or both groups.
- A group that is masked off must not accumulate, forward, or emit results.
- Per-group counters report active and idle cycles.
- Group scheduling is an implementation detail inside the Stage 8 array and
  must not change Attention external ready/valid semantics.

## PE Types

Type-A and Type-B are assigned per row in each group:

- Type-A: columns 0, 2, 4, and 6, corresponding to paper PEs 1, 3, 5, and 7.
- Type-B: columns 1, 3, 5, and 7, corresponding to paper PEs 2, 4, 6, and 8.

Type-A supports local product participation in the reduction path. Type-B
supports the forwarded-input role described by the paper. In outer-product
mode both PE types can perform local accumulation when active; type differences
matter for the inner-product reduction topology.

## PE Cell Contract

Each PE cell has:

- Operand A input.
- Operand B input.
- FP32 partial-sum input.
- FP32 local accumulator.
- FP32 forwarded partial-sum output.
- Mode.
- Valid.
- Ready.
- Status.
- Invalid flag.
- Row, column, group, and type identity parameters.

Inactive PE behavior:

- Inactive PEs do not update local accumulators.
- Inactive PEs forward a neutral FP32 +0 partial sum unless forwarding an
  explicitly valid upstream partial sum.
- Inactive PEs must not create invalid status.
- Tail masks are applied before accumulation and before reduction.

## Modes

Stage 8 freezes two required modes:

```text
MODE_INNER_PRODUCT
MODE_OUTER_PRODUCT
```

No separate GEMV mode is added unless a later stage proves that it cannot be
represented by one of these mappings.

### Inner-Product Mode

Inputs:

- Q or input vector fragment in FP16.
- K or weight fragment in FP16.
- Row, column, lane, group, and tail masks.
- Metadata.

Behavior:

- FP16 operands expand exactly to FP32.
- PEs multiply active operand pairs.
- L1 reduction reduces each row across the 8 columns.
- L2 reduction reduces the 8 L1 row sums in each group.
- The final score is FP32.
- Partial sums follow the Stage 8 topology order, not the legacy balanced tree
  unless they happen to match.

Completion:

- A result is valid after all active rows, columns, and groups for the command
  have reduced.
- Payload remains stable until `result_ready`.

### Outer-Product Mode

Inputs:

- FP32 scalar probability or normalized activation.
- V or weight fragment in FP16.
- Optional FP32 partial-sum seed.
- Row, column, lane, group, and tail masks.
- Metadata.

Behavior:

- The scalar is broadcast to active PE cells.
- FP16 vector elements expand exactly to FP32.
- Active PEs compute FP32 FMA into local accumulators.
- No spatial partial-sum transmission is used for the main outer-product
  accumulation.

Completion:

- A result vector is valid after the last active sequence/tile update.
- Payload remains stable until `result_ready`.

## Data Directions

The paper explicitly freezes the outer-product scalar broadcast. The remaining
RTL bus directions are repository design decisions:

- Inner QK: Q is provided as the query vector fragment. K is read from the
  Stage 5 token-major cache as `(token, dimension)` and mapped without a
  permanent transpose buffer.
- Outer sV: probability is broadcast. V is read from the Stage 5 token-major
  cache as `(token, dimension)`.
- D_HEAD tails use lane/column masks.
- Sequence tails use row/group masks.

## Reduction

L1 reduction:

- Per-row reduction over 8 columns.
- Type-A/Type-B mapping follows Figure 5(d).
- Masked lanes contribute FP32 +0 and cannot raise invalid.

L2 reduction:

- Per-group reduction over 8 row L1 outputs.
- The 8th PE role in the paper is represented explicitly in the group
  reduction path.
- Two groups are combined only when both are active for the same command.

## Numeric Contract

Stage 8 changes topology only. It preserves:

- Input operands: FP16.
- FP16-to-FP32: exact expansion.
- Multiply/add/MAC: existing FP32 wrapper contract.
- Partial sum: FP32.
- Local accumulator: FP32.
- Output quantization points: unchanged from Stage 5/6/7.
- FP32-to-FP16: existing RNE, FTZ, saturation, and invalid rules.

Stage 8 must not claim paper-exact arithmetic unless every bit-level rule is
later backed by paper evidence.

## Floating-Point Add Order

The Stage 8 array may use a different FP32 add order from the legacy balanced
tree. Therefore:

- Stage 8 RTL must match the Stage 8 bit-accurate model bit-for-bit.
- Stage 8 bit model versus legacy Stage 7 bit model is bit-for-bit only if the
  add order matches.
- If add order differs, reports must include max absolute error, MAE, RMSE,
  relative L2, cosine similarity, ULP difference, and Attention ranking or
  argmax consistency.
- A high-precision model may be used only for error statistics, not as RTL
  tolerance.

## Reset And Abort

The paper does not define RTL reset semantics. Repository rules:

- Reset clears command state, result state, done state, accumulators, metadata,
  status, and counters that are scoped to the active transaction.
- Reset prevents duplicate result and duplicate done commits.
- After reset, a clean command can be accepted after required state reload.
- Abort semantics, if exposed through an adapter, clear in-flight array state
  without changing Stage 5 KV cache commit rules.

## Ready/Valid And Backpressure

Repository rules:

- Commands remain stable until accepted.
- No new command is accepted while a previous command has in-flight data unless
  the controller explicitly supports independent transaction tags.
- Input backpressure must not drop operands.
- Output backpressure must keep payload stable.
- Partial sums must not be re-accumulated during stalls.
- Mode switching is legal only after the previous transaction fully retires.
- Done remains stable until accepted.

## Attention Mapping Rules

The Stage 8 Attention integration must preserve both architecture choices:

```text
ATTENTION_PE_ARCH = LEGACY_PE
ATTENTION_PE_ARCH = PAPER_ARRAY
```

Only the selected architecture may be instantiated in a synthesized top.

QK mapping:

- Use MODE_INNER_PRODUCT.
- Keep K cache layout token-major.
- Do not add permanent K transpose storage.
- Sequence tail masks invalid tokens.
- D_HEAD tail masks invalid dimensions.
- Scale remains after QK score generation, at the existing Stage 5/6 point.

sV mapping:

- Use MODE_OUTER_PRODUCT.
- Softmax probability index must match V token index.
- Keep V cache layout token-major.
- Accumulate output tiles in FP32.
- D_HEAD and sequence tails are masked.

Softmax scheduling remains staged serial:

```text
QK complete -> existing Softmax/SFU -> sV
```

SFU-PE interleaving is out of scope for Stage 8.
