# Stage 8 Paper Evidence

Stage 8 uses the local repository copy of the VEDA paper as the primary
source: `2507.00797v1.pdf`.

- Title: VEDA: Efficient LLM Generation Through Voting-based KV Cache Eviction and Dataflow-flexible Accelerator
- Version: arXiv v1
- Date shown in paper header: 1 Jul 2025
- Scope used by this stage: Section IV-A, Figure 4, Figure 5, Figure 7, and Table I

This repository implements a paper-structured topology with the repository's
frozen FP16/FP32 arithmetic contract. The paper does not publish all RTL
microarchitecture details, so unspecified items are explicitly treated as
repository design decisions and are not described as paper-exact.

| Design item | Paper evidence | Figure/section/page | Confidence | RTL decision |
|---|---|---|---|---|
| Array dimensions | Table I lists the PE Array parameter as `8*8*2 Reconfigurable PEs`; Figure 7 labels the PE array `8x8x2`. | Table I, PDF p. 6; Figure 7, PDF p. 5 | High | Instantiate 2 groups, each with 8 rows and 8 columns, for 128 physical PE cells. |
| Two PE groups | The paper's hardware summary uses `8*8*2`; it does not describe separate group scheduling details in text. | Table I, PDF p. 6; Figure 7, PDF p. 5 | Medium | Model the third dimension as two independent 8x8 groups with explicit group index, group mask, and per-group activity counters. Group scheduling is a repository design decision. |
| Type-A PE | Section IV-A states that PEs 1, 3, 5, and 7 are type-A, with one adder input from local multiplication and one from another PE. | Section IV-A, Figure 5(d), PDF p. 4 | High | Odd columns in each row are Type-A in inner-product reduction mode. Type-A can consume local product plus forwarded product or partial sum. |
| Type-B PE | Section IV-A states that PEs 2, 4, 6, and 8 are type-B and both adder operands come from other PEs; Figure 5(a) marks Type-B-only datapath. | Section IV-A, Figure 5(a,d), PDF p. 4 | High | Even columns in each row are Type-B in inner-product reduction mode. Type-B exposes forwarding inputs and does not rely on local product for the Type-B adder role. |
| Inner-product path | The paper maps `q x K^T = s` to inner product, with sequence length spatially mapped and one score element produced per cycle. | Section IV-A, Figure 4(a), Figure 5(c), PDF p. 4 | High | Use MODE_INNER_PRODUCT for QK. Q is broadcast across active token columns/groups, K is read in token-major `(l,d)` layout, and D_HEAD is spatially reduced. |
| Outer-product path | The paper maps `s' x V = o` to outer product, with one scalar from `s'` broadcast each cycle and local accumulation over `d`. | Section IV-A, Figure 4(b), Figure 5(b), PDF p. 4 | High | Use MODE_OUTER_PRODUCT for sV. Probability is broadcast, V is spatially mapped by dimension, and per-lane FP32 accumulators hold output partial sums. |
| Partial-sum forwarding | Figure 5 and Section IV-A describe partial sums being transmitted across PEs in inner-product configuration. | Section IV-A, Figure 5(a,c), PDF p. 4 | High | Inner-product mode implements explicit column and row partial-sum forwarding order. Exact ready/valid staging is a repository design decision. |
| L1 reduction | Section IV-A says the L1 adder tree receives multiplication results and accumulates at row level. | Section IV-A, Figure 5(c,d), PDF p. 4 | High | Implement per-row L1 reduction over each 8-column group, respecting the Type-A/Type-B mapping. |
| L2 reduction | Section IV-A says L2 aggregates L1 results and follows similar principles, with the 8th PE acting as a local adder for L2. | Section IV-A, Figure 5(c,d), PDF p. 4 | High | Implement group-level L2 reduction over the 8 row results. Exact pipeline staging is a repository design decision. |
| Broadcast directions | The paper explicitly states the outer-product configuration shares a broadcast input scalar. Figure 5 labels input/weight paths but does not define RTL bus directions. | Section IV-A, Figure 5(b), PDF p. 4 | Medium | Broadcast probability/scalar to all active PEs in outer mode. Broadcast Q or stream Q/K according to inner-mode mapping needed by the current Stage 5 cache interface. |
| Mode switching | Section IV-A describes each PE being controlled by a 2-bit signal for accumulation, forwarding, clearing, or disabling. | Section IV-A, Figure 5(a), PDF p. 4 | Medium | Expose MODE_INNER_PRODUCT and MODE_OUTER_PRODUCT commands. Mode switches are accepted only after the previous transaction fully retires. Ready/valid semantics are repository-defined. |
| Arithmetic format | Section VI says VEDA uses FP16 as default arithmetic, but the paper does not publish bit-level FP exception, rounding, or accumulator contracts. | Section VI, PDF p. 6 | Medium | Keep repository-frozen arithmetic: FP16 operands, exact FP16-to-FP32 expansion, FP32 wrappers for multiply/add/MAC, FP32 partial sums, and existing FP32-to-FP16 boundaries. This is not claimed to be paper-exact arithmetic. |
| KV storage orientation | Section IV-A says the KV matrix can be uniformly stored in `(l,d)` format and read one `(1,d)` vector per step. | Section IV-A, PDF p. 4 | High | Preserve Stage 5 token-major cache layout `K_cache[head][token][dimension]` and `V_cache[head][token][dimension]`; do not add permanent transposed K storage. |
| SFU scheduling boundary | Section IV-B describes element-serial scheduling that overlaps SFU and PE array work. | Section IV-B, Figure 6, PDF pp. 4-5 | High for paper intent; out of scope for Stage 8 | Stage 8 does not implement SFU-PE interleaving. Softmax remains the existing staged serial path: QK complete, Softmax/SFU, then sV. |
| Cache eviction boundary | The paper includes voting-based KV cache eviction, but Stage 8 explicitly excludes eviction. | Sections III and V, Figure 3, Figure 7, PDF pp. 3 and 5 | High for paper intent; out of scope for Stage 8 | Preserve Stage 5 cache-full behavior and all-head atomic commit. No eviction logic is added. |

## Unspecified By The Paper

The following details are not sufficiently specified in the paper text or
figures and are therefore repository design decisions:

- RTL ready/valid timing and backpressure microarchitecture.
- Reset, abort, and in-flight transaction clearing rules.
- Exact pipeline latency of FP wrappers and PE forwarding paths.
- Exact bit-level FP invalid, NaN, Inf, signed-zero, RNE, FTZ, and saturation rules.
- Group-level scheduling between the two 8x8 PE groups.
- Metadata propagation format and status bit encoding.
- Performance counter definitions.
- Testbench vector format and assertion names.
