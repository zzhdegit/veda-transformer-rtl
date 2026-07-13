# Hardware Stage H9 Paper Schedule Evidence

Hardware Stage H9 uses the local repository paper copy:

- `2507.00797v1.pdf`
- Title: VEDA: Efficient LLM Generation Through Voting-based KV Cache Eviction and Dataflow-flexible Accelerator
- Version: arXiv v1
- Paper header date: 1 Jul 2025

## Evidence Freeze

| Item | Paper evidence | H9 interpretation |
|---|---|---|
| Score production granularity | Section IV-A and Figure 5 map `q x K^T = s` to an inner-product configured PE array and describe sequence length as spatial/element output. | H9 treats each scaled score as an ordered element packet. |
| SFU consumption granularity | Section IV-B and Figure 6 describe the SFU receiving serial element-by-element output from the inner-product PE array. | H9 streams score packets into a bounded score buffer and softmax reduction engine. |
| Softmax reduction | Figure 6 decomposes softmax into max and `exp_sum`. Text says the reduction unit receives serial output, identifies tile maximum, updates maximum, stores tile in FIFO, then updates `exp_sum`; the process is similar to online normalizer calculation [10]. | H9 keeps the existing repository online reduction arithmetic and records this as paper-supported at algorithm level, not RTL-exact. |
| Normalization | Section IV-B describes normalization as element-wise subtraction, exponentiation, and division. Figure 6 shows normalization feeding outer-product PE array serially. | H9 replays stored scores after final max/exp_sum and emits ordered probability packets. |
| Raw score caching | Figure 6(c) shows a FIFO between inner-product output/reduction and later normalization. | H9 uses a bounded score replay buffer. |
| Probability output order | Section IV-B states that outer-product configuration enables element-by-element serial input into the PE array. | H9 emits probability packets in token order and maps probability index directly to V token index. |
| sV consumption order | Figure 5(b) and Figure 6(c) show `s' x V = o` as outer product with scalar broadcast. | H9 broadcasts one probability per V token and updates the output accumulator once per probability. |
| Two PE groups | Table I and Figure 7 identify an `8x8x2` PE array, but do not specify group scheduling. | Group scheduling is a repository design decision. |
| QK/SFU overlap | Figure 6 shows the PE array active while the SFU performs reduction. | H9 permits QK PE overlap with SFU reduction. |
| SFU/sV overlap | Figure 6 shows normalization and outer-product PE activity overlapped. | H9 permits SFU normalization overlap with sV PE activity after inner-product work has retired and mode switch is complete. |
| QK/sV mutual exclusion | The same PE array is reconfigured between inner and outer modes. | H9 forbids simultaneous QK and sV on one paper array. |

## Repository Design Decisions

- Ready/valid timing, packet field widths, reset behavior, FIFO depths, and assertions are repository-defined.
- The H9 native dimension mapping interleaves dimensions by group, then row, then column to avoid the H8 low-lane-only adapter.
- The repository keeps the current Stage 3 online softmax reduction and replay normalization arithmetic. The paper cites online normalizer style behavior but does not publish bit-level recurrence, wrapper latency, exception, or rounding rules.
- Score and probability FIFOs are bounded register-array correctness structures, not SRAM macro or PPA claims.
- H9 does not implement voting eviction, SRAM macros, global PE sharing, Projection/FFN migration, timing closure, or physical PPA.
