# KV Cache Implications

Query date: 2026-07-13

Formula used:

```text
KV bytes per token per layer =
    2 * num_key_value_heads * head_dim * bytes_per_element
```

The leading 2 accounts for K and V. FP16/BF16 uses 2 bytes; FP32 uses 4 bytes.

## KV Size Table

| Model | Layers | Q Heads | KV Heads | Head dim | Context | FP16/BF16 per token per layer | FP16/BF16 full context |
|---|---:|---:|---:|---:|---:|---:|---:|
| `facebook/opt-125m` | 12 | 12 | 12 | 64 | 2048 | 3072 B | 72 MiB |
| `roneneldan/TinyStories-1M` | 8 | 16 | 16 | 4 | 2048 | 256 B | 4 MiB |
| `HuggingFaceTB/SmolLM2-135M` | 30 | 9 | 3 | 64 | 8192 | 768 B | 180 MiB |
| `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | 22 | 32 | 4 | 64 | 2048 | 1024 B | 44 MiB |
| `Qwen/Qwen2.5-0.5B` | 24 | 14 | 2 | 64 | 32768 | 512 B | 384 MiB |
| `Qwen/Qwen3-0.6B` | 28 | 16 | 8 | 128 | 40960 config | 4096 B | 4480 MiB |
| `meta-llama/Llama-2-7b-hf` | 32 | 32 | 32 | 128 | 4096 | 8192 B | 1024 MiB |

Generated estimates for non-gated configs are in
`reports/model_selection/generated/kv_cache_estimates.csv`.

## Current RTL Cache Semantics

The current Stage 5/6/7 stack appends one K/V vector for every successful token
and increments `valid_seq_len` only after the all-head atomic commit. Cache-full
prevents new commit. There is no eviction pointer, no position table, no
per-layer cache selection, and no GQA/MQA mapping.

## MHA vs GQA vs MQA

Standard MHA:

- `num_key_value_heads == num_attention_heads`.
- Current RTL stores K/V per attention head.
- Eviction can be indexed by token position and head, then aggregated by layer
  or all heads depending on the algorithm.

GQA:

- `num_key_value_heads < num_attention_heads`.
- Each KV head is shared by a group of Q heads.
- Cache storage shrinks, but attention computation must map every Q head to its
  owning KV head.
- Current RTL cannot simply treat `N_HEAD` as both Q and KV heads.
- Voting can be accumulated by Q head but eviction must remove the shared KV
  token for all Q heads mapped to that KV head. Per-Q-head eviction would
  duplicate or fragment shared KV storage.

MQA:

- `num_key_value_heads == 1`.
- All Q heads share one K/V stream per layer.
- Cache storage is smallest, but per-head eviction is mostly a score aggregation
  problem; the actual storage eviction is one shared token slot per layer.

For all GQA/MQA models, a hardware cache address should be keyed by
`layer, kv_head, logical_position`, while a score/vote path may retain
`q_head` provenance before reducing to a token eviction decision.

## RoPE and Eviction Position Semantics

RoPE rotates Q and K using the logical position at which a token appears in the
sequence. After a token is evicted, remaining cached K vectors must keep their
original logical position. Re-numbering cache slots from 0 after eviction is
not equivalent; it changes the phase used by future Q/K dot products and
therefore changes attention scores.

Required metadata for RoPE-compatible eviction:

- A physical cache slot index.
- The original logical token position for each cached K/V entry.
- A monotonically increasing next-token logical position.
- For sliding-window or compressed caches, a mapping from physical order to
  logical position.
- Optional per-layer position table if layers can evict different tokens.

For current RTL, RoPE support would need to happen before K cache commit and
future query evaluation. Eviction must update physical slot validity without
rewriting RoPE logical position.

## Voting Eviction Implications

The VEDA paper's voting engine computes thresholds from attention scores, votes
for unimportant KV entries, maintains vote counts, and selects the highest vote
count for eviction, with earliest-position tie break. The paper states the
voting engine operates layer-wise and aggregates/averages all heads.

For this project:

- Current RTL has no vote count storage.
- Current Stage 5 emits per-head attention output but does not expose a stable
  full attention probability stream as an eviction interface.
- Implementing VEDA-style voting requires softmax probability observation,
  vote counters, reserved-stage logic, threshold mean/std computation, eviction
  index selection, and cache compaction or indirection.
- GQA models should aggregate votes across Q heads but evict shared KV slots.
- MHA models are simpler because Q and KV head counts match.

## Eviction Candidate Suitability

| Candidate | Suitability | Reason |
|---|---|---|
| `Qwen/Qwen2.5-0.5B` | Best | Long context, Apache-2.0, small enough, GQA stresses shared-KV eviction. |
| `Qwen/Qwen3-0.6B` | Very good | Newer Qwen, GQA ratio 2:1, but head_dim=128 and very large full cache. |
| `TinyLlama-1.1B` | Good | Llama-like and close to VEDA family; larger but context 2048 keeps cache moderate. |
| `SmolLM2-135M` | Good for prototyping | Smallest Llama-like; context 8192 and GQA useful. |
| `Llama-2-7B` | Paper reference only | Most comparable to VEDA paper but gated and too large for RTL. |
| `OPT-125M` | Limited | MHA and no RoPE simplify hardware, but it does not test Llama-style eviction issues. |

## Recommendation For Cache Design

Use a two-level view:

1. Hardware cache storage: `layer, kv_head, physical_slot -> {K, V, logical_pos}`.
2. Eviction scoring: optional `layer, q_head, logical_pos -> vote/score`, reduced
   to `layer, logical_pos` or global `logical_pos` depending on the algorithm.

Do not bake per-Q-head K/V storage into future GQA/MQA designs.
