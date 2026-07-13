# Final Recommendation

Query date: 2026-07-13

## Primary Model

Primary for fastest real-checkpoint RTL adaptation:

```text
facebook/opt-125m
revision: 27dcfa74d334bc871f3234de431e71c6eeba5dd6
```

Rationale:

- It is the closest public trained model to the current non-Llama parts of the
  RTL: decoder-only, causal LM, Pre-LN, standard MHA, ReLU FFN, and 4D FFN.
- It has official Transformers support and public model/card/config metadata.
- It avoids RoPE, SwiGLU, and GQA, so the smallest meaningful hardware changes
  are LayerNorm and bias rather than a full Llama-like rewrite.

RTL match:

- Match: decoder-only, Pre-LN order, standard MHA, ReLU FFN, `D_FFN=4D`,
  context 2048, public tokenizer/weights.
- Mismatch: LayerNorm not RMSNorm, linear biases, learned absolute position
  embeddings, D_MODEL=768 not currently supported by RMSNorm constants, multiple
  layers, embedding/final norm/LM head outside RTL.

Required hardware changes:

- Add a LayerNorm path or define a software boundary that feeds already-normalized
  hidden states into RTL.
- Add optional Q/K/V/O and FFN bias support for faithful full-layer comparison.
- Extend model dimensions and RMSNorm/LayerNorm constants if validating a full
  OPT layer in RTL.

Immediate simulation range:

- Single MHA real-weight validation is conditionally feasible.
- Single FFN real-weight validation is conditionally feasible because ReLU/4D
  match.
- Full one-layer validation is conditional on LayerNorm+bias support.

Not feasible now:

- Exact current Stage 7 run with OPT full-layer weights without changing or
  bypassing norm/bias semantics.
- Multi-layer full RTL simulation.

KV cache/eviction:

- Uses standard MHA, so it is useful for initial append-only cache checks but
  not ideal for Llama-style GQA/RoPE eviction research.

## Backup Model

Backup for future Llama-like hardware adaptation:

```text
HuggingFaceTB/SmolLM2-135M
revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
```

Rationale:

- Smallest detailed Llama-style candidate with Apache-2.0 license, public
  safetensors, official model card, and clear config.
- Matches RMSNorm and no attention bias.
- Much smaller than TinyLlama and vastly smaller than Llama-2-7B.

RTL match:

- Match: decoder-only, Pre-Norm Llama layer order, RMSNorm, no attention bias,
  head_dim 64, public tokenizer/weights.
- Mismatch: RoPE, GQA with 9 Q heads / 3 KV heads, SwiGLU gate/up/down FFN,
  intermediate_size 1536 not 4D, D_MODEL=576, 30 layers.

Required hardware changes:

- RoPE before attention.
- GQA cache and Q-head to KV-head mapping.
- SwiGLU with gate/up/down matrices and SiLU.
- Variable `D_FFN`.
- Multi-layer software/RTL orchestration for anything beyond one layer.

Immediate simulation range:

- Software extraction and single-layer software/RTL co-sim planning are feasible.
- Full RTL is not realistic at the current project stage.

KV cache/eviction:

- Good prototype for GQA-aware token eviction with much lower cache size than
  MHA models.

## Hardware-Matched Training Fallback

Recommended fallback:

- Architecture: decoder-only, Pre-Norm, RMSNorm, standard MHA, ReLU W1/W2,
  no bias, `D_FFN=4D`.
- Dimensions: D_MODEL=64, N_HEAD=8, D_HEAD=8, D_FFN=256.
- Layer count: 2 initially; 1 and 4 as ablation points.
- Context: 128 initially, 256 for eviction experiments.
- Tokenizer: character-level for fastest reproducibility, or small BPE with
  1k-8k vocab if TinyStories text quality matters.
- Dataset: TinyStories for meaningful short generation; Tiny Shakespeare for
  fastest debug-only training.
- Estimated parameters: about 0.7M to 1.5M with vocab/head tying choices.
- Training effort: minutes to hours on one consumer GPU for character-level or
  tiny BPE; several hours to about a day for a better TinyStories run.

Why keep it:

- It is the only Level 0 path to a trained model exactly matching current RTL.
- It enables bit-exact full-layer RTL validation before committing to RoPE,
  SwiGLU, GQA, or LayerNorm hardware.
- It can still support perplexity trends and simple eviction experiments,
  although it will not represent modern LLM quality.

## Two-Model Strategy

Recommended: yes.

Use two targets instead of forcing one model to serve every purpose:

- Model A for fastest RTL deployment path: hardware-matched Route C model, with
  `facebook/opt-125m` as the closest existing-checkpoint adaptation target once
  LayerNorm and bias are in scope.
- Model B for VEDA/eviction research: `Qwen/Qwen2.5-0.5B` for long-context GQA
  eviction experiments, with `HuggingFaceTB/SmolLM2-135M` as the smaller
  Llama-like hardware adaptation backup.

## Answers To Required Questions

1. Already trained direct-match model:
   no. None of the investigated checkpoints directly matches current
   `RMSNorm + MHA + ReLU FFN + no bias + D_FFN=4D + no RoPE/GQA`.

2. Minimum-modification existing model:
   `facebook/opt-125m`, because ReLU, Pre-LN, MHA, and 4D FFN match. It still
   requires LayerNorm and bias support or a documented software boundary.

3. Llama-like adaptation vs self-training:
   for immediate RTL validation, self-training is more reasonable. For VEDA
   publication-aligned architecture and eviction research, adapt a small
   Llama-like model.

4. VEDA paper model suitability:
   VEDA evaluates the Llama-2 family, especially Llama-2-7B style dimensions and
   8x8x2 PE mapping. It is suitable for software eviction studies and single
   layer architectural reference. It is not suitable for complete RTL simulation
   in the current project because it is gated, 7B scale, RoPE/SwiGLU, and
   32-layer.

5. Best model for Voting eviction:
   `Qwen/Qwen2.5-0.5B` for practical experiments; `meta-llama/Llama-2-7b-hf`
   only as paper-reference if access/license is approved.

6. GQA/MQA cache impact:
   cache storage must be keyed by KV head, not Q head. Q-head votes can be
   accumulated, but eviction removes shared KV slots for all Q heads mapped to
   that KV head.

7. RoPE eviction position:
   retained tokens must keep original logical positions. Do not renumber after
   eviction; store logical position metadata per physical slot.

8. 8x8x2 mapping:
   most investigated hidden/head/intermediate dimensions are multiples of 8, so
   tail tiles are not the main issue. The main issue is absolute size,
   unsupported D_MODEL constants, RoPE/GQA/SwiGLU, and multi-layer orchestration.

9. Tail/utilization risks:
   TinyStories-1M has D_HEAD=4, underusing 8-wide lanes. Qwen2.5 has 14 Q heads,
   which creates non-power-of-two head-loop/control handling. Qwen3/Llama-2
   D_HEAD=128 increases per-head cycle cost.

10. One model or two:
    choose two. A deployment/RTL-validation model and an eviction research model
    optimize different constraints.

## Freeze Recommendation

Do not freeze a single universal target model yet. Freeze two provisional
targets:

- RTL validation target: Route C strict-match tiny model.
- External checkpoint adaptation target: `facebook/opt-125m`, pending
  LayerNorm+bias approval.

For eviction research, freeze `Qwen/Qwen2.5-0.5B` only after the user confirms
that RoPE/GQA/SwiGLU software experiments are in scope and the RTL path is not
expected to run the full model.

## User Decisions Still Needed

- Whether Stage 8 should prioritize LayerNorm+bias for OPT or RoPE/GQA/SwiGLU
  for Llama-like models.
- Whether to approve training a tiny hardware-matched checkpoint.
- Whether to accept a two-model strategy as the project baseline.
- Whether Llama-2 gated access/license should be pursued for paper comparison.
