# Simulation Feasibility

## Feasibility Definitions

- Feasible: can be done with the current RTL plus weight/vector conversion or a
  small, explicitly scoped wrapper.
- Conditional: possible after named hardware/software work.
- Not realistic: incompatible with current scope or too large for this project
  phase.

## Candidate Feasibility Matrix

| Candidate | Single-layer real-weight RTL | Few-token end-to-end RTL | Full model SW-RTL co-sim | Multi-layer full RTL | KV eviction software experiment |
|---|---|---|---|---|---|
| `facebook/opt-125m` | Conditional | Conditional | Conditional | Not realistic | Conditional |
| `roneneldan/TinyStories-1M` | Conditional | Conditional | Conditional | Not realistic | Weak |
| `HuggingFaceTB/SmolLM2-135M` | Conditional | Conditional | Feasible after adapters | Not realistic | Feasible |
| `TinyLlama-1.1B` | Conditional | Not realistic for full hidden | Feasible in software | Not realistic | Feasible |
| `Qwen/Qwen2.5-0.5B` | Conditional | Not realistic for full hidden | Feasible in software | Not realistic | Feasible |
| `Qwen/Qwen3-0.6B` | Conditional | Not realistic for full hidden | Feasible in software | Not realistic | Feasible |
| `Llama-2-7B` | Not realistic | Not realistic | Conditional, gated | Not realistic | Conditional, gated |
| Route C trained model | Feasible | Feasible | Feasible | Conditional for 1-4 layers | Conditional |

## Fastest Practical RTL Path

The fastest genuine RTL path is Route C, because it can exactly match the
current layer:

- D_MODEL 64 or 128.
- N_HEAD 8 or 16 with D_HEAD 8.
- D_FFN 4D.
- RMSNorm.
- ReLU W1/W2.
- no bias.
- no RoPE/GQA.

If an existing model must be used, `facebook/opt-125m` is the closest structural
candidate because it has ReLU, 4D FFN, standard MHA, and no RoPE. It still needs
LayerNorm and bias support or a documented boundary that starts after
LayerNorm/bias effects have already been applied in software.

## 8x8x2 Array Mapping

The VEDA paper highlights an 8x8x2 reconfigurable PE array and D=8x8 spatial
mapping for attention. Current RTL uses `PE_NUM=8` lanes in checked configs.

| Model | D_HEAD | D_HEAD tail risk | D_MODEL with PE_NUM=8 | D_FFN/intermediate with PE_NUM=8 | Notes |
|---|---:|---|---:|---:|---|
| Current H2/D8 baseline | 8 | none | 16 no tail | 64 no tail | Verified. |
| Current H2/D16 baseline | 16 | none | 32 no tail | 128 no tail | Verified. |
| `TinyStories-1M` | 4 | underuses lanes | 64 no D_MODEL tail | 256 no FFN tail | Head dim smaller than 8. |
| `OPT-125M` | 64 | maps cleanly | 768 no tail | 3072 no tail | Large but no PE_NUM tail. |
| `SmolLM2-135M` | 64 | maps cleanly | 576 no tail | 1536 no tail | D_MODEL unsupported by current RMSNorm constants. |
| `TinyLlama-1.1B` | 64 | maps cleanly | 2048 no tail | 5632 no tail | Very large simulation. |
| `Qwen2.5-0.5B` | 64 | maps cleanly | 896 no tail | 4864 no tail | Q heads=14 creates non-power-of-two head loop. |
| `Qwen3-0.6B` | 128 | maps cleanly | 1024 no tail | 3072 no tail | Larger per-head dot products. |
| `Llama-2-7B` | 128 | maps cleanly | 4096 no tail | 11008 no tail | Full RTL unrealistic. |

The main utilization issue is not tail tiles for these public models; most
dimensions are multiples of 8. The main cost is absolute hidden/FFN size,
unsupported `D_MODEL` constants, and GQA/RoPE/SwiGLU control.

## Training Fallback Feasibility

Recommended fallback:

- Dataset: TinyStories for simple natural language, or Tiny Shakespeare for
  fastest character-level experiments.
- Tokenizer: character-level for strict control, or small BPE/unigram vocab
  around 1k-8k tokens.
- Architecture: 2 layers, D_MODEL=64, N_HEAD=8, D_HEAD=8, D_FFN=256,
  RMSNorm, ReLU, no bias, standard MHA.
- Context: 128 for initial training, 256 for eviction experiments.
- Parameter estimate: with vocab 4096 and tied embeddings, about 0.7M to 1.5M
  parameters depending on layer count and LM head tying.
- Training effort: minutes to a few hours on a single consumer GPU for toy
  corpora; several hours to a day for a more useful TinyStories model.
- Expected quality: enough for deterministic generation, perplexity trends,
  and RTL hidden/weight extraction; not enough to benchmark modern LLM quality.

This fallback is the only path that can produce a trained model with Level 0
current-RTL compatibility.
