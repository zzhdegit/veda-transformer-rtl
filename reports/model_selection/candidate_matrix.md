# Candidate Matrix

Query date: 2026-07-13

Scoring weights:

| Criterion | Weight |
|---|---:|
| Current RTL compatibility | 25 |
| Model size and simulation feasibility | 20 |
| Weight/tokenizer availability | 10 |
| Architecture/source clarity | 10 |
| License clarity | 10 |
| KV eviction research value | 10 |
| VEDA-paper structural closeness | 10 |
| Community/tool support | 5 |

## Score Summary

| Rank | Model | Score | Primary role |
|---:|---|---:|---|
| 1 | `HuggingFaceTB/SmolLM2-135M` | 78 | Small Llama-like backup / future hardware path |
| 2 | `Qwen/Qwen3-0.6B` | 74 | KV eviction research |
| 3 | `Qwen/Qwen2.5-0.5B` | 73 | KV eviction research |
| 4 | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | 68 | VEDA/Llama reference below 2B |
| 5 | `EleutherAI/pythia-70m-deduped` | 67 | open GPT-NeoX fallback |
| 6 | `facebook/opt-125m` | 65 | fastest existing-model RTL-adaptation path |
| 7 | `distilgpt2` | 61 | mature GPT-2 tooling baseline |
| 8 | `roneneldan/TinyStories-1M` | 57 | fastest tiny software/RTL smoke path |
| 9 | `meta-llama/Llama-2-7b-hf` | 48 | VEDA paper reference only |

## Detailed Candidate Records

### `facebook/opt-125m`

- Revision: `27dcfa74d334bc871f3234de431e71c6eeba5dd6`
- Last modified: 2023-09-15
- Type: decoder-only OPT causal LM.
- Parameters/checkpoint: 125M family; API checkpoint bytes about 250.5 MB.
- Tokenizer/vocab/context: GPT-2 byte-level BPE style, vocab 50272, context
  2048.
- Training: pretrained CLM on OPT corpus; model card reports 180B tokens for
  the OPT training data mixture.
- License: HF `other`; not Apache. Treat as research-usable only after reading
  the OPT license terms.
- Structure: 12 layers, hidden 768, 12 MHA heads, head_dim 64, FFN 3072,
  activation ReLU, Pre-LN (`do_layer_norm_before=true`), learned absolute
  position embedding, tied embeddings not set in config.
- Main mismatch: LayerNorm, attention/MLP bias, learned positions, hidden size
  exceeds current RMSNorm implementation range.
- Compatibility level: Level 2 for meaningful single-layer fidelity
  (LayerNorm+bias); Level 1 if using software to feed already-normalized hidden
  states into MHA/FFN-only tests.

### `roneneldan/TinyStories-1M`

- Revision: `77f1b168e219585646439073245fe87e56b3023e`
- Last modified: 2025-12-18
- Type: GPT-Neo causal LM.
- Parameters/checkpoint: advertised 1M; API checkpoint bytes about 48.6 MB.
- Tokenizer/vocab/context: GPT-2 tokenizer files present, vocab 50257,
  context 2048.
- Training: TinyStories dataset/model-card family.
- License: not declared in HF card/API. Do not use as primary until licensing
  is resolved.
- Structure: 8 layers, hidden 64, 16 heads, head_dim 4, inferred 4D MLP,
  GELU-new, LayerNorm epsilon 1e-5, alternating global/local attention with
  window size 256.
- Main mismatch: LayerNorm, GELU, GPT-Neo local attention, bias, no RMSNorm,
  head_dim 4 underuses the 8-wide PE lane design.
- Compatibility level: Level 3 because local attention semantics are not the
  current append-only full-cache MHA.

### `HuggingFaceTB/SmolLM2-135M`

- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Last modified: 2025-02-06
- Type: Llama causal LM.
- Parameters/checkpoint: 135M family; API checkpoint bytes about 269.1 MB.
- Tokenizer/vocab/context: tokenizer JSON present, vocab 49152, context 8192.
- Training: model card reports Transformer decoder, 2T pretraining tokens,
  BF16 precision, 64 H100 GPUs.
- License: Apache-2.0.
- Structure: 30 layers, hidden 576, 9 Q heads, 3 KV heads, head_dim 64,
  intermediate 1536, RMSNorm epsilon 1e-5, RoPE theta 100000,
  SiLU/SwiGLU-style gate/up/down MLP, no attention bias, tied embeddings.
- Main mismatch: RoPE, GQA, SwiGLU/gated FFN, variable intermediate size,
  multiple layers, dimensions not in current checked RTL set.
- Compatibility level: Level 3.

### `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`

- Revision: `59f6f375b26bde864a6ca194a9a3044570490064`
- Last modified: 2024-09-27
- Type: Llama causal LM.
- Parameters/checkpoint: model card reports 1B params; API checkpoint bytes
  about 8.80 GB because this checkpoint is stored as F32.
- Tokenizer/vocab/context: Llama tokenizer files present, vocab 32000,
  context 2048.
- Training: project states 1.1B Llama model on 3T tokens; card says same
  architecture/tokenizer as Llama 2.
- License: Apache-2.0.
- Structure: 22 layers, hidden 2048, 32 Q heads, 4 KV heads, head_dim 64,
  intermediate 5632, RMSNorm epsilon 1e-5, SiLU/SwiGLU-style MLP, RoPE,
  untied embeddings.
- Main mismatch: RoPE, GQA, SwiGLU/gated FFN, large hidden/FFN, multiple
  layers, F32 checkpoint.
- Compatibility level: Level 3 for single-layer study, Level 4 for full RTL.

### `Qwen/Qwen2.5-0.5B`

- Revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Last modified: 2024-09-25
- Type: Qwen2 causal LM.
- Parameters/checkpoint: model card reports 0.49B, 0.36B non-embedding;
  API checkpoint bytes about 988.1 MB.
- Tokenizer/vocab/context: tokenizer JSON present, vocab 151936, full context
  32768.
- Training: pretraining base model; Qwen2.5 family model card.
- License: Apache-2.0.
- Structure: 24 layers, hidden 896, 14 Q heads, 2 KV heads, head_dim 64,
  intermediate 4864, RMSNorm epsilon 1e-6, RoPE theta 1,000,000, SwiGLU,
  QKV attention bias, tied embeddings.
- Main mismatch: QKV bias, RoPE, GQA, SwiGLU/gated FFN, very large vocab and
  long context.
- Compatibility level: Level 3.

### `Qwen/Qwen3-0.6B`

- Revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Last modified: 2025-07-26
- Type: Qwen3 causal LM.
- Parameters/checkpoint: model card reports 0.6B, 0.44B non-embedding; API
  checkpoint bytes about 1.50 GB.
- Tokenizer/vocab/context: tokenizer JSON present, vocab 151936; config
  `max_position_embeddings=40960`, model card states context length 32768.
- Training: pretraining and post-training according to the model card.
- License: Apache-2.0.
- Structure: 28 layers, hidden 1024, 16 Q heads, 8 KV heads, head_dim 128,
  intermediate 3072, RMSNorm epsilon 1e-6, RoPE theta 1,000,000, SwiGLU,
  no attention bias in config, tied embeddings.
- Main mismatch: RoPE, GQA, SwiGLU/gated FFN, head_dim 128, huge context.
- Compatibility level: Level 3.

### `meta-llama/Llama-2-7b-hf`

- Revision/API metadata: `01c7f73d771dfac7d292323805ebc428287df4f9`
- Raw config access: not available without account approval on 2026-07-13.
- Type: Llama 2 pretrained 7B.
- Parameters/context: model card reports 7B, 4k context, no GQA for 7B.
- Tokenizer/weights: gated; must share contact information with Meta and
  accept the Llama 2 Community License.
- License: custom Llama 2 license.
- Structure from Llama 2 family and Transformers Llama implementation:
  decoder-only, RMSNorm, RoPE, SwiGLU gate/up/down MLP, standard MHA for 7B,
  no QKV bias in common Llama config, final RMSNorm and bias-free LM head.
- Main mismatch: RoPE, SwiGLU/gated FFN, D_MODEL 4096, D_HEAD 128,
  D_FFN 11008, 32 layers, license/gating.
- Compatibility level: Level 4 for this RTL project as a full model; Level 3
  for software eviction experiments or a single-layer architectural reference.

## Goal-Specific Rankings

Fastest real RTL simulation path:

1. Hardware-matched Route C training fallback.
2. `facebook/opt-125m`, if LayerNorm and bias are added or bypassed at the
   layer boundary.
3. `roneneldan/TinyStories-1M`, only for tiny smoke tests after resolving
   license and local-attention/GELU handling.

Best path toward VEDA/Llama structure:

1. `HuggingFaceTB/SmolLM2-135M`
2. `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
3. `Qwen/Qwen3-0.6B`

Best KV eviction research path:

1. `Qwen/Qwen2.5-0.5B`
2. `Qwen/Qwen3-0.6B`
3. `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
4. `HuggingFaceTB/SmolLM2-135M`

Qwen2.5 is ranked first for eviction because it is Apache-2.0, has official
0.5B base weights, long context, strong tooling, and a high GQA compression
ratio. It is not the fastest RTL model.
