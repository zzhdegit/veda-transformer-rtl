# Candidate Longlist

Generated support files:

- `reports/model_selection/generated/candidate_matrix.csv`
- `reports/model_selection/generated/kv_cache_estimates.csv`
- `reports/model_selection/generated/candidate_scores.csv`

All generated files are metadata only; no model weights or tokenizer caches are
stored.

| Model | Route | Detail | Type | D_MODEL | Heads | KV Heads | D_HEAD | D_FFN/int | Context | Act | Norm | License status | Notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `facebook/opt-125m` | A | Yes | OPT | 768 | 12 | 12 | 64 | 3072 | 2048 | ReLU | LayerNorm | HF: other | Closest off-the-shelf Route A shape except norm/bias/positions. |
| `roneneldan/TinyStories-1M` | A | Yes | GPT-Neo | 64 | 16 | 16 | 4 | 256 inferred | 2048 | GELU-new | LayerNorm | Not declared | Very small trained checkpoint; local attention mismatch. |
| `roneneldan/TinyStories-3M` | A | No | GPT-Neo | 128 | 16 | 16 | 8 | 512 inferred | 2048 | GELU-new | LayerNorm | Not declared | Same family as 1M; larger. |
| `roneneldan/TinyStories-8M` | A | No | GPT-Neo | 256 | 16 | 16 | 16 | 1024 inferred | 2048 | GELU-new | LayerNorm | Not declared | Exceeds current RMSNorm checked D_MODEL range. |
| `roneneldan/TinyStories-33M` | A | No | GPT-Neo | 768 | 16 | 16 | 48 | 3072 inferred | 2048 | GELU-new | LayerNorm | Not declared | Heavier TinyStories checkpoint. |
| `HuggingFaceTB/SmolLM-135M` | B | No | Llama | 576 | 9 | 3 | 64 | 1536 | 2048 | SiLU/SwiGLU | RMSNorm | Apache-2.0 | Older SmolLM baseline. |
| `HuggingFaceTB/SmolLM2-135M` | B | Yes | Llama | 576 | 9 | 3 | 64 | 1536 | 8192 | SiLU/SwiGLU | RMSNorm | Apache-2.0 | Best small Llama-like candidate. |
| `HuggingFaceTB/SmolLM2-360M` | B | No | Llama | 960 | 15 | 5 | 64 | 2560 | 8192 | SiLU/SwiGLU | RMSNorm | Apache-2.0 | Same family, larger. |
| `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | B | Yes | Llama | 2048 | 32 | 4 | 64 | 5632 | 2048 | SiLU/SwiGLU | RMSNorm | Apache-2.0 | Strong VEDA/Llama-style research candidate but large. |
| `Qwen/Qwen2.5-0.5B` | B | Yes | Qwen2 | 896 | 14 | 2 | 64 | 4864 | 32768 | SiLU/SwiGLU | RMSNorm | Apache-2.0 | Strong eviction candidate; QKV bias. |
| `Qwen/Qwen3-0.6B` | B | Yes | Qwen3 | 1024 | 16 | 8 | 128 | 3072 | 40960 cfg / 32768 card | SiLU/SwiGLU | RMSNorm | Apache-2.0 | Newer Qwen small model; high KV cost. |
| `EleutherAI/pythia-70m-deduped` | A | No | GPT-NeoX | 512 | 8 | 8 | 64 | 2048 | 2048 | GELU | LayerNorm | Apache-2.0 | Small and open but partial RoPE/parallel residual. |
| `EleutherAI/pythia-160m-deduped` | A | No | GPT-NeoX | 768 | 12 | 12 | 64 | 3072 | 2048 | GELU | LayerNorm | Apache-2.0 | Larger Pythia. |
| `distilgpt2` | A | No | GPT-2 | 768 | 12 | 12 | 64 | not explicit | 1024 | GELU-new | LayerNorm | Apache-2.0 | Mature but far from RTL norm/activation. |
| `meta-llama/Llama-2-7b-hf` | B | Yes | Llama 2 | 4096 from card/paper | 32 | 32 | 128 | 11008 from paper | 4096 | SiLU/SwiGLU | RMSNorm | Llama 2 gated/custom | VEDA paper-family reference; not a deployment target. |

Notes:

- `D_FFN/int` for GPT-Neo entries is inferred from Transformers GPT-Neo
  default behavior when `intermediate_size` is null. It is not overwritten in
  the generated CSV.
- `meta-llama/Llama-2-7b-hf` raw config was not accessible without account
  approval; dimensions are from Meta/HF model card and Llama 2 paper-family
  documentation, not from raw `config.json`.
