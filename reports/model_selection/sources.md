# Sources

Access date for online sources: 2026-07-13.

## Local Repository Sources

| Source | Type | Fields used |
|---|---|---|
| `docs/stage_07/spec.md` | Local authoritative spec | Stage 7 layer order, numeric boundaries, excluded features. |
| `reports/stage_07/acceptance_audit.md` | Local audit | accepted scope, hierarchy, reset/backpressure, limitations. |
| `reports/stage_06/acceptance_audit.md` | Local audit | Stage 6 MHA/KV/cache semantics. |
| `rtl/transformer/transformer_layer.sv` | RTL | top parameters, instances, interfaces, token order. |
| `rtl/transformer/rmsnorm_engine.sv` | RTL | RMSNorm operation, D_MODEL support, epsilon. |
| `rtl/transformer/ffn_engine.sv` | RTL | W1/W2 layout, ReLU, no bias, FP16/FP32 boundaries. |
| `rtl/attention/projection_integrated_mha.sv` | RTL | QKV/WO layout, concat, Stage 5 cache handoff. |
| `model/transformer/*` | Bit model | trace nodes and reference operation order. |
| `model/projection/*` | Bit model | projection and MHA reference behavior. |

## Generated Metadata

| Source | Type | Fields used |
|---|---|---|
| `reports/model_selection/generated/candidate_matrix.csv` | Generated CSV | revisions, config fields, API metadata. |
| `reports/model_selection/generated/kv_cache_estimates.csv` | Generated CSV | KV byte estimates. |
| `reports/model_selection/generated/candidate_scores.csv` | Generated CSV | reproducible score components. |

## Model Cards, Configs, And APIs

| Model | Official page/title | Source type | Fields used | URL |
|---|---|---|---|---|
| `facebook/opt-125m` | `facebook/opt-125m - Hugging Face` | Model card/API/config | license field, training data, tokenizer, context, revision, architecture config | https://huggingface.co/facebook/opt-125m |
| `facebook/opt-125m` | `config.json` | Raw config | hidden size, heads, layers, FFN, activation, Pre-LN flag | https://huggingface.co/facebook/opt-125m/raw/main/config.json |
| `roneneldan/TinyStories-1M` | `roneneldan/TinyStories-1M - Hugging Face` | Model card/API/config | GPT-Neo type, tokenizer, dataset tag, revision, no declared license | https://huggingface.co/roneneldan/TinyStories-1M |
| `roneneldan/TinyStories-1M` | `config.json` | Raw config | hidden size, layers, heads, attention_types, window size, activation | https://huggingface.co/roneneldan/TinyStories-1M/raw/main/config.json |
| `HuggingFaceTB/SmolLM2-135M` | `HuggingFaceTB/SmolLM2-135M - Hugging Face` | Model card/API/config | license, training tokens, BF16, hardware, revision | https://huggingface.co/HuggingFaceTB/SmolLM2-135M |
| `HuggingFaceTB/SmolLM2-135M` | `config.json` | Raw config | Llama config, RMSNorm, GQA, RoPE, vocab, context | https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json |
| `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `TinyLlama-1.1B - Hugging Face` | Model card/API/config | model purpose, 3T token checkpoint, Llama 2 architecture/tokenizer statement, license/API | https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T |
| `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `config.json` | Raw config | layers, hidden, Q/KV heads, intermediate, activation, RMSNorm | https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/raw/main/config.json |
| `Qwen/Qwen2.5-0.5B` | `Qwen/Qwen2.5-0.5B - Hugging Face` | Model card/API/config | Apache-2.0, architecture bullet, params, GQA heads, context | https://huggingface.co/Qwen/Qwen2.5-0.5B |
| `Qwen/Qwen2.5-0.5B` | `config.json` | Raw config | hidden, layers, Q/KV heads, intermediate, RoPE theta, RMSNorm eps | https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json |
| `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-0.6B - Hugging Face` | Model card/API/config | Apache-2.0, training stage, params, layers, GQA heads, context | https://huggingface.co/Qwen/Qwen3-0.6B |
| `Qwen/Qwen3-0.6B` | `config.json` | Raw config | hidden, layers, Q/KV heads, head_dim, RoPE theta, attention_bias | https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json |
| `EleutherAI/pythia-70m-deduped` | `EleutherAI/pythia-70m-deduped - Hugging Face` | Model card/API/config | parameter count, dataset, license/API, revision | https://huggingface.co/EleutherAI/pythia-70m-deduped |
| `EleutherAI/pythia-70m-deduped` | `config.json` | Raw config | GPT-NeoX dimensions, GELU, partial RoPE, parallel residual | https://huggingface.co/EleutherAI/pythia-70m-deduped/raw/main/config.json |
| `distilgpt2` | `distilgpt2 - Hugging Face` | Model card/API/config | GPT-2 baseline metadata, Apache-2.0, revision | https://huggingface.co/distilgpt2 |
| `distilgpt2` | `config.json` | Raw config | GPT-2 layer/head dimensions and activation | https://huggingface.co/distilgpt2/raw/main/config.json |
| `meta-llama/Llama-2-7b-hf` | `meta-llama/Llama-2-7b-hf - Hugging Face` | Model card/API/license | gated access, license, 7B, context, no GQA for 7B, training data, model dates | https://huggingface.co/meta-llama/Llama-2-7b-hf |

## Official Implementation And Papers

| Source | Type | Fields used | URL |
|---|---|---|---|
| Transformers Llama source | Official implementation | RMSNorm, RoPE, gate/up/down MLP, GQA repeat, pre-norm residual order, bias flags, final norm/LM head | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py |
| Transformers Qwen2 docs | Official docs | Qwen2 family uses GQA, RoPE, sliding/full attention; config class fields | https://huggingface.co/docs/transformers/en/model_doc/qwen2 |
| Transformers Qwen2 source | Official implementation | Qwen2 MLP gate/up/down, RoPE, KV repeat | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen2/modeling_qwen2.py |
| Transformers OPT docs | Official docs | OPT is decoder-only causal LM family, example model support | https://huggingface.co/docs/transformers/en/model_doc/opt |
| VEDA paper PDF | Official paper/arXiv | Voting eviction algorithm, Llama-2 evaluation, 8x8x2 PE array, layer-wise head aggregation | https://arxiv.org/pdf/2507.00797 |
| VEDA arXiv HTML | Official paper/arXiv HTML | same as above, searchable text | https://arxiv.org/html/2507.00797v1 |
| Llama 2 paper/model card | Official Meta/HF paper-card source | 7B/13B/70B family, context, GQA only for 70B, license and gated access | https://huggingface.co/meta-llama/Llama-2-7b-hf |

## Source Gaps

- `meta-llama/Llama-2-7b-hf` raw `config.json` returned unauthorized without
  account approval. This report therefore does not claim a config revision for
  that file; it uses the model card, paper family details, and Transformers
  Llama implementation for architectural analysis.
- TinyStories model cards/API do not declare a license. The license state is
  recorded as "not declared" rather than inferred.
