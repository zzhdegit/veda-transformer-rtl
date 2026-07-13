# Stage M1 Search Log

Query date: 2026-07-13

## Search Keywords

- `VEDA Voting eviction KV cache LLM`
- `VEDA Efficient LLM Generation Through Voting-based KV Cache Eviction`
- `TinyLlama 1.1B config Llama architecture Hugging Face`
- `SmolLM2 135M config Hugging Face`
- `Qwen2.5 0.5B config GQA RoPE SwiGLU`
- `Qwen3 0.6B config GQA RoPE`
- `TinyStories 1M GPT-Neo config`
- `OPT-125M config ReLU do_layer_norm_before`
- `Pythia 70M GPT-NeoX config`
- `distilgpt2 config GPT2`
- `small decoder only pretrained language model ReLU MLP`
- `RMSNorm ReLU pretrained decoder model`

## Source Types Checked

- Hugging Face model cards.
- Hugging Face model API with `?blobs=true`.
- Hugging Face raw `config.json`.
- Official Transformers model docs and source files.
- VEDA paper PDF and arXiv HTML.
- Llama 2 model card and license text.

## Longlist Summary

Fifteen real model IDs were checked:

1. `facebook/opt-125m`
2. `roneneldan/TinyStories-1M`
3. `roneneldan/TinyStories-3M`
4. `roneneldan/TinyStories-8M`
5. `roneneldan/TinyStories-33M`
6. `HuggingFaceTB/SmolLM-135M`
7. `HuggingFaceTB/SmolLM2-135M`
8. `HuggingFaceTB/SmolLM2-360M`
9. `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
10. `Qwen/Qwen2.5-0.5B`
11. `Qwen/Qwen3-0.6B`
12. `EleutherAI/pythia-70m-deduped`
13. `EleutherAI/pythia-160m-deduped`
14. `distilgpt2`
15. `meta-llama/Llama-2-7b-hf`

## Detailed Comparison Set

The detailed set is:

- `facebook/opt-125m`
- `roneneldan/TinyStories-1M`
- `HuggingFaceTB/SmolLM2-135M`
- `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen3-0.6B`
- `meta-llama/Llama-2-7b-hf`

This is seven detailed candidates, exceeding the required five.

## Checked But Not Detailed

| Model | Reason not detailed |
|---|---|
| `roneneldan/TinyStories-3M` | Same GPT-Neo structural mismatches as TinyStories-1M, larger D_MODEL. |
| `roneneldan/TinyStories-8M` | Same mismatch family; D_MODEL=256 exceeds current RMSNorm checked range. |
| `roneneldan/TinyStories-33M` | Same mismatch family; D_MODEL=768 is much heavier for RTL. |
| `HuggingFaceTB/SmolLM-135M` | Superseded for this study by SmolLM2-135M with longer context and newer card. |
| `HuggingFaceTB/SmolLM2-360M` | Same architecture as SmolLM2-135M but larger. |
| `EleutherAI/pythia-70m-deduped` | Useful fallback, but GPT-NeoX parallel residual/GELU/partial RoPE differs more than OPT for Route A. |
| `EleutherAI/pythia-160m-deduped` | Same family as Pythia-70M, larger. |
| `distilgpt2` | Mature tooling, but GPT-2 LayerNorm/GELU/bias/learned positions are far from current RTL. |

## Negative Result

No public trained checkpoint was found that nearly directly maps to the current
`RMSNorm + standard MHA + ReLU W1/W2 + no bias + D_FFN=4D + no RoPE/GQA`
hardware contract. The closest existing trained models split into two groups:

- OPT-style models match ReLU, Pre-LN, MHA, and 4D FFN but use LayerNorm,
  learned absolute positions, and bias.
- Llama/Qwen/SmolLM-style models match RMSNorm and no/low bias but require
  RoPE, SwiGLU/SiLU, variable FFN width, and GQA/MQA in many cases.
