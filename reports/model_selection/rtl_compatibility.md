# RTL Compatibility Study

## Current RTL Contract From Code

The following points were verified directly in RTL/Python code, not copied only
from prose documentation:

- `rtl/transformer/transformer_layer.sv` defines `D_MODEL=N_HEAD*D_HEAD` and
  `D_FFN=4*D_MODEL`.
- `transformer_layer.sv` instantiates exactly one `projection_integrated_mha`,
  two `rmsnorm_engine` instances, two `residual_add_engine` instances, and one
  `ffn_engine`.
- `rmsnorm_engine.sv` accepts FP32 inputs, FP16 gamma, uses
  `EPS_FP32=32'h3727_C5AC`, sequential fused MAC sum of squares, exact
  power-of-two mean scale, sqrt, reciprocal, `(x*inv_rms)*gamma`, then FP16
  output.
- `rmsnorm_engine.sv` currently supports only `D_MODEL` values 8, 16, 32, 64,
  and 128.
- `ffn_engine.sv` stores W1 as `[D_FFN][D_MODEL]`, W2 as
  `[D_MODEL][D_FFN]`, applies ReLU, quantizes activation to FP16, and has no
  bias inputs.
- `projection_integrated_mha.sv` loads WQ/WK/WV/WO by output/input index,
  quantizes Q/K/V to FP16, runs the Stage 5 per-head generation engine, stores
  concat as FP16, then computes W_O to FP32.
- The Stage 5 child exposes append-only `current_valid_seq_len` and no eviction
  interface.
- Tokenizer, embedding, learned position, final norm, LM head, and multi-layer
  sequencing are not present in RTL.

## Compatibility Levels

- Level 0: no RTL change, only weight/vector conversion.
- Level 1: software-side adapter only.
- Level 2: local RTL modules, such as LayerNorm, GELU, bias, RoPE, or variable
  FFN.
- Level 3: visible architecture change, such as SwiGLU, GQA/MQA, sliding/local
  attention, shared KV layout, or multi-layer sequencing.
- Level 4: not suitable as the current primary deployment model.

## Candidate Feature Matrix

| Feature | Current RTL | OPT-125M | TinyStories-1M | SmolLM2-135M | TinyLlama-1.1B | Qwen2.5-0.5B | Qwen3-0.6B | Llama-2-7B |
|---|---|---|---|---|---|---|---|---|
| Norm | RMSNorm | LayerNorm | LayerNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| Pre/Post | Pre-Norm | Pre-LN config | GPT-Neo pre-LN style | Pre-Norm | Pre-Norm | Pre-Norm | Pre-Norm | Pre-Norm |
| Attention | standard MHA | MHA | alternating global/local | GQA | GQA | GQA | GQA | MHA for 7B |
| Position | none | learned absolute | learned/local window | RoPE | RoPE | RoPE | RoPE | RoPE |
| FFN | ReLU W1/W2 | ReLU W1/W2 | GELU MLP | SwiGLU gate/up/down | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| D_FFN | 4D | 4D | 4D inferred | 2.67D | 2.75D | 5.43D | 3D | 2.69D |
| Bias | none | attention/MLP bias | bias likely | no attention bias | no attention bias | QKV bias | no attention bias | generally no bias |
| KV heads | Q heads | equal Q heads | equal Q heads | 3 KV / 9 Q | 4 KV / 32 Q | 2 KV / 14 Q | 8 KV / 16 Q | equal Q heads |
| Layer count | one | 12 | 8 | 30 | 22 | 24 | 28 | 32 |
| Layer output | FP32 | software dtype | software dtype | BF16 weights | F32 ckpt | BF16 | BF16 | FP16/BF16 common |
| KV cache | append-only | full MHA | local/full mix | GQA cache | GQA cache | GQA cache | GQA cache | full MHA cache |
| Compatibility | baseline | Level 2 | Level 3 | Level 3 | Level 3/4 | Level 3 | Level 3 | Level 4 |

## Required Change Matrix

| Candidate | Match summary | Required changes |
|---|---|---|
| `facebook/opt-125m` | Best existing-model match for ReLU, 4D FFN, MHA, no RoPE. | Add LayerNorm or boundary bypass; add linear bias; learned-position stays software-side; extend supported D_MODEL to 768 for full layer. |
| `roneneldan/TinyStories-1M` | Tiny dimensions fit quick tests, but attention/activation/norm differ. | Add LayerNorm, GELU, bias, GPT-Neo local attention/window semantics; D_HEAD=4 lane handling is inefficient but possible. |
| `HuggingFaceTB/SmolLM2-135M` | Matches RMSNorm and no attention bias; otherwise Llama-like. | Add RoPE, GQA mapping/repeat, SwiGLU gate/up/down, variable intermediate size, D_MODEL=576 support, multi-layer software/RTL wrapper. |
| `TinyLlama-1.1B` | Very close to VEDA paper model family. | Same as SmolLM2 plus much larger D_MODEL/weights and F32 checkpoint conversion. |
| `Qwen2.5-0.5B` | Excellent eviction model; architecture is not current RTL. | Add QKV bias, RoPE, GQA, SwiGLU, variable D_FFN, long-context position tracking. |
| `Qwen3-0.6B` | Similar to Qwen2.5, no attention bias in config, head_dim=128. | Add RoPE, GQA, SwiGLU, variable D_FFN, D_HEAD=128 tiling, long-context position tracking. |
| `Llama-2-7B` | Paper reference, not current implementation target. | Add RoPE, SwiGLU, D_MODEL=4096, D_HEAD=128, 32-layer orchestration, gated license flow; full RTL is unrealistic. |

## Direct-Compatibility Finding

No investigated trained model is Level 0. The project should not claim that any
public checkpoint can be directly run through the current Stage 7 layer without
architectural changes or a carefully documented software boundary.
