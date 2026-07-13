# Stage M1 Model Selection Requirements

Query date: 2026-07-13

Stage M1 is a research-only stage for selecting a real trained target model
for later VEDA-inspired Transformer RTL validation. It does not download model
weights, export tokenizer caches, change RTL, change the existing bit models, or
start model deployment.

## Trusted Local Baseline

The local contract was extracted from:

- `AGENTS.md`
- `PROJECT_STATE.md`
- `HANDOFF.md`
- `README.md`
- `docs/stage_07/spec.md`
- `reports/stage_07/summary.md`
- `reports/stage_07/acceptance_audit.md`
- `reports/stage_06/acceptance_audit.md`
- `rtl/transformer/transformer_layer.sv`
- `rtl/transformer/rmsnorm_engine.sv`
- `rtl/transformer/ffn_engine.sv`
- `rtl/attention/projection_integrated_mha.sv`
- `model/transformer/*`
- `model/projection/*`

The ignored local planning file
`transformer_rtl_plan_md/LATE_STAGE_REAL_MODEL_VALIDATION_PLAN.md` was readable
but is treated only as local planning, not as a stage specification.

## Frozen RTL Contract For Selection

- Decoder-style single-token generation only.
- One accepted Pre-Norm Transformer layer:
  `RMSNorm -> MHA -> residual -> RMSNorm -> W1/ReLU/W2 -> residual`.
- `D_MODEL = N_HEAD * D_HEAD`.
- `D_FFN = 4 * D_MODEL`.
- Current RMSNorm implementation supports power-of-two `D_MODEL` values
  `8, 16, 32, 64, 128`.
- WQ/WK/WV/WO have no bias and use output-row-major weights.
- FFN has no bias and uses output-row-major W1 and W2.
- Hidden and weights are FP16 at RTL-visible matrix input boundaries.
- GEMV accumulation, MHA output, residuals, and final layer output are FP32.
- Q/K/V projection outputs are FP32, then quantized to FP16.
- MHA head concat is logical FP32 in the bit model and stored as FP16 in RTL
  before W_O.
- ReLU maps negative finite values, signed zero, NaN, and Inf to +0; NaN/Inf
  are invalid.
- There is no RoPE, learned position hardware, GELU, SiLU, SwiGLU, bias,
  GQA/MQA, multiple layers, tokenizer, embedding, final norm, or LM head.
- Current KV cache is append-only, stores K/V per attention head, has no
  eviction, and preserves Stage 5 all-head atomic commit semantics.

## Candidate Routes

Route A seeks a trained model with the fewest hardware changes. The ideal model
would be decoder-only, Pre-Norm, RMSNorm, standard MHA, ReLU two-matrix FFN,
no bias, `D_FFN = 4D`, small hidden size, and public weights/tokenizer.

Route B seeks a small Llama-like model for future VEDA-aligned work. These
models are expected to require RoPE, SiLU/SwiGLU, variable FFN width, and often
GQA/MQA.

Route C keeps a hardware-matched training fallback: train a tiny strict-match
model with `D_MODEL` in `{16, 32, 64, 128}`, standard MHA, RMSNorm, ReLU FFN,
no bias, and context length 32 to 256.

## Online Source Rules

Use first-party or primary sources:

- Hugging Face model card.
- Hugging Face `config.json`.
- Hugging Face API metadata for revision, gating, license, and file lists.
- Official model GitHub or technical report.
- Official Transformers implementation or documentation.
- License text or license field.

Do not rely on secondary rankings or model names to infer architecture.

## Automation Policy

Scripts under `scripts/model_selection/` must:

- Fetch config/model-card/API metadata only.
- Not download model weights.
- Not require a Hugging Face token.
- Not call `trust_remote_code=True`.
- Preserve missing fields instead of inventing them.
- Record source URLs and query date.
- Keep generated artifacts small and weight-free.
