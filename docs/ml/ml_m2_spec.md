# ML-M2 Specification: Hardware-Matched Language Model Training

## Scope

Model Stage M2 builds a reproducible training, inference, export, and trace
foundation for a self-trained language model that matches the accepted Stage 7
single Transformer layer. It is a model/software stage, not a hardware stage.

ML-M2 does not implement or modify RTL. It does not add LayerNorm, bias, RoPE,
GQA, SwiGLU, PE arrays, SFUs, KV eviction, or Hardware Stage H8 changes.

## Frozen Initial Model

```text
architecture: decoder-only causal language model
num_layers: 1
d_model: 64
num_attention_heads: 8
num_key_value_heads: 8
d_head: 8
d_ffn: 256
norm: RMSNorm
rms_norm_eps: 1.0e-5
activation: ReLU
bias: false
attention: standard MHA
position: learned absolute position embedding, software-side
context_length: 128
tokenizer: BPE
initial_vocab_size: 2048
embedding: software-side
final_norm: software-side RMSNorm
lm_head: software-side
weight_tying: true by default
dropout: 0.0 for first smoke/export path
dataset: TinyStories
smoke_dataset: Tiny Shakespeare or built-in fixture
```

The layer equation is:

```text
n1 = RMSNorm(x)
a  = MHA(n1)
r1 = x + a
n2 = RMSNorm(r1)
h1 = W1(n2)
h  = ReLU(h1)
f  = W2(h)
y  = r1 + f
```

## Hardware Contract Reconfirmed From Repository

The accepted Stage 7 contract is:

- decoder-style single-token generation;
- Pre-Norm;
- RMSNorm;
- standard MHA;
- `N_Q_HEAD = N_KV_HEAD = N_HEAD`;
- `D_MODEL = N_HEAD * D_HEAD`;
- `D_FFN = 4 * D_MODEL`;
- WQ/WK/WV/WO have no bias;
- FFN W1/W2 have no bias;
- ReLU FFN;
- current-token causal attention can attend to the just-appended token;
- hidden and weights are FP16 at RTL boundaries;
- GEMV accumulation is FP32;
- residuals are FP32;
- Q/K/V and concat have explicit FP32-to-FP16 quantization points;
- only one Transformer layer top is accepted;
- KV cache is append-only, per attention head;
- K/V commit is all-head atomic;
- no RoPE, no GQA, no SwiGLU;
- no embedding, final norm, LM head, or tokenizer in RTL.

No conflict was found between the Stage 7 audit and the accepted RTL contract
for ML-M2. Hardware Stage H8 may add a paper-structured PE array in a separate
line, but ML-M2 exports first target the accepted Stage 7 legacy datapath.

## Position Boundary

Position is software-side only:

```text
hidden_input[t] = token_embedding[token_id] + position_embedding[t]
```

Rules:

- position embedding does not enter RTL as a separate operation;
- exported layer input already includes token and position information;
- position index starts at 0;
- prompt and generation use continuous logical positions;
- RTL must not add position encoding again;
- this is not RoPE or Llama-compatible position handling.

## Tokenizer Contract

Default special tokens:

```text
PAD = 0
BOS = 1
EOS = 2
UNK = 3
```

Default tokenizer is deterministic BPE with `vocab_size=2048`, configurable to
4096. Training text and config must be captured in a tokenizer manifest.

## Sequence Contract

Training examples are packed into fixed-length causal LM blocks. Each block
stores:

```text
input_ids:  [context_length]
labels:     [context_length]
attention:  implicit causal mask
```

Labels are the next-token target. Test prompts are deterministic and excluded
from training.

## State Dict Names

Required names:

```text
token_embedding.weight
position_embedding.weight
layers.0.norm1.weight
layers.0.attn.wq.weight
layers.0.attn.wk.weight
layers.0.attn.wv.weight
layers.0.attn.wo.weight
layers.0.norm2.weight
layers.0.ffn.w1.weight
layers.0.ffn.w2.weight
final_norm.weight
lm_head.weight
```

When weight tying is enabled, `lm_head.weight` aliases
`token_embedding.weight`.

## Tensor Shapes

```text
token_embedding.weight: [vocab_size, d_model]
position_embedding.weight: [context_length, d_model]
norm gamma: [d_model]
WQ/WK/WV/WO: [d_model, d_model]
W1: [d_ffn, d_model]
W2: [d_model, d_ffn]
final_norm.weight: [d_model]
lm_head.weight: [vocab_size, d_model]
```

PyTorch `Linear.weight` uses `[output_index, input_index]`, matching the RTL
logical layout `weight[output_index][input_index]`.

## Acceptance Criteria

ML-M2 pipeline acceptance requires:

- isolated ML branch/worktree;
- `ml/`, `scripts/ml/`, `docs/ml/`, and `reports/ml_m2/` artifacts;
- deterministic tokenizer save/load;
- one-layer model matches the Stage 7 structure;
- no bias, RMSNorm, standard MHA, ReLU, `D_FFN=4D`;
- causal mask and current-token semantics tests;
- full forward and incremental KV decode agreement;
- CPU smoke training with loss decrease, no NaN/Inf, checkpoint reload, and
  greedy generation;
- formal training completed or explicitly PENDING when GPU resources are absent;
- FP16 export manifests with SHA256;
- layer and KV traces;
- hardware-aware bit-model path can run;
- no RTL or Hardware Stage H8 file modifications.

