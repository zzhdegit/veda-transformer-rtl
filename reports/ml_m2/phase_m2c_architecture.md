# ML-M2C Report: Architecture and Unit Tests

## Result

ML-M2C implements the one-layer hardware-matched PyTorch causal LM.

## Implemented Structure

```text
hidden_input = token_embedding + position_embedding
n1 = RMSNorm(x)
a  = MHA(n1)
r1 = x + a
n2 = RMSNorm(r1)
h1 = W1(n2)
h  = ReLU(h1)
f  = W2(h)
y  = r1 + f
final_norm -> tied lm_head
```

## Contract Checks

- `D_MODEL = N_HEAD * D_HEAD`
- `D_FFN = 4 * D_MODEL`
- `N_Q_HEAD = N_KV_HEAD`
- WQ/WK/WV/WO no bias
- W1/W2 no bias
- LM head no bias
- RMSNorm, not LayerNorm
- ReLU, not GELU/SiLU/SwiGLU
- standard causal MHA
- append-only KV cache
- software-side learned absolute position embedding

## Unit Tests

Tests cover:

- tensor shapes;
- causal mask and no future-token access;
- current-token visibility;
- RMSNorm math;
- residual order;
- FFN ReLU;
- no bias;
- required state_dict names;
- deterministic forward;
- incremental KV vs full sequence;
- one-token and multi-token greedy decode;
- cache reset.

