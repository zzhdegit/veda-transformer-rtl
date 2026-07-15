# ML-M3 Weight Mapping Audit

RTL layer weights use `weight[output_index][input_index]`. Embedding, learned position, final RMSNorm, and tied LM head remain software-side.

| Tensor | PyTorch Shape | RTL Shape | Transpose | Elements | SHA256 | Result |
|---|---:|---:|---|---:|---|---|
| wq | [64, 64] | [64, 64] | False | 4096 | `1a41f37ebfe62c7fff2edf56c490682ff084997f6b8e6fdbfd8a8600bbbd4ee8` | PASS |
| wk | [64, 64] | [64, 64] | False | 4096 | `c8a2a2057de5c9d304f072334460d669bdc4f391a83c5969b2f1e100a0a8db0e` | PASS |
| wv | [64, 64] | [64, 64] | False | 4096 | `e5f1fe6b9e3294b1bc5b226aef3de51c5fb5a71952a5a9f7a5e44069f52b3d56` | PASS |
| wo | [64, 64] | [64, 64] | False | 4096 | `fbaa28df74bed82cc0830fce3f7cffb9986749afacd0ebeb8ae84a414855ab30` | PASS |
| norm1_gamma | [64] | [64] | False | 64 | `ec9b19a387d1fda30f77c96e6bd44565058a3223015a83bd19a72168f2bce3ef` | PASS |
| norm2_gamma | [64] | [64] | False | 64 | `a68ea45d8d57600a57628a3320a4190acb7e0370c4972572c736179091c1d5ec` | PASS |
| w1 | [256, 64] | [256, 64] | False | 16384 | `06412ff080a2adff5a97567fb22d4b50c0ea424913d271de5a3c3f6a18e81cae` | PASS |
| w2 | [64, 256] | [64, 256] | False | 16384 | `88efc691fa49e102df9fc61e16b78c304e58fbcdf8f4d78d1678b1241bfe42ff` | PASS |

Overall: **PASS**
