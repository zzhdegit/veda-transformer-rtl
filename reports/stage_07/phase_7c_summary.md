# Stage 7C FFN/ReLU RTL

## Result

Stage 7C PASS. FFN/ReLU RTL foundation is implemented and verified. Full Stage
7 Transformer layer integration is not yet accepted.

## Implemented

- `ffn_engine` with `D_FFN = 4 * D_MODEL`.
- W1 and W2 share exactly one `reconfigurable_pe_core`.
- ReLU passes positive finite FP32 values, maps negative finite and signed zero
  to `+0`, and treats NaN/Inf as invalid.
- ReLU output is quantized through the existing FP32-to-FP16 converter before
  W2.
- Final FFN output remains FP32.

## Verification

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-synth'
```

Results:

- Stage 7C vectors: PASS.
- FFN/ReLU VCS simulations for D_MODEL 8 and 16: PASS.
- Stage 7C lint/vlogan: PASS with no diagnostics.
- DC analyze/elaborate/link/check_design: PASS for D_MODEL 8/16
  `ffn_engine`.

No area, power, WNS, frequency, process timing, STA, layout, or PPA conclusion
is produced.

## Deferred

- Full `transformer_layer` top integration around Stage 6 MHA.
- Stage 7 full-top VCS assertions, lint/vlogan, and DC structural checks.
