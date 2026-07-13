# Stage 7B RMSNorm and Residual RTL

## Result

Stage 7B PASS. RMSNorm and residual-add RTL foundations are implemented and
verified. Full Stage 7 Transformer layer integration is not yet accepted.

## Implemented

- `fp32_sqrt_wrapper` using DesignWare `DW_fp_sqrt` behind the project wrapper
  boundary.
- `rmsnorm_engine` with serial dimension-order sum-square fused MAC, exact
  power-of-two mean scale, `EPS_FP32`, sqrt, reciprocal, `(x * inv_rms) * gamma`
  apply order, and FP32-to-FP16 output quantization.
- `residual_add_engine` using the project FP32 add wrapper.
- RTL vectors and testbench coverage for D_MODEL 8 and 16 with output
  backpressure.

## Verification

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-synth'
```

Results:

- Stage 7B vectors: PASS.
- RMSNorm/residual VCS simulations for D_MODEL 8 and 16: PASS.
- Stage 7B lint/vlogan: PASS with only DesignWare pragma-no-effect warnings.
- DC analyze/elaborate/link/check_design: PASS for `fp32_sqrt_wrapper`,
  D_MODEL 8/16/128 `rmsnorm_engine`, and D_MODEL 16/128
  `residual_add_engine`.

No area, power, WNS, frequency, process timing, STA, layout, or PPA conclusion
is produced.

## Deferred

- FFN/ReLU RTL.
- Full `transformer_layer` top integration.
- Stage 7 full-top VCS assertions, lint/vlogan, and DC structural checks.
