# Stage 6 Summary

## Final Status

STAGE 6 PASS. Acceptance audit PASS.

projection-integrated multi-head attention correctness accepted.

throughput, physical memory, and timing pipeline provisional.

## Scope

Stage 6 closes projection-integrated multi-head attention:

```text
hidden FP16
-> Q/K/V projection
-> Q/K/V FP32-to-FP16 quantization
-> Stage 5 multi-head current-token causal attention
-> streamed head concat quantization
-> FP16 concat buffer
-> W_O projection
-> final FP32 output
```

Stage 6 does not implement a complete Transformer layer. Norm, residual, FFN,
activation functions, SRAM macro binding, timing closure, STA, layout, and PPA
are not included.

## Artifacts

Main RTL:

- `rtl/attention/projection_integrated_mha.sv`
- `rtl/projection/head_concat_quantizer.sv`
- `rtl/projection/concat_fp16_buffer.sv`
- `rtl/projection/output_projection_controller.sv`

Reports:

- `reports/stage_06/phase_6a_spec.md`
- `reports/stage_06/phase_6b_summary.md`
- `reports/stage_06/phase_6c_summary.md`
- `reports/stage_06/phase_6d_summary.md`
- `reports/stage_06/phase_6e_summary.md`
- `reports/stage_06/phase_6f_summary.md`

## Acceptance Points

- Stage 6A-6D regressions continue to pass.
- Head concat index is frozen and verified.
- RTL stores only FP16 concat for W_O.
- W_O reuses the existing shared projection GEMV.
- No new PE instance is added for W_O.
- Final top output is bit-exact against the bit model.
- Dense deterministic weights pass in H2/D8.
- Multi-token, cache-full, output backpressure, done backpressure, metadata, and
  valid sequence length checks pass.
- Final-top directed reset coverage passes for reset during Q, K, V, QKV
  stream, attention, concat quantization, W_O, final output stall, and final
  done stall, with clean one-token recovery after weight reload.
- Stage 5 current-token causal semantics and all-head atomic commit are
  preserved.
- Assertions execute under VCS.
- vlogan and DC structural checks pass.
- Cycle counters are recorded and documented.

## Final Verification

Host:

- `python scripts/sim/run_stage5_tests.py`: PASS
- `python scripts/sim/run_stage6_tests.py`: PASS

Docker:

- `make stage5-test stage5-rtl-sim stage5-lint stage5-synth stage6-test stage6-rtl-sim stage6-lint stage6-synth`: PASS
- `make stage6-test`: PASS
- `make stage6-rtl-sim`: PASS
- `make stage6-lint`: PASS
- `make stage6-synth`: PASS

Additional phase and baseline regressions:

- `make stage6e-rtl-sim`, `make stage6e-lint`, `make stage6e-synth`: PASS
- `make stage6d-rtl-sim`, `make stage6d-lint`, `make stage6d-synth`: PASS
- `make stage6c-rtl-sim`, `make stage6c-lint`, `make stage6c-synth`: PASS
- `make stage6b-rtl-sim`, `make stage6b-lint`, `make stage6b-synth`: PASS
- `make stage5-rtl-sim`, `make stage5-lint`, `make stage5-synth`: PASS

## Cycle Counter Notes

Counters are cumulative from reset. Cache-full steps do not increment
`perf_generation_steps`. Stage 5 counters are exposed directly and are not
double-counted as projection cycles.

No-stall cycle model examples are produced by:

```bash
python model/projection/projection_mha_cycle_model.py
```

The model explains:

```text
total_cycles =
hidden_load
+ q_projection
+ k_projection
+ v_projection
+ qkv_quantization
+ attention
+ concat_quantization
+ output_projection
+ final_output
+ control_overhead
```

If the model differs from RTL counters, expected differences come from
ready/valid handshakes, DesignWare model latency, output stalls, Stage 5 SFU
stalls, and serialized projection PE stalls.
