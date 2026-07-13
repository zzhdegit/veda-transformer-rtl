# Stage 6E Head Concat and Output Projection Summary

## Scope

Stage 6E adds streamed head concat, FP32-to-FP16 concat quantization, and W_O
output projection to the projection-integrated MHA path.

Dataflow:

```text
hidden FP16
-> shared Q/K/V projection
-> FP32-to-FP16 Q/K/V quantization
-> Stage 5 multi-head attention
-> streamed FP32 head output concat
-> FP32-to-FP16 concat quantization
-> FP16 concat buffer
-> shared W_O GEMV
-> final tiled FP32 output
```

## RTL Added

- `rtl/projection/head_concat_quantizer.sv`
- `rtl/projection/concat_fp16_buffer.sv`
- `rtl/projection/output_projection_controller.sv`
- `rtl/attention/projection_integrated_mha.sv`

`projection_integrated_mha` instantiates one `projection_controller`. Q, K, V,
and W_O share that single controller, its `shared_gemv_projection_core`, and the
single underlying `reconfigurable_pe_core`.

No new PE instance was added for W_O.

## Concat Implementation

The logical concat index is:

```text
concat_index = output_head * D_HEAD + output_base_dim + lane_index
```

The Python bit model keeps `head_output_fp32`, `concat_fp32`, `concat_fp16`, and
`wo_output_fp32` for node comparison. RTL does not keep a complete FP32 concat
buffer. It serializes active output lanes through one `fp32_to_fp16` converter
and writes only `concat_fp16[D_MODEL]`.

The concat buffer tracks duplicate writes, missing elements, range errors, and
complete state. W_O cannot start before concat completion.

## Transaction Order

For each token:

1. QKV projection completes.
2. Stage 5 attention completes and commits K/V atomically.
3. Stage 5 head outputs are collected and quantized into FP16 concat storage.
4. W_O runs through the shared projection GEMV.
5. Final tiled FP32 output and final done are produced.
6. The next hidden-state token is accepted only after final done handshakes.

If Stage 5 reports cache-full before commit, W_O does not start and
`valid_seq_len` does not change. If a later concat/W_O error occurs after Stage
5 commit, K/V is not rolled back and final done reports invalid.

## Verification

Host:

- `python scripts/sim/run_stage6e_tests.py`: PASS
- `python scripts/sim/run_stage6_tests.py`: PASS

Docker:

- `bash scripts/sim/run_stage6e_vcs.sh`: PASS
- `python3 scripts/lint/run_stage6e_lint.py`: PASS
- `python3 scripts/synth/run_stage6e_synth_check.py`: PASS

VCS configs:

- H1/D8
- H2/D8
- H4/D8
- H2/D16

Each config runs repeated tokens through `MAX_SEQ_LEN`, includes a cache-full
extra step, checks final tiled FP32 output bit-exactly against the bit model,
checks final status/metadata/valid sequence length, and exercises output/done
backpressure.

The H2/D8 vector set uses deterministic dense WQ/WK/WV/WO weights with mixed
signs, cancellation, and powers-of-two cases. Other configs use sparse exact
QKV matrices and dense W_O coverage.

## Cycle Counters

The VCS report records cumulative counters per token. Final cache-full steps do
not increment `perf_generation_steps`; final `perf_peak_valid_seq_len` remains
8 for all four configs.

Representative final cumulative counters:

| Config | steps | total | Q | K | V | QKV quant | attention | concat | W_O | proj PE stall | output stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1/D8 | 8 | 9722 | 1674 | 1674 | 1674 | 432 | 2776 | 136 | 1577 | 0 | 4 |
| H2/D8 | 8 | 30417 | 6210 | 6210 | 6210 | 864 | 5552 | 272 | 5672 | 10640 | 5 |
| H4/D8 | 8 | 105409 | 23922 | 23922 | 23922 | 1728 | 11104 | 544 | 21544 | 63840 | 11 |
| H2/D16 | 8 | 104105 | 23922 | 23922 | 23922 | 1728 | 9816 | 544 | 21544 | 63840 | 34 |

These are RTL counters only and are not PPA.

## Known Limits

- Behavioral projection, concat, and cache memories remain provisional.
- D_MODEL=128 DC checking is address/control elaboration on key Stage 6
  components, not a physical memory implementation.
- No bias is implemented.
- No RMSNorm, residual path, FFN, Transformer layer, SRAM macro binding, STA,
  layout, area, power, WNS, or frequency result is claimed.
