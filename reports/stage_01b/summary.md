# Stage 1B Summary

Status: STAGE 1B PASS.

Stage 1B implemented and verified the finite FP16-to-FP32 conversion boundary and a unified FP32 MAC ready/valid wrapper around local DesignWare `DW_fp_mac`.

## Implemented
- `rtl/arithmetic/fp16_to_fp32.sv`
- `rtl/arithmetic/fp32_mac_wrapper.sv`
- `model/arithmetic/fp16_fp32_reference.py`
- `model/arithmetic/fp32_mac_reference.py`
- Stage 1B Python tests, VCS testbenches, VCS runner, lint runner, DC elaboration script, and Makefile targets.

## Frozen Numeric Behavior
- FP16 normal finite values convert exactly to FP32.
- FP16 positive and negative zero preserve sign.
- FP16 subnormals flush to signed zero and set `underflow_or_ftz`.
- FP16 NaN/Inf are illegal, set `invalid`, and trigger simulation assertions on accepted inputs.
- FP32 MAC computes fused `a*b+c` using local `DW_fp_mac #(23, 8, 1)`.
- Local DW RNE mode for `DW_fp_mac` is `rnd=3'b100`.
- Exact zero MAC results are positive zero for the selected DW RNE mode.
- FP32 NaN/Inf MAC operands are illegal, set aligned `invalid`, and trigger simulation assertions.

## Interface Closure
- `fp16_to_fp32`: latency 1, initiation interval 1, ready/valid, metadata and `last` aligned.
- `fp32_mac_wrapper`: latency 1 external output register, initiation interval 1, ready/valid, metadata and `last` aligned, raw DW `status[7:0]` preserved.
- The MAC wrapper is a correctness baseline with a combinational DW arithmetic path before the output register. It is not a high-frequency physical pipeline claim.

## Verification
- Host `python -m pytest tb\model tb\unit`: 32 passed.
- Host `python -m pytest tb\model\test_reference_attention.py`: 7 passed.
- Host `python scripts\sim\run_stage1_tests.py`: 32 tests passed and Python compile passed; host RTL sim skipped because VCS is only in Docker.
- Host `python scripts\sim\run_stage1b_tests.py`: 32 tests passed and Python compile passed; host RTL sim skipped because VCS is only in Docker.
- Docker `make stage1-rtl-sim`: PASS, `STAGE1_RTL_SIM_PASS`.
- Docker `make stage1b-rtl-sim`: PASS.
  - Converter exhaustive test covered all 65,536 FP16 input bit patterns.
  - Converter invalid assertion negative test observed the expected `$fatal`.
  - DW MAC semantics probe confirmed fused behavior and RNE `rnd=4`.
  - MAC wrapper passed 105 directed/random vectors with backpressure, metadata, and `last`.
  - MAC invalid assertion negative test observed the expected `$fatal`.
- Docker `make stage1b-lint`: PASS; vlogan compile exit 0 with no diagnostics.
- Docker `make stage1b-synth`: PASS; DC analyze/elaborate/link/check_design exit 0.

## Environment Notes
- Docker container: `nailong`.
- Repository path in Docker: `/workspace/VEDA`.
- VCS: `O-2018.09-SP2-2_Full64`.
- DC: `L-2016.03-SP1`.
- Docker Python is 3.6.9 and cannot run the current Python sources; host Python 3.12 was used for Python regression.
- No PDK, standard-cell target library, SRAM macro, P&R, formal STA, or PPA was used.

## Stage 2 Readiness
Stage 2 Reconfigurable PE Core may start and should use the frozen `fp16_to_fp32` and `fp32_mac_wrapper` interfaces. Stage 2 must not instantiate DesignWare native ports directly and must not make physical PPA claims without real technology inputs.
