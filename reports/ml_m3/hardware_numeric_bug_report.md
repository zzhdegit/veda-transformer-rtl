# ML-M3 Hardware Numeric Bug Report

## Closure Note

This report is preserved as the pre-repair hardware numeric bug record. The bug is closed by hardware tag `hw-h9-real-weight-numeric-repair-accepted` at commit `a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485`; the model branch did not modify hardware RTL. Current ML-M3 acceptance is `MODEL STAGE M3 PASS` with length1/2/8/16 H8/H9/bit-model bit-exact and zero mismatches.

## Historical Bug Record

- Hardware tag: `hw-h9-sfu-pe-interleaving-thesis-accepted`
- Hardware commit: `9e0b4c9ba42356ee68e489e99cc5cf64e94f607e`
- Affected path: common H8/H9 FFN W2 GEMV reduction path (`reconfigurable_pe_core` / `fp32_reduction_tree` / `fp32_add_wrapper` interaction).
- H8 staged and H9 interleaved produce identical one-token output and identical captured SHA, so this is not an H9-only scheduler issue.
- Vector/export issue excluded: W2 row=1 FP16 operands match the bit model.
- Lane product issue excluded: all 256 row=1 W2 lane products match the bit model.
- Bit-model issue not supported: standalone stable `fp32_add_wrapper` replay of the first divergent operands matches the current bit model and NumPy float32.

## Minimal Failing Operation

- Cycle: `185551`
- Operation: W2 reduction-tree add, tile base `8`, width `8`, pair `3`
- Operand A: `0x3c81aa0c`
- Operand B: `0x39699f40`
- RTL result in full PE path: `0x3c837d4b`
- Bit/stable replay expected: `0x3c837d4a`
- ULP distance: `1`

## Suspected RTL Area

- `rtl/pe/fp32_reduction_tree.sv`: single-wrapper sequential reduction handshake.
- `rtl/pe/reconfigurable_pe_core.sv`: reduction output to tile-accumulator handshake.
- `rtl/arithmetic/fp32_add_wrapper.sv` and `rtl/common/stream_reg.sv`: registered output and ready/valid interaction.

## Proposed Hardware-Line Follow-Up

Create a hardware-owned fix task that reproduces this one-add operation inside the PE reduction context, then adjusts the RTL so `fp32_add_wrapper` outputs are consumed only after stable registered results. The model branch must not patch RTL.

## Regression Command

```powershell
cd D:/IC_Workspace/VEDA_ml_m2
python scripts/ml/run_ml_m3_vcs.py --length 1 --schedule staged --schedule interleaved --run-id numeric_alignment_repro --diagnostic
python scripts/ml/run_ml_m3_numeric_replay.py --run-id numeric_alignment_first_add_const
python scripts/ml/run_ml_m3_numeric_classification.py
```
