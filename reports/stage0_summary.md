# Stage 0 Summary

## Audit Conclusion

PASS. Stage 0 can close and Stage 1 can start from the scoped implementation contract in `docs/stage_00_spec.md`.

## Scope Completed

- Created minimal project structure: `docs/`, `model/`, `rtl/`, `tb/`, `scripts/`, `reports/`, and `build/`.
- Added long-term rules in `AGENTS.md`.
- Added root `.gitignore`, `README.md`, `PROJECT_STATE.md`, and `HANDOFF.md`.
- Added `docs/stage_00_spec.md`.
- Added floating reference model in `model/reference_attention.py`.
- Added model tests in `tb/model/test_reference_attention.py`.
- Re-audited Stage 0 against repository contents, the original Stage 0 plan, the scoped Stage 0 spec, the reference paper, the reference model, and all `tb/model` tests.
- Clarified that the original broad Stage 0 plan's bit/cycle/PPA items are deferred to Stage 1+ and are not claimed complete.
- Marked legacy backend notes as non-authoritative for VEDA Stage 0/1.

## Frozen Decisions

- No voting-based KV cache eviction in the current project line.
- First implementation target is single-head generation attention.
- `qK^T` uses inner-product dataflow.
- Softmax uses element-serial reduction and normalization.
- `s'V` uses outer-product dataflow.
- K/V cache layout is token-major.
- No physical transposed K cache is allowed for the first version.
- Module streams use ready/valid.
- One PE array is not used for `qK^T` and `s'V` at the same time.
- Stage 0 does not run PDK-dependent synthesis, STA, power, or layout flows.

## Provisional Decisions

- FP16 input/weight and FP16 output are the first numeric targets.
- MAC and `s'V` accumulation should use FP32 or equivalent higher precision.
- Softmax internal values should use FP32-equivalent precision until bit-accurate Stage 1 work selects exact behavior.
- Round-to-nearest-even is recommended for FP16 output conversion unless selected arithmetic IP requires a documented alternative.
- Full NaN/Inf/denormal behavior is out of first-version scope.

## Tests Run

```bash
python -m pytest tb/model
python tb/model/test_reference_attention.py
python -m py_compile model/reference_attention.py tb/model/test_reference_attention.py
python model/reference_attention.py --d-head 8 --seq-len 32 --seed 1 --check
```

## Results

- `python -m pytest tb/model`: 7 passed.
- Pure Python fallback: PASS 7 tests.
- Python compile check: passed.
- Direct reference-model `--check`: passed.

## Hygiene Checks

- No `pdk`, `PDK`, `technology`, or `tool_install` directories were found.
- No new large generated artifacts were retained.
- Test-generated `__pycache__` and `.pytest_cache` directories were removed after verification.

## Open Issues

- No blocking Stage 0 audit issues remain.
- Exact arithmetic IP and FP behavior are not confirmed.
- SRAM macro dimensions, port count, latency, and collision semantics are not known.
- Bit-accurate and cycle models are deferred to Stage 1 and must precede RTL numeric tuning.
- Legacy backend notes remain in the repository for traceability but are explicitly non-authoritative.

## Stage 1 Ready Inputs

- `docs/stage_00_spec.md`.
- `model/reference_attention.py`.
- `tb/model/test_reference_attention.py`.
- Frozen dataflow, storage layout, and ready/valid principles in `PROJECT_STATE.md`.
