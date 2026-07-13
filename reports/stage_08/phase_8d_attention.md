# Stage 8D Attention Integration Report

## Result

Stage 8D paper-array Attention integration: PASS

Date: 2026-07-13

Branch: `stage8-paper-pe-array`

## Implemented Scope

Stage 8D adds the selectable paper-array Attention path:

- `ATTENTION_PE_ARCH=0`: legacy `reconfigurable_pe_core`
- `ATTENTION_PE_ARCH=1`: `paper_attention_adapter` and `paper_array_8x8x2`

Only Attention QK and sV are mapped to the paper array. Projection WQ/WK/WV/WO
and Stage 7 FFN W1/W2 remain legacy PE users.

## Added Files

- `rtl/attention/paper/paper_attention_adapter.sv`
- `model/attention/paper_attention_reference.py`
- `model/attention/paper_attention_cycle_model.py`
- `tb/model/test_stage8_paper_attention.py`
- `scripts/sim/run_stage8d_tests.py`
- `scripts/sim/run_stage8d_vcs.sh`
- `scripts/sim/run_stage8d_layer_vcs.sh`
- `scripts/lint/run_stage8d_lint.py`
- `scripts/synth/stage8d_elaborate.tcl`
- `scripts/synth/run_stage8d_synth_check.py`

## RTL Integration

Updated wrappers pass `ATTENTION_PE_ARCH` and paper-array counters through:

- `single_head_attention_controller`
- `single_head_attention`
- `multi_head_generation_engine`
- `projection_integrated_mha`
- `transformer_layer`

The generate structure instantiates only the selected architecture in the
synthesized top.

## Model Results

Command:

```text
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8d-test'
```

Result: PASS

The paper attention model compared PAPER_ARRAY against legacy for D8/D16
coverage. Current results are bit-exact for raw scores, scaled scores,
probabilities, and sV output.

## RTL Results

Command:

```text
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8d-rtl-sim'
```

Result: PASS

Attention-only coverage:

| Test | Result |
|---|---|
| single-head D8 | PASS |
| single-head D16 | PASS |
| multi-head H1/D8 | PASS |
| multi-head H2/D8 | PASS |
| multi-head H4/D8 | PASS |
| multi-head H2/D16 | PASS |

Full layer paper coverage:

| Test | Result |
|---|---|
| transformer H1/D8 | PASS |
| transformer H2/D8 | PASS |
| transformer H2/D8 two-token | PASS |
| transformer H4/D8 | PASS |
| transformer H2/D16 | PASS |

Cache-full extra-token behavior remains unchanged in the multi-head tests:
the final step reports `seq_before=8 seq_after=8`.

## Lint And DC

Command:

```text
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8d-lint'
```

Result: PASS. Vlogan reports only existing `PHNE` pragma warnings.

Command:

```text
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8d-synth'
```

Result: PASS.

DC was used only for analyze/elaborate/link/check_design and hierarchy counts.
It does not produce area, power, frequency, timing, or PPA conclusions. DC
emitted tolerated `UID-109` base-design alias diagnostics after parameterized
elaboration; `dc_exit_code=0` and all requested reports were generated.

## Known Limitations

- The adapter is correctness-first and maps current `PE_NUM=8` attention tiles
  into masked paper-array lanes.
- The mapping preserves covered D8/D16 bit patterns but does not yet exploit
  full 128-cell throughput.
- SFU-PE interleaving is not implemented.
- Projection and FFN remain legacy.
- KV cache eviction is not implemented.
