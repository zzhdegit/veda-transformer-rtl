# Stage 8 Summary

## Result

Stage 8 implementation status: PASS through Stage 8D integration regressions.

Paper-structured 8x8x2 reconfigurable PE array correctness accepted for the
implemented repository mapping.

Attention QK and sV mapping correctness accepted for the covered D8/D16,
H1/H2/H4, multi-token, and cache-full regressions.

SFU-PE interleaving, global array sharing, physical memory, timing closure,
and PPA remain provisional.

## Scope

Stage 8 changes only the Attention QK/sV PE architecture. It preserves:

- Stage 7 Pre-Norm Transformer math;
- FP16/FP32 numeric boundaries;
- RMSNorm, Residual, FFN, and Softmax behavior;
- KV cache layout and commit semantics;
- cache-full behavior;
- serial QK -> Softmax/SFU -> sV scheduling.

Projection WQ/WK/WV/WO and FFN W1/W2 remain on the legacy PE path.

## Commits

- Stage 8A: `1aed1c4 stage8a: freeze paper structured pe array specification`
- Stage 8B: `6077ef3 stage8b: add bit accurate 8x8x2 pe array model`
- Stage 8C: `cc241d8 stage8c: implement paper structured 8x8x2 pe array`
- Stage 8D: `dff24b2 stage8d: map attention inner and outer products to paper array`

The final closeout commit records top-level documentation and final regression
status.

## Evidence And Decisions

The local paper source is `2507.00797v1.pdf`. Evidence is recorded in
`docs/stage_08/paper_evidence.md`.

Several microarchitectural details are not fully specified by the paper and
are marked as repository design decisions:

- exact ready/valid protocol;
- reset and abort semantics;
- group masks and tail masks;
- minimal PE-like adapter mapping for current Stage 5 controller integration;
- RTL counter definitions.

## Verification

Host:

- `python scripts/sim/run_stage8_tests.py`: PASS

Docker:

- `make stage8-test`: PASS
- `make stage8-rtl-sim`: PASS
- `make stage8-lint`: PASS
- `make stage8-synth`: PASS
- `make stage8d-test`: PASS
- `make stage8d-rtl-sim`: PASS
- `make stage8d-lint`: PASS
- `make stage8d-synth`: PASS

Previously completed Stage 8C:

- `make stage8c-test`: PASS
- `make stage8c-rtl-sim`: PASS
- `make stage8c-lint`: PASS
- `make stage8c-synth`: PASS

Legacy regressions:

- Stage 7A/7B/7C/7D: PASS.
- Stage 6: PASS.
- Stage 5: PASS.

## Known Limitations

- The Stage 8D adapter is correctness-first and does not exploit full
  128-cell throughput for current `PE_NUM=8` tiles.
- Dense full-layer RTL coverage uses existing Stage7D layer vectors; dense
  Attention RTL coverage is provided through Stage3/Stage5 paper-path tests.
- RTL internal node observation is focused on QK/sV outputs and wrapper
  outputs, not every intermediate PE partial sum in full-layer simulation.
- DC is a structural check only.
- No timing closure, area, power, or frequency result is claimed.
- Multi-layer execution, Embedding, LM Head, Tokenizer, cache eviction, and
  SFU-PE interleaving are not implemented.
