# Hardware Stage H9 Hierarchical Coverage Summary

Status: multi-head hierarchical blocker closed; full-layer strict IP-grade
hierarchical blocker deferred from undergraduate thesis scope.

Thesis acceptance status:

```text
HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE
```

Strict verification status:

```text
STRICT IP-GRADE H9 VERIFICATION NOT CLOSED
```

## Closed In This Strict Hierarchical Closure

- Multi-head independent reset coverage now targets the real
  `multi_head_generation_engine` hierarchy, not the direct interleaved datapath
  harness.
- Multi-head reset result: 20/20 independent rows PASS.
- Multi-head reset configurations covered: H2/D8, H4/D8, H2/D16, and H1/D64
  dimension-compatibility subset.
- Multi-head random backpressure now targets the real
  `multi_head_generation_engine` hierarchy.
- Multi-head random result: 24/24 fixed-seed runs PASS.
- Multi-head random configurations covered:
  - H2/D8 seq2: seeds 101, 211, 307, 401.
  - H2/D8 seq8: seeds 503, 601, 701, 809.
  - H2/D8 seq16: seeds 907, 1009, 1103, 1201.
  - H4/D8 seq8: seeds 1301, 1409, 1511, 1601.
  - H2/D16 seq8: seeds 1709, 1801, 1907, 2003.
  - H1/D64 seq8: seeds 2111, 2203, 2309, 2411.
- Legal multi-head random endpoints covered:
  `token_valid_gap`, `multi_head_output_ready`, `multi_head_done_ready`,
  real head-boundary pressure, and real commit-near pressure.
- No DUT internal ready was forced.
- No proxy label is used for any multi-head acceptance row.

## Remaining Strict IP-Grade Blocker

Full-layer independent reset/random coverage is not closed for strict IP-grade
verification. It is not a blocker for the undergraduate thesis acceptance
scope defined in `docs/hw_h9/thesis_acceptance_scope.md`.

The real DUT for that blocker must be `rtl/transformer/transformer_layer.sv`.
The current production top-level does not expose legal testbench control over
several requested child-stage ready endpoints, including MHA child output/done
stall points. Those points must be closed with a test-only wrapper, observation
interface, or another legal child-interface interception strategy. They must not
be replaced with direct datapath labels, fixed-cycle waits, or forced internal
ready signals.

## Decision

Do not claim `FULL IP-GRADE VERIFICATION CLOSURE`.

Do not create `hw-h9-sfu-pe-interleaving-accepted`.

Do not enter Hardware Stage H10.
