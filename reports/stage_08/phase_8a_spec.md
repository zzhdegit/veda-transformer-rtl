# Stage 8A Specification Freeze Report

## Result

Stage 8A specification freeze: PASS

Date: 2026-07-13

Branch: `stage8-paper-pe-array`

Baseline commit: `8ad948343179d2ae801fcea3679dce6b143382c3`

Baseline tag: `stage7-correctness-accepted`

## Paper Source

Primary source: `2507.00797v1.pdf`

- Title: VEDA: Efficient LLM Generation Through Voting-based KV Cache Eviction and Dataflow-flexible Accelerator
- Version: arXiv v1
- Date shown in paper header: 1 Jul 2025
- Evidence file: `docs/stage_08/paper_evidence.md`

The specification uses the name Paper-Structured 8x8x2 PE Array. The term
paper-exact topology is not used because the paper does not publish every RTL
handshake, reset, and scheduling detail needed for an implementation.

## Accepted Evidence

The following paper items are sufficient for Stage 8A:

- 8x8x2 PE Array physical size from Table I and Figure 7.
- Flexible-product dataflow from Section IV-A and Figure 4.
- Runtime reconfigurable PE behavior from Section IV-A and Figure 5(a).
- Outer-product configured PE array from Figure 5(b).
- Inner-product configured PE array from Figure 5(c).
- Type-A and Type-B row mapping from Figure 5(d).
- L1/L2 reduction structure from Section IV-A and Figure 5(c,d).
- Token-major KV storage and transpose-avoidance motivation from Section IV-A.

Unspecified details are frozen as repository decisions in
`docs/stage_08/spec.md`.

## Frozen Stage 8A Decisions

- Physical hierarchy: 8 rows x 8 columns x 2 groups = 128 PE cells.
- Required modes: MODE_INNER_PRODUCT and MODE_OUTER_PRODUCT.
- QK uses inner-product mapping.
- sV uses outer-product mapping.
- Stage 5 KV cache layout remains `K/V[head][token][dimension]`.
- Projection PE and FFN PE remain legacy `reconfigurable_pe_core`.
- Softmax remains staged serial; no SFU-PE interleaving is implemented.
- Cache eviction remains out of scope.
- Repository FP16/FP32 arithmetic contract remains frozen.

## Baseline Regression

The accepted Stage 7 baseline was re-run before Stage 8 implementation.

Host:

- `python scripts/sim/run_stage7a_tests.py`: PASS
- `python scripts/sim/run_stage6_tests.py`: PASS
- `python scripts/sim/run_stage5_tests.py`: PASS

Docker in container `nailong`:

- `make stage7a-test`: PASS
- `make stage7b-test && make stage7b-rtl-sim && make stage7b-lint && make stage7b-synth`: PASS
- `make stage7c-test && make stage7c-rtl-sim && make stage7c-lint && make stage7c-synth`: PASS
- `make stage7d-test && make stage7d-rtl-sim && make stage7d-lint && make stage7d-synth`: PASS
- `make stage6-test && make stage6-rtl-sim && make stage6-lint && make stage6-synth`: PASS
- `make stage5-test && make stage5-rtl-sim && make stage5-lint && make stage5-synth`: PASS

Observed non-failing diagnostics:

- Existing VCS/DesignWare PHNE pragma warnings in Stage 7B/7D lint.
- Container clock-skew warnings from generated build files.
- DC checks are analyze/elaborate/link/check_design only and do not provide PPA.

## Exit Criteria For 8A

- Paper evidence table added: PASS
- Stage 8 specification added: PASS
- Unspecified paper details marked as repository decisions: PASS
- No RTL behavior changed: PASS
- No Stage 7 math changed: PASS
- No SFU-PE interleaving started: PASS
- No KV cache eviction started: PASS
