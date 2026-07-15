# Hardware Stage H9 Summary

Status:

```text
HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE
```

Strict verification status:

```text
STRICT IP-GRADE H9 VERIFICATION NOT CLOSED
```

## Scope

Hardware Stage H9 implements paper Attention full-array native mapping and
SFU/PE element-serial interleaving. The undergraduate thesis acceptance scope is
defined in `docs/hw_h9/thesis_acceptance_scope.md`.

The accepted thesis baseline covers the architecture, numerical correctness,
matched RTL performance, cycle-model calibration, multi-head integration,
full-layer integration, long-sequence/cache-full behavior, direct reset/random
stress, independent multi-head reset/random stress, assertion execution, lint,
DC structural checks, and Stage5/6/7/8 regressions.

The strict IP-grade full-layer internal reset and full-layer internal
multi-endpoint random-backpressure matrix remains deferred. This limitation is
documented in `reports/hw_h9/deferred_ip_verification.md`.

## Results

- H9 host/model tests: PASS.
- H9 vs H8 bit-model comparison: bit-exact for D_HEAD 8, 16, 64, and 128.
- Numerical comparison: max_abs_error=0, MAE=0, RMSE=0, relative_L2=0,
  max_ULP=0, cosine=1, and attention ranking matches.
- H9 calibrated cycle model: exact total-cycle match against matched RTL A/B
  for D_HEAD 8, 16, and 64 at seq 1, 2, 8, 16, 32, and 64.
- H9 matched single-head RTL A/B performance: PASS at seq16 and seq32 for
  D_HEAD 8, 16, and 64.
- H9 multi-head interleaved RTL: PASS for H1/D8, H2/D8, H4/D8, H2/D16, and
  H1/D64.
- H9 full-layer interleaved RTL: PASS for H1/D8, H2/D8, H2/D8 two-token,
  H4/D8, and H2/D16.
- H9 long-sequence/cache-full: PASS for MAX_SEQ_LEN=32 plus one extra token.
- Direct H9 datapath reset: 64/64 PASS.
- Direct H9 random backpressure: 20/20 fixed seeds PASS.
- Independent multi-head reset: 20/20 PASS against the real
  `multi_head_generation_engine` hierarchy.
- Independent multi-head random: 24/24 fixed-seed runs PASS against the real
  `multi_head_generation_engine` hierarchy.
- Assertion execution: 23 explicit named SVA properties compile, bind, pass
  positive execution, and trigger in isolated negative tests.
- H9 lint/vlogan: PASS with only accepted DesignWare pragma-no-effect warnings.
- H9 DC structural check: PASS for analyze/elaborate/link/check_design and
  hierarchy only.
- Stage5/6/7/8 regressions in the Docker EDA environment: PASS.

Matched RTL A/B remains the performance authority. The older structural cycle
model is retained only as trend evidence and is not used to decide H9 speedup.
The measured performance gain is the combined result of native full-array
mapping plus SFU/PE interleaving; it is not pure interleaving benefit.

## Matched RTL Performance

| D_HEAD | Seq | Staged RTL | Interleaved RTL | Result |
|---:|---:|---:|---:|---|
| 8 | 16 | 1363 | 1169 | PASS |
| 8 | 32 | 2707 | 2209 | PASS |
| 16 | 16 | 2472 | 1171 | PASS |
| 16 | 32 | 4920 | 2211 | PASS |
| 64 | 16 | 9126 | 1183 | PASS |
| 64 | 32 | 18198 | 2223 | PASS |

## Deferred Strict Items

The following remain open for strict IP-grade verification and are not claimed
as closed:

- independent reset injection at every true internal micro-stage of the real
  `transformer_layer` DUT;
- broad full-layer internal multi-endpoint randomized backpressure;
- exhaustive full-layer internal simultaneous stall coverage;
- functional coverage closure;
- assertion coverage closure;
- formal property proof;
- IP signoff-level verification.

No PDK, SRAM macro binding, STA, P&R, timing closure, area, power, frequency,
WNS/TNS, or PPA flow was run or claimed.
