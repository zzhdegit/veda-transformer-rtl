# HW-H9-N1 Numeric Contract

## FP32 Add

`fp32_add_wrapper` is the only project wrapper for standalone FP32 addition.
Its contract is:

- finite FP32 inputs only;
- one output register stage;
- initiation interval 1;
- `DW_fp_add` IEEE compliance enabled;
- RNE rounding by DesignWare `rnd=3'b000`;
- status and invalid bits registered with the result payload;
- output payload held stable until `out_ready`.

The wrapper rejects non-finite input in simulation when
`ASSERT_ON_INVALID=1`.

## Reduction Tree

`fp32_reduction_tree` consumes a full lane vector and lane mask, zeros inactive
lanes, and serializes balanced pair additions through one `fp32_add_wrapper`.
The functional order is unchanged by HW-H9-N1:

1. load masked lane values;
2. reduce adjacent pairs at the current width;
3. store each pair result at the compacted pair index;
4. halve the active width;
5. repeat until width 1;
6. emit the final sum with the original metadata and last flag.

The repair does not change latency, external handshake, pair order, tile order,
or accumulation order.

## Verification Assertions

HW-H9-N1 adds simulation-only checks to the reduction tree to guard the failure
class:

- no add launch while an add result is still inflight;
- no add result without a matching inflight operation;
- result-valid pair id matches the launched pair id;
- result-valid width matches the launched reduction width;
- add operands remain associated with the launched operation until result;
- output payload and metadata remain stable while stalled.

These assertions are not synthesis logic and do not create debug ports.
