# Hardware Stage H9 Stream Protocol

## Score Packet

| Field | Meaning |
|---|---|
| `valid` / `ready` | Ready/valid handshake. |
| `token_meta` | Transaction metadata propagated from the token. |
| `head_id` | Logical attention head. |
| `logical_token_index` | Token order inside the active head. |
| `cache_slot` | K cache slot used for this score. |
| `score_index` | Ordered score index; monotonic within a head. |
| `tile_id` | Paper-array score tile id. |
| `lane_mask` | Active native paper-array lane mask. |
| `score_fp32` | Scaled FP32 score. |
| `last_in_tile` | Last score for the current tile. |
| `last_in_head` | Last score for the head. |
| `status` | Accumulated status bits. |
| `invalid` | Hard invalid flag. |

Payload must remain stable while `valid && !ready`.

## Probability Packet

| Field | Meaning |
|---|---|
| `valid` / `ready` | Ready/valid handshake. |
| `token_meta` | Transaction metadata propagated from the token. |
| `head_id` | Logical attention head. |
| `logical_token_index` | Token order inside the active head. |
| `cache_slot` | V cache slot consumed by sV. |
| `probability_index` | Ordered probability index; monotonic within a head. |
| `probability_fp32` | Normalized FP32 probability. |
| `last_probability` | Last probability for the head. |
| `status` | Accumulated status bits. |
| `invalid` | Hard invalid flag. |

Probability index must match the V token index exactly.

## Reset And Errors

Reset clears all in-flight score and probability packets, FIFO occupancy, read/write pointers, and packet-valid state. Error status travels with the packet and must not lose transaction identity under stalls.
