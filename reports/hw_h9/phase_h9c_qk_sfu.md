# Hardware Stage H9C QK-SFU Overlap

Status: checkpoint implemented, acceptance incomplete.

Model evidence:

- `estimate_h9_interleaved_cycles(64, seq_len)` reports `qk_sfu_overlap_cycles > 0` for seq 8, 16, and 32.
- The model uses bounded score FIFO occupancy and reports full/empty stalls.

RTL smoke counters implemented:

- `perf_qk_sfu_overlap_cycles`
- `perf_qk_only_cycles`
- `perf_sfu_during_qk_cycles`
- `perf_score_fifo_full_stall_cycles`
- `perf_score_fifo_empty_cycles`
- `perf_score_fifo_peak_occupancy`

RTL smoke evidence:

- `tb_h9_single_head` D_HEAD=8: `qk_sfu_overlap=135`.
- `tb_h9_single_head` D_HEAD=16: `qk_sfu_overlap=135`.
- `tb_h9_single_head` D_HEAD=64: `qk_sfu_overlap=135`.

Open acceptance coverage:

- Broad random backpressure/deadlock testing.
- Multi-head and full-layer interleaved RTL.
- Required long-sequence/cache-full matrix.
