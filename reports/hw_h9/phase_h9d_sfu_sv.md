# Hardware Stage H9D SFU-sV Overlap

Status: checkpoint implemented, acceptance incomplete.

Model evidence:

- `estimate_h9_interleaved_cycles(64, seq_len)` reports `sfu_sv_overlap_cycles > 0` for seq 8, 16, and 32.
- Probability packet order is verified by `tb/model/test_hw_h9_interleaved_attention.py`.

RTL smoke counters implemented:

- `perf_sfu_sv_overlap_cycles`
- `perf_sfu_only_cycles`
- `perf_sv_only_cycles`
- `perf_probability_fifo_full_stall_cycles`
- `perf_probability_fifo_empty_stall_cycles`
- `perf_probability_fifo_peak_occupancy`
- `perf_inner_to_outer_switch_cycles`
- `perf_pipeline_bubble_cycles`

RTL smoke evidence:

- `tb_h9_single_head` D_HEAD=8: `sfu_sv_overlap=66`.
- `tb_h9_single_head` D_HEAD=16: `sfu_sv_overlap=66`.
- `tb_h9_single_head` D_HEAD=64: `sfu_sv_overlap=66`.

Open acceptance coverage:

- Broad output-stall and probability-consumer backpressure tests.
- Multi-head and full-layer interleaved RTL.
- Required long-sequence/cache-full matrix.
