# Stage 8 Cycle And Utilization Comparison

## Scope

These are RTL cycle counters and no-stall structural model estimates. They are
not timing closure, frequency, area, power, or PPA conclusions.

## Structural Model

Command:

```text
python model/attention/paper_attention_cycle_model.py
```

| D_HEAD | Seq Len | QK | Softmax | sV | Total |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 10 | 6 | 3 | 20 |
| 8 | 8 | 80 | 20 | 17 | 118 |
| 16 | 1 | 10 | 6 | 3 | 20 |
| 16 | 8 | 80 | 20 | 17 | 118 |
| 128 | 1 | 10 | 6 | 3 | 20 |
| 128 | 8 | 80 | 20 | 17 | 118 |

## RTL Counters

Single-head mixed cases from `phase_8d_vcs_rtl_sim.txt`:

| Config | Seq Len | Total | QK | Softmax Reduction/Norm | sV | PE Stall | SFU Stall | Output Stall | Paper Active | Inner | Outer | Tail Masked | Mode Switch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D8 | 8 | 693 | 472 | 174 | 94 | 522 | 60 | 1 | 1220 | 1060 | 160 | 4800 | 7 |
| D16 | 8 | 1250 | 936 | 174 | 187 | 1044 | 60 | 1 | 2440 | 2120 | 320 | 9600 | 7 |

Multi-head final cache-full step counters:

| Config | Total | Per-Head Attention | Cache Read | Cache Write | PE Stall | SFU Stall | Output Stall | Paper Active | Inner | Outer | Tail Masked | Mode Switch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1/D8 | 4553 | 4398 | 576 | 64 | 2356 | 300 | 2 | 2196 | 1908 | 288 | 8640 | 15 |
| H2/D8 | 9095 | 8795 | 1152 | 128 | 4712 | 600 | 3 | 4392 | 3816 | 576 | 17280 | 31 |
| H4/D8 | 18189 | 17602 | 2304 | 256 | 9424 | 1200 | 18 | 8784 | 7632 | 1152 | 34560 | 63 |
| H2/D16 | 16851 | 16281 | 2304 | 256 | 9424 | 600 | 9 | 8784 | 7632 | 1152 | 34560 | 31 |

## Utilization Interpretation

Current utilization is intentionally low because Stage 8D uses a PE-like
adapter that maps existing `PE_NUM=8` attention tiles into the paper array and
masks inactive cells. The high `tail_masked_pe_cycles` values are expected for
this correctness-first mapping.

The counters demonstrate that:

- inner-product and outer-product modes both execute;
- mode switches are counted;
- group counters are exposed through the array and wrappers;
- output stalls are recorded;
- cache-full extra-token behavior does not increase valid sequence length.
