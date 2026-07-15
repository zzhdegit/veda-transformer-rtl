# HW-H9-N1 Q2 Length1 Results

Target:

```bash
make hw-h9-q2-length1
```

External read-only vector:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m3/vectors/len_1/case_len_1.mem
```

## Results

| Run | Result | Output transactions | Output lanes | Done count | valid_seq_len | total_layer_cycles |
|---|---|---:|---:|---:|---:|---:|
| H8 staged, no stall | PASS | 8 | 64 | 1 | 1 | 127461 |
| H8 staged, output stall | PASS | 8 | 64 | 1 | 1 | 127463 |
| H9 interleaved, no stall | PASS | 8 | 64 | 1 | 1 | 128285 |
| H9 interleaved, output stall | PASS | 8 | 64 | 1 | 1 | 128286 |

All four runs are full `transformer_layer` runs with the real Q2 length1 input
and weights. Each run is 64/64 bit-exact against the M3 hardware-aware
expected output.

H8 staged and H9 interleaved are bit-identical after the repair for the tested
length1 no-stall and output-stall scenarios.
