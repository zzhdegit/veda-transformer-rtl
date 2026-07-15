# HW-H9-N1 Random Reduction Results

Target:

```bash
make hw-h9-numeric-repair
```

Result:

```text
Build dir: build/hw_h9_numeric_repair
HW_H9_NUMERIC_VECTOR_COUNTS add=30 reduction=106 core=46
HW_H9_NUMERIC_KNOWN_ADD_PASS count=30
HW_H9_NUMERIC_REDUCTION_PASS count=106
HW_H9_NUMERIC_PE_CORE_PASS vectors=46 outputs=7
HW_H9_NUMERIC_REPAIR_PASS known_operand=3c837d4a random_reductions=100 core_vectors=46
result=PASS
```

The reduction vector set includes 100 fixed-seed random reductions plus
directed pair, tile, gap, stall, reset, signed-zero, subnormal-adjacent,
small-plus-large, cancellation, positive accumulation, and negative
accumulation coverage.
