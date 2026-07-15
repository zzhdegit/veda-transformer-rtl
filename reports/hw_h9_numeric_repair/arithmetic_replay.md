# HW-H9-N1 Arithmetic Replay

## Known Operand

The known first-divergent pair is:

```text
3c81aa0c + 39699f40
```

Expected RNE result:

```text
3c837d4a
```

Committed regression result:

```text
HW_H9_NUMERIC_KNOWN_ADD_PASS count=30
HW_H9_NUMERIC_REPAIR_PASS known_operand=3c837d4a random_reductions=100 core_vectors=46
result=PASS
```

## Real Reduction Path

The same operand class is exercised through:

- `fp32_reduction_tree`;
- pair 0 through pair 3 placements;
- W2 tile-base-8 style lane product vector;
- `reconfigurable_pe_core` tile accumulation.

Committed regression result:

```text
HW_H9_NUMERIC_REDUCTION_PASS count=106
HW_H9_NUMERIC_PE_CORE_PASS vectors=46 outputs=7
```
