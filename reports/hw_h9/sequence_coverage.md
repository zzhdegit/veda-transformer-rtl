# Hardware Stage H9 Sequence Coverage

Status: coverage entry implemented; not executed in the current environment.

The H9 single-head matched RTL A/B evidence covers:

```text
D_HEAD = 8, 16, 64
seq = 1, 2, 8, 16, 32, 64
```

The H9 multi-head Stage 5 vector generator now supports configurable
`--max-seq-len`. The H9 RTL script generates:

```text
MAX_SEQ_LEN=8  for H1/D8, H2/D8, H4/D8, H2/D16, H1/D64
MAX_SEQ_LEN=32 for H1/D8 long-sequence/cache-full coverage
```

The `MAX_SEQ_LEN=32` Stage 5 stream exercises successful sequence lengths 1
through 32 and then one extra cache-full token. This includes the required
irregular points 3, 7, 9, 15, and 31.

Current execution result:

```text
vcs: NOT FOUND
result=FAIL
```

Because VCS is unavailable in this environment, the expanded sequence set is
implemented but not accepted as run evidence.
