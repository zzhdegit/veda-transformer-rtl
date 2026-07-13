# Hardware Stage H9 Reset Results

Status: partial reset coverage exists; full reset interrupt matrix remains
open.

Implemented H9-relevant reset coverage in the Stage 5 multi-head testbench:

- reset during provisional append;
- reset during attention start;
- post-reset `valid_seq_len == 0`;
- post-reset `token_ready` restored;
- the following clean token stream must execute successfully.

The H9 single-head and full-layer RTL scripts compile assertions with
`-assert svaext` when VCS is available, but the full requested interrupt matrix
is not yet implemented as independent reset injection points for:

```text
Q projection, K projection, V projection, QK in-flight, score FIFO nonempty,
SFU running max, replay, exp/sum, normalization, probability FIFO nonempty,
inner drain, mode switch, sV in-flight, outer drain, W_O, residual1, FFN,
final output stall, done stall.
```

Current execution result:

```text
vcs: NOT FOUND
result=FAIL
```

Hardware Stage H9 cannot be accepted until the full reset matrix is implemented
and executed in an environment with VCS.
