# Hardware Stage H9 Reset Results

Status: partial reset coverage passes; full reset interrupt matrix remains
open.

Executed reset coverage:

- H9 score buffer and probability FIFO initial reset: PASS.
- H9 single-head initial reset before clean run: PASS.
- H9 matched A/B initial reset before each staged/interleaved run: PASS.
- H9 multi-head Stage 5 wrapper reset during provisional append: PASS.
- H9 multi-head Stage 5 wrapper reset during attention start: PASS.
- H9 full-layer Stage 7D wrapper initial reset before clean run: PASS.
- Stage5/6/7/8 regression reset coverage: PASS where those accepted stages
  already define directed reset scenarios.

The H9 Stage 5 reset checks verify:

- reset restores `valid_seq_len` to zero before the clean stream;
- `token_ready` is restored;
- no duplicate commit is observed in the following clean token stream;
- the following clean token stream executes successfully.

Open for final H9 acceptance:

```text
input load, Q projection, K projection, V projection, QK issue, QK in-flight,
score FIFO nonempty, SFU running max, score replay, exp/sum, normalization,
probability FIFO nonempty, inner drain, mode switch, sV issue, sV in-flight,
outer drain, head boundary, middle head, head output stall, W_O, residual1,
FFN, final output stall, done stall.
```

Those reset injection points are not implemented as independent H9 tests. This
prevents `HARDWARE STAGE H9 PASS`.
