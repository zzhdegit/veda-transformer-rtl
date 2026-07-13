# ML-M2 Pretraining Numeric Audit

## Result

PASS.

The previous CPU smoke losses were far above random-uniform loss:

```text
old_vocab_size=256
old_initial_loss=44.47267532348633
old_final_loss=15.681257247924805
old_validation_loss=14.336987495422363
random_uniform_log_vocab=5.545177444479562
```

## Root Cause

Two issues were found before formal training:

- The tied token embedding and LM head used PyTorch default initialization,
  producing excessively large initial logits for this tiny model.
- Padded label positions used PAD token id 0, while the loss ignored only
  `-100`, so PAD targets contributed to cross entropy.

## Fixes

- Added `initializer_range=0.02` to `HardwareMatchedConfig`.
- Initialized Linear and Embedding weights with normal `std=0.02`.
- Zeroed the PAD embedding row.
- Initialized RMSNorm gamma to 1.
- Kept LM head tied to the token embedding after initialization.
- Converted padded labels to `-100` in `build_lm_sequences`.

## Regression Tests

Added tests:

```text
test_label_shift
test_padding_ignore
test_vocab_range
test_logits_layout
test_tied_lm_head
test_initial_loss_scale
test_single_batch_overfit
```

## Sanity Results

```text
untrained_loss_vocab_256=5.566272735595703
log_vocab_256=5.545177444479562
ratio_256=1.0038042589849208
passes_2x_log_vocab_256=true

untrained_loss_vocab_2048=7.611394882202148
log_vocab_2048=7.6246189861593985
ratio_2048=0.9982656046182432
passes_2x_log_vocab_2048=true
```

Single-batch overfit:

```text
initial_loss=4.178725242614746
final_loss=0.41808220744132996
steps=13
top1=0.921875
reload_max_abs_diff=0.0
checkpoint_sha256=ecdcb025bdb14d09eea4d06d3bc5438c57d1517eb4d0f56ead286ab472401263
```

Formal training was allowed to start only after this audit passed.
