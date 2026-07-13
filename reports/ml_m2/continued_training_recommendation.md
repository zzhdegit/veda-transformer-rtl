# ML-M2 Continued Training Recommendation

## Decision

Continue training is recommended only if the next goal includes a better
interactive text demo. It is not required for RTL numeric validation.

## A. RTL Numeric Validation

The current checkpoint is sufficient for RTL numerical validation:

- it is trained, not random;
- validation loss is far below the untrained baseline;
- FP16 export exists for all 12 tensors;
- hardware-aware traces exist;
- full forward and incremental KV agree within `3.337860107421875e-06` on the
  prompt suite.

## B. Simple Text Demonstration

The checkpoint can support a simple demonstration that text generation runs, but
it should not be presented as a quality TinyStories model. Greedy output
collapses, and sampling output is more diverse but still weak and often
ungrammatical.

## C. Validation Loss Headroom

The validation curve was still improving at the end of formal training:

```text
step=700 validation_loss=3.321539410523006
step=800 validation_loss=3.3088431528636386
step=891 validation_loss=3.300639416490282
```

There is likely remaining loss headroom.

## D. Repetition Cause

The repetition is caused by both model capacity and decode strategy:

- the one-layer, 64-wide model is intentionally small;
- next-token probability after "Once upon a time" assigns `0.979621` to a
  space token, so greedy decoding is brittle;
- sampling and top-p improve diversity, which shows decode strategy matters;
- EOS rate is 0 across the suite, which points to insufficient training/model
  capacity for natural sequence termination.

## E. Recommendation

For RTL work:

```text
do_not_continue_training_required=true
use_current_checkpoint_for_m3=true
```

For better interactive demo quality:

```text
continue_training=true
keep_model_structure=true
keep_tokenizer=BPE-2048
dataset=TinyStories
target_train_stories=300000
start_checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/checkpoints/ml_m2_formal_best.pt
learning_rate=1e-4
epochs=2
optional_extra_epochs=1-2 if validation loss continues improving
save_best_by=validation_loss
keep_best_and_last_checkpoints=true
do_not_change_RTL=true
```

This task does not start that continued training run.
