# Hardware Stage H9 Softmax Schedule

The repository keeps the current Stage 3 softmax arithmetic:

```text
online reduction: max and exp_sum
replay normalization: exp(score - max) / exp_sum
```

The paper supports element-serial reduction and normalization around the PE array, but does not define bit-level RTL arithmetic. Therefore, H9 freezes the existing repository arithmetic and changes only scheduling.

## Reduction Phase

```text
PE score producer
-> score packet
-> score replay buffer
-> bounded score FIFO
-> softmax reduction consumer
```

QK can continue producing later score packets while the SFU reduces earlier packets.

## Normalization Phase

```text
score replay
-> softmax normalization
-> probability packet
-> bounded probability FIFO
-> sV outer-product consumer
```

sV can consume earlier probabilities while the SFU produces later probabilities. sV starts only after QK has retired and the inner-to-outer mode switch is complete.
