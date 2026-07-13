# ML-Q1 Tokenizer Audit

```text
vocab_size=2048
special_token_ids={'BOS': 1, 'EOS': 2, 'PAD': 0, 'UNK': 3}
average_chars_per_token_train_sample=2.2503083720781825
average_tokens_per_story_train_sample=385.4921
space_token_ratio=0.4249614972654433
single_character_token_count=90
leading_space_word_token_count=2
merge_count=1954
merge_utilization=0.9559686888454012
unk_ratio_train_sample=0.0030913733381306647
unk_ratio_validation=2.478416435095368e-06
train_validation_distribution_l1=0.06280456841398585
decode_reencode_stable_ratio_train=0.9267
decode_reencode_stable_ratio_validation=0.9995
```

Space conclusion: The tokenizer uses a standalone high-frequency space token; this is expected for the simple character-seeded BPE and is not by itself a tokenizer bug.

Prompt encoding for `Once upon a time`:

```text
ids=[1, 230, 5, 240, 5, 54, 5, 204]
tokens=['<bos>', 'Once', ' ', 'upon', ' ', 'a', ' ', 'time']
roundtrip=Once upon a time
```

Most common tokens and bigrams are recorded in `D:/IC_Workspace/VEDA_artifacts/ml_q1/audits/tokenizer_audit.json`.
