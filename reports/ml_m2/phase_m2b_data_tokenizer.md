# ML-M2B Report: Dataset and Tokenizer Pipeline

## Result

ML-M2B adds a dataset and tokenizer pipeline for Model Stage M2 without
committing datasets, tokenizer caches, or large artifacts.

## Data Sources

TinyStories:

- Source: Hugging Face dataset `roneneldan/TinyStories`
- URL: https://huggingface.co/datasets/roneneldan/TinyStories
- Revision tracked by scripts: `main`
- Access date: 2026-07-13
- License recorded on HF card: `cdla-sharing-1.0`
- Files referenced: `TinyStories-train.txt`, `TinyStories-valid.txt`
- Git policy: full dataset must remain under `VEDA_ML_DATA_ROOT`

Tiny Shakespeare smoke:

- Source: Hugging Face dataset `karpathy/tiny_shakespeare`
- URL: https://huggingface.co/datasets/karpathy/tiny_shakespeare
- Access date: 2026-07-13
- Use: smoke-data option only
- Git policy: only repository-authored small fixture text is committed

## Implemented Files

- `ml/data/dataset_manifest.py`
- `ml/data/tinystories_loader.py`
- `ml/data/tiny_shakespeare_loader.py`
- `ml/data/sequence_builder.py`
- `ml/data/dataset_hash.py`
- `ml/data/fixtures.py`
- `ml/tokenizer/train_bpe.py`
- `ml/tokenizer/load_tokenizer.py`
- `ml/tokenizer/validate_tokenizer.py`
- `ml/tokenizer/tokenizer_manifest.py`
- `scripts/ml/run_ml_m2_data_tests.py`
- `ml/tests/test_data_tokenizer.py`

## Tokenizer Contract

The first tokenizer is deterministic pure-Python BPE:

```text
PAD=0
BOS=1
EOS=2
UNK=3
default_vocab_size=2048
optional_vocab_size=4096
```

The tokenizer JSON and manifest are written to the artifact directory. The
repository commits code and tests only, not trained tokenizer artifacts.

## Tests

Unit tests cover:

- TinyStories local text splitting and manifest metadata;
- Tiny Shakespeare fixture manifest;
- deterministic train/validation/test split;
- causal LM sequence packing;
- deterministic BPE training;
- tokenizer save/load/validate;
- BOS/EOS/PAD/UNK IDs.

