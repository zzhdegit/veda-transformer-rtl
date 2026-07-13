"""Prepare ML-M2 Formal TinyStories data artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.data.formal_data import prepare_formal_data


def main() -> int:
    train_stories = int(os.environ.get("VEDA_ML_M2_TRAIN_STORIES", "100000"))
    validation_stories = int(os.environ.get("VEDA_ML_M2_VALIDATION_STORIES", "10000"))
    tokenizer_docs = int(os.environ.get("VEDA_ML_M2_TOKENIZER_DOCS", "0"))
    manifest = prepare_formal_data(
        train_stories=train_stories,
        validation_stories=validation_stories,
        tokenizer_docs=None if tokenizer_docs == 0 else tokenizer_docs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
