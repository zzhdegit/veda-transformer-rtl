"""Tiny Shakespeare smoke dataset helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from ml.data.dataset_hash import sha256_text
from ml.data.dataset_manifest import ACCESS_DATE, DatasetManifest
from ml.data.fixtures import fixture_text


TINY_SHAKESPEARE_HF_URL = "https://huggingface.co/datasets/karpathy/tiny_shakespeare"
TINY_SHAKESPEARE_RAW_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_tiny_shakespeare(path: str | Path | None = None) -> str:
    if path is None:
        return fixture_text()
    return Path(path).read_text(encoding="utf-8", errors="replace")


def download_tiny_shakespeare(output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(TINY_SHAKESPEARE_RAW_URL, target)
    return target


def make_tiny_shakespeare_manifest(path: str | Path | None = None) -> DatasetManifest:
    text = load_tiny_shakespeare(path)
    return DatasetManifest(
        name="TinyShakespeare-smoke" if path else "builtin_fixture",
        source_type="huggingface_dataset" if path else "repository_fixture",
        source_url=TINY_SHAKESPEARE_HF_URL if path else "ml/data/fixtures.py",
        revision="main" if path else "local",
        access_date=ACCESS_DATE,
        license="not fully specified on HF card; fixture text is repository-authored",
        local_path=str(path) if path else None,
        sha256=sha256_text(text),
        num_documents=1,
        num_characters=len(text),
        notes=[
            "HF card describes Tiny Shakespeare as 40,000 lines and points to karpathy/char-rnn.",
            "Built-in fixture is used for offline unit tests and smoke tests.",
        ],
    )

