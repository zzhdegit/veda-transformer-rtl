"""TinyStories local-file loader and manifest definitions.

This module does not download TinyStories by default. The full files are large
and must be stored under `VEDA_ML_DATA_ROOT`, not in Git.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from ml.data.dataset_hash import sha256_file
from ml.data.dataset_manifest import ACCESS_DATE, DatasetManifest


TINYSTORIES_DATASET_ID = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "main"
TINYSTORIES_LICENSE = "cdla-sharing-1.0"
TINYSTORIES_CARD_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
TINYSTORIES_TRAIN_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
)
TINYSTORIES_VALID_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
)


def split_tinystories_text(text: str) -> list[str]:
    marker = "<|endoftext|>"
    if marker in text:
        stories = [part.strip() for part in text.split(marker)]
    else:
        stories = [part.strip() for part in text.splitlines() if part.strip()]
    return [story for story in stories if story]


def load_tinystories_local(path: str | Path, limit: int | None = None) -> list[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    stories = split_tinystories_text(text)
    return stories if limit is None else stories[:limit]


def download_tinystories_file(url: str, output_path: str | Path) -> Path:
    """Explicit download helper for user-invoked dataset setup commands."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, target)
    return target


def make_tinystories_manifest(path: str | Path | None = None, split: str = "train") -> DatasetManifest:
    local_path = str(path) if path else None
    sha256 = sha256_file(path) if path else None
    num_documents = None
    num_characters = None
    if path:
        stories = load_tinystories_local(path)
        num_documents = len(stories)
        num_characters = sum(len(story) for story in stories)
    return DatasetManifest(
        name=f"TinyStories-{split}",
        source_type="huggingface_dataset",
        source_url=TINYSTORIES_CARD_URL,
        revision=TINYSTORIES_REVISION,
        access_date=ACCESS_DATE,
        license=TINYSTORIES_LICENSE,
        local_path=local_path,
        sha256=sha256,
        num_documents=num_documents,
        num_characters=num_characters,
        notes=[
            "HF card lists license cdla-sharing-1.0 and files TinyStories-train.txt/TinyStories-valid.txt.",
            "Full dataset files are not committed to Git.",
        ],
    )

