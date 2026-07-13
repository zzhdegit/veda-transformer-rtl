"""Compare two checkpoints by SHA and selected metadata."""

from __future__ import annotations

from pathlib import Path

from ml.data.dataset_hash import sha256_file


def checkpoint_file_summary(path: str | Path) -> dict[str, str | int]:
    target = Path(path)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }

