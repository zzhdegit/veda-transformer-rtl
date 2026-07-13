"""RTL stream export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_fp16_hex_stream(array: np.ndarray, path: str | Path) -> None:
    values = array.astype(np.float16).view(np.uint16).reshape(-1)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{int(value):04x}\n" for value in values), encoding="ascii")

