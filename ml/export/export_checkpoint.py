"""Load a checkpoint and export FP16 weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.export.export_fp16_weights import export_fp16_weights
from ml.training.checkpoint import load_checkpoint


def export_checkpoint(checkpoint_path: str | Path, output_dir: str | Path):
    payload_config = None
    import torch

    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    payload_config = payload["config"]
    config = HardwareMatchedConfig.from_json_dict(payload_config)
    model = HardwareMatchedCausalLM(config)
    load_checkpoint(checkpoint_path, model)
    return export_fp16_weights(model, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = export_checkpoint(args.checkpoint, args.output_dir)
    print(f"exported_tensors={manifest.tensor_count}")


if __name__ == "__main__":
    main()

