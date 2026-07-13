"""Inspect intermediate ML-M2 tensors for one prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.inference.interactive import DEFAULT_ARTIFACT_ROOT, inspect_prompt, load_interactive_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--full-tensor", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    bundle = load_interactive_bundle(args.artifact_root)
    report = inspect_prompt(
        bundle,
        args.prompt,
        output_dir=args.output_dir or None,
        full_tensor=args.full_tensor,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"prompt: {report['prompt_text']}")
    print(f"prompt_token_ids: {report['prompt_token_ids']}")
    print(f"prompt_tokens: {report['prompt_tokens']}")
    print(f"output_path: {report['output_path']}")
    for record in report["records"]:
        print(
            f"{record['name']}: shape={record['shape']} dtype={record['dtype']} "
            f"min={record['min']:.6f} max={record['max']:.6f} "
            f"mean={record['mean']:.6f} rms={record['rms']:.6f} "
            f"checksum={record['checksum']}"
        )
        if "values" in record:
            print(f"  values={record['values']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
