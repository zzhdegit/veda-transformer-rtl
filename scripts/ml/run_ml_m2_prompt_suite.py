"""Run the fixed ML-M2 interactive prompt suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.inference.interactive import DEFAULT_ARTIFACT_ROOT, evaluate_prompt_suite, load_interactive_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--prompt-suite", default="ml/evaluation/ml_m2_prompt_suite.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    prompts = json.loads(Path(args.prompt_suite).read_text(encoding="utf-8"))["prompts"]
    bundle = load_interactive_bundle(args.artifact_root)
    report = evaluate_prompt_suite(bundle, prompts, output_path=args.output or None)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"output_path: {report['output_path']}")
        for name, summary in report["variant_summary"].items():
            print(
                f"{name}: repeat={summary['average_repeat_rate']:.4f} "
                f"distinct1={summary['average_distinct_1']:.4f} "
                f"distinct2={summary['average_distinct_2']:.4f} "
                f"eos={summary['eos_rate']:.4f} "
                f"entropy={summary['average_entropy']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
