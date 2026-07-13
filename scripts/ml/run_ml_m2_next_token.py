"""Show next-token probabilities for the accepted ML-M2 formal model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.inference.interactive import DEFAULT_ARTIFACT_ROOT, load_interactive_bundle, next_token_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    bundle = load_interactive_bundle(args.artifact_root)
    report = next_token_report(bundle, args.prompt, top_k=args.top_k)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"prompt: {report['prompt_text']}")
    print(f"token_ids: {report['tokenized_prompt']['token_ids']}")
    print(f"tokens: {report['tokenized_prompt']['tokens']}")
    print(f"last_position: {report['last_position']}")
    print(f"entropy: {report['entropy']:.6f}")
    print(f"special_token_probability: {report['special_token_probability']:.8f}")
    print(f"eos_probability: {report['eos_probability']:.8f}")
    for title in ["top_1", "top_5", "top_10"]:
        print(title + ":")
        for row in report[title]:
            print(
                f"  id={row['token_id']} token={row['token']!r} "
                f"logit={row['logit']:.6f} prob={row['probability']:.8f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
