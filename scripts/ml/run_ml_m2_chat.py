"""Interactive text generation CLI for the accepted ML-M2 formal model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.inference.interactive import DEFAULT_ARTIFACT_ROOT, ChatSession, GenerationConfig, load_interactive_bundle


HELP = """Commands:
  /help
  /config
  /clear
  /seed <n>
  /greedy
  /sample
  /temperature <value>
  /top-k <value>
  /top-p <value>
  /repetition-penalty <value>
  /max-tokens <value>
  /quit
"""


def _print_record(record: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(record, indent=2, sort_keys=True))
        return
    print(f"prompt_text: {record['prompt_text']}")
    print(f"prompt_token_ids: {record['prompt_token_ids']}")
    print(f"prompt_tokens: {record['prompt_tokens']}")
    print(f"prompt_length: {record['prompt_length']}")
    print(f"generated_token_ids: {record['generated_token_ids']}")
    print(f"generated_tokens: {record['generated_tokens']}")
    print(f"generated_text: {record['generated_text']}")
    print(f"kv_cache_length: {record['kv_cache_length']}")
    print(f"generation_time_seconds: {record['generation_time_seconds']:.6f}")
    print(f"tokens_per_second: {record['tokens_per_second']:.3f}")


def _handle_command(session: ChatSession, command: str) -> bool:
    parts = command.strip().split()
    if not parts:
        return True
    name = parts[0].lower()
    if name == "/help":
        print(HELP)
    elif name == "/config":
        print(json.dumps(session.config.__dict__, indent=2, sort_keys=True))
    elif name == "/clear":
        session.clear()
        print("state cleared")
    elif name == "/seed" and len(parts) == 2:
        session.set_seed(int(parts[1]))
        print(f"seed={session.config.seed}")
    elif name == "/greedy":
        session.config.mode = "greedy"
        print("mode=greedy")
    elif name == "/sample":
        session.config.mode = "sample"
        print("mode=sample")
    elif name == "/temperature" and len(parts) == 2:
        session.config.temperature = float(parts[1])
        print(f"temperature={session.config.temperature}")
    elif name == "/top-k" and len(parts) == 2:
        session.config.top_k = int(parts[1])
        print(f"top_k={session.config.top_k}")
    elif name == "/top-p" and len(parts) == 2:
        session.config.top_p = float(parts[1])
        print(f"top_p={session.config.top_p}")
    elif name == "/repetition-penalty" and len(parts) == 2:
        session.config.repetition_penalty = float(parts[1])
        print(f"repetition_penalty={session.config.repetition_penalty}")
    elif name == "/max-tokens" and len(parts) == 2:
        session.config.max_new_tokens = int(parts[1])
        print(f"max_new_tokens={session.config.max_new_tokens}")
    elif name == "/quit":
        return False
    else:
        print("unknown command; use /help")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mode", choices=["greedy", "sample"], default="sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    bundle = load_interactive_bundle(args.artifact_root)
    config = GenerationConfig(
        mode=args.mode,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
    )
    session = ChatSession(bundle, config)
    if args.prompt:
        _print_record(session.generate(args.prompt), args.json)
        return 0
    print("ML-M2 interactive generation. Use /help for commands.")
    while True:
        try:
            text = input("Prompt> ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text.startswith("/"):
            if not _handle_command(session, text):
                break
            continue
        _print_record(session.generate(text), args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
