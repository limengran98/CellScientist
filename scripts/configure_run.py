"""Create an independent run configuration for a chosen LLM and compute backend."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("configs/bbbc036_047_formal.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Unique experiment name used for protocol and output paths")
    parser.add_argument("--model", required=True, help="Model ID accepted by your API endpoint")
    parser.add_argument("--backend", choices=("torch_cuda", "torch_cpu", "sklearn"))
    parser.add_argument("--local-api", action="store_true", help="Allow an endpoint that runs without an API key")
    parser.add_argument("--token-parameter", choices=("max_tokens", "max_completion_tokens"))
    parser.add_argument("--omit-temperature", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        parser.error("--name uses letters, digits, underscores and hyphens")
    if not args.model.strip():
        parser.error("--model must be nonempty")
    config = json.loads(args.base.read_text(encoding="utf-8"))
    config["protocol_id"] = f"cellscientist_{args.name}"
    config["llm"].update(model=args.model, enabled=True, selection_status="frozen")
    if args.local_api:
        config["llm"]["require_credential"] = False
    if args.token_parameter:
        config["llm"]["max_tokens_parameter"] = args.token_parameter
    if args.omit_temperature:
        config["llm"]["temperature"] = None
    if args.backend:
        config["model"]["backend"] = args.backend
    config["outputs"] = {"root": f"outputs/{args.name}", "cache": f"outputs/{args.name}_cache"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print(f"Configuration: {args.output}")
    print(f"Protocol: {config['protocol_id']}")
    print(f"Results: {config['outputs']['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
