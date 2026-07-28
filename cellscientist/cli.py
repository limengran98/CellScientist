"""Command-line entry points for the clean CellScientist release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from .artifacts import atomic_write_json
from .data import load_task_data
from .lca_audit import run_lca_audit
from .llm_client import OpenAICompatiblePolicy
from .protocol import load_config, task_specs, write_lock
from .routing_audit import run_routing_audit
from .runner import run_matrix, run_one


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="validate data and show partitions")
    inspect.add_argument("--config", type=Path, required=True)

    preflight = commands.add_parser("preflight", help="check Python, CUDA, and optionally LLM access")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--check-llm", action="store_true")

    freeze = commands.add_parser("freeze", help="write a reproducibility lock")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--lock", type=Path, required=True)

    run = commands.add_parser("run", help="run one CellScientist trajectory")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--lock", type=Path)
    run.add_argument("--task", required=True)
    run.add_argument("--seed", type=int, required=True)

    matrix = commands.add_parser("run-matrix", help="run the registered CellScientist exploration matrix")
    matrix.add_argument("--config", type=Path, required=True)
    matrix.add_argument("--lock", type=Path)
    matrix.add_argument("--tasks", help="comma-separated task IDs")
    matrix.add_argument("--seeds", help="comma-separated seeds")
    matrix.add_argument("--jobs", type=int, default=1)

    routing = commands.add_parser("routing-audit", help="run registered component-routing audit")
    routing.add_argument("--output", type=Path, default=Path("audit_outputs/routing_audit.json"))
    routing.add_argument("--seed", type=int, default=11)
    routing.add_argument("--config", type=Path, help="optional LLM configuration")

    lca = commands.add_parser("lca-audit", help="run protected-contract audit")
    lca.add_argument("--output", type=Path, default=Path("audit_outputs/lca_audit.json"))
    return parser


def _csv_ints(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _csv_strings(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root()
    if args.command == "inspect":
        config = load_config(args.config)
        summary = [
            load_task_data(spec, config["data"], config["search"]).partition_summary
            for spec in task_specs(config)
        ]
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if args.command == "preflight":
        config = load_config(args.config)
        result = {
            "llm_enabled": bool(config["llm"].get("enabled", False)),
            "llm_credential_present": bool(os.environ.get(config["llm"]["api_key_env"])),
        }
        try:
            import torch

            result.update(
                {
                    "torch": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_count": int(torch.cuda.device_count()),
                }
            )
        except ImportError:
            result.update({"torch": "unavailable", "cuda_available": False, "cuda_device_count": 0})
        if args.check_llm:
            policy = OpenAICompatiblePolicy(config["llm"])
            decision = policy.choose(
                '{"candidate_id":"health","address":"conditioning"}',
                ["health"],
                ["conditioning"],
            )
            result["llm_health"] = {"response_sha256": decision.response_sha256}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "freeze":
        print(json.dumps(write_lock(args.config, args.lock, root), indent=2, ensure_ascii=False))
        return 0
    if args.command == "run":
        value = run_one(
            root=root,
            config_path=args.config,
            task_id=args.task,
            controller="cellscientist",
            initialization="standard_h0",
            seed=args.seed,
            lock_path=args.lock,
        )
        print(json.dumps(value["test_metrics"], indent=2))
        return 0
    if args.command == "run-matrix":
        run_matrix(
            root=root,
            config_path=args.config,
            lock_path=args.lock,
            task_filter=_csv_strings(args.tasks),
            controller_filter=["cellscientist"],
            initialization_filter=["standard_h0"],
            seed_filter=_csv_ints(args.seeds),
            jobs=args.jobs,
        )
        return 0
    output = args.output if args.output.is_absolute() else root / args.output
    if args.command == "routing-audit":
        policy = None
        if args.config is not None:
            llm_config = load_config(args.config)["llm"]
            key_present = bool(os.environ.get(str(llm_config["api_key_env"])))
            endpoint_env = str(llm_config.get("base_url_env", ""))
            endpoint_present = bool(os.environ.get(endpoint_env)) if endpoint_env else bool(llm_config.get("base_url"))
            if bool(llm_config.get("enabled", False)) and key_present and endpoint_present:
                policy = OpenAICompatiblePolicy(llm_config)
        value = run_routing_audit(args.seed, policy)
    else:
        value = run_lca_audit()
    atomic_write_json(output, value)
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0
