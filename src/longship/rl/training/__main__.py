from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from longship.rl.config import ExperimentConfig
from longship.rl.training.runner import ExperimentRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longship-rl-train",
        description="Plan or run a validated Longship RL experiment.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Longship workspace root (default: current directory).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("experiment", type=Path)
        command.add_argument(
            "--output",
            required=True,
            type=Path,
            help="New run directory. The run command refuses to reuse it.",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    experiment = ExperimentConfig.load(args.experiment)
    runner = ExperimentRunner(workspace=args.root)
    if args.command == "plan":
        plan = runner.plan(experiment, args.output)
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        return 0
    checkpoint = runner.run(experiment, args.output)
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
