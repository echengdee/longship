from __future__ import annotations

import argparse
from pathlib import Path

from longship.rl.sim2sim.dds import DdsContract, check_host
from longship.rl.sim2sim.launch import backend_launch
from longship.rl.sim2sim.preflight import preflight_all, preflight_backend


def main() -> int:
    parser = argparse.ArgumentParser(description="Longship RL Sim2Sim tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="validate models and simulator assets")
    preflight.add_argument("backend", choices=["all", "holosoma", "sonic", "instinctlab", "php"])
    preflight.add_argument("--root", type=Path, default=Path.cwd())
    dds = subparsers.add_parser("dds-check", help="check host support for DDS loopback")
    dds.add_argument("--interface", default="lo")
    dds.add_argument("--domain-id", type=int, default=0)
    launch = subparsers.add_parser("commands", help="print simulator, controller, and keyboard commands")
    launch.add_argument("backend", choices=["holosoma", "sonic", "instinctlab", "php"])
    launch.add_argument("--root", type=Path, default=Path.cwd())
    launch.add_argument("--python", default="python")
    args = parser.parse_args()

    if args.command == "dds-check":
        contract = DdsContract(domain_id=args.domain_id, interface=args.interface)
        contract.validate()
        result = check_host(contract)
        for message in result.checks:
            print(f"ok: {message}")
        for message in result.blockers:
            print(f"blocker: {message}")
        if result.ready:
            print("Run `python -m longship.rl.sim2sim.dds_probe` for a real message roundtrip.")
        return 0 if result.ready else 2

    if args.command == "commands":
        launch = backend_launch(args.root, args.backend, args.python)
        print(
            f"DDS domain={launch.contract.domain_id} interface={launch.contract.interface} "
            f"state={launch.contract.lowstate_topic} command={launch.contract.lowcmd_topic}"
        )
        if launch.contract.depth_topic:
            print(f"depth={launch.contract.depth_topic}")
        print(f"terminal 1 ({launch.simulator.name}): {launch.simulator.shell_command()}")
        print(f"terminal 2 ({launch.controller.name}): {launch.controller.shell_command()}")
        print(f"terminal 3 ({launch.teleop.name}): {launch.teleop.shell_command()}")
        for note in launch.notes:
            print(f"note: {note}")
        return 0

    results = preflight_all(args.root) if args.backend == "all" else (preflight_backend(args.root, args.backend),)
    for result in results:
        state = "READY" if result.ready else "BLOCKED"
        print(f"{result.backend}: {state}")
        for message in result.checks:
            print(f"  ok: {message}")
        for message in result.blockers:
            print(f"  blocker: {message}")
    return 0 if all(result.ready for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
