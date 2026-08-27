from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from longship.rl.deploy.launch import build_deployment_launch
from longship.rl.deploy.profile import (
    DeploymentProfile,
    bundled_deployment_profile,
)


def _override_sensor_serial(
    profile: DeploymentProfile, serial: str
) -> DeploymentProfile:
    sensors = list(profile.sensors)
    indices = [i for i, sensor in enumerate(sensors) if sensor.type == "realsense_depth_dds"]
    if len(indices) != 1:
        raise ValueError("--camera-serial requires exactly one RealSense sensor")
    index = indices[0]
    sensor = sensors[index]
    sensors[index] = replace(sensor, config={**sensor.config, "serial": serial})
    return replace(profile, sensors=tuple(sensors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a registered physical RL deployment")
    parser.add_argument("profile", help="profile name, or path to a deployment YAML")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--interface")
    parser.add_argument("--domain-id", type=int)
    parser.add_argument("--camera-serial")
    parser.add_argument("--visualization-bind-host")
    parser.add_argument("--visualization-port", type=int)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--print-command", "--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    candidate = Path(args.profile)
    profile_path = (
        candidate.resolve()
        if candidate.suffix in (".yaml", ".yml") or candidate.exists()
        else bundled_deployment_profile(args.profile)
    )
    profile = DeploymentProfile.load(profile_path, root)
    if args.interface is not None:
        profile = replace(profile, dds=replace(profile.dds, interface=args.interface))
    if args.domain_id is not None:
        profile = replace(profile, dds=replace(profile.dds, domain_id=args.domain_id))
    if args.camera_serial is not None:
        profile = _override_sensor_serial(profile, args.camera_serial)
    if args.no_visualization and profile.visualization is not None:
        profile = replace(
            profile, visualization=replace(profile.visualization, enabled=False)
        )
    if args.visualization_bind_host is not None or args.visualization_port is not None:
        if profile.visualization is None:
            raise ValueError("visualization overrides require a visualization profile")
        profile = replace(
            profile,
            visualization=replace(
                profile.visualization,
                bind_host=(
                    profile.visualization.bind_host
                    if args.visualization_bind_host is None
                    else args.visualization_bind_host
                ),
                port=(
                    profile.visualization.port
                    if args.visualization_port is None
                    else args.visualization_port
                ),
            ),
        )
    profile.validate()
    launch = build_deployment_launch(root, args.python, profile)
    # Keep --help and profile parsing independent of heavyweight policy runtimes.
    from longship.rl.deploy.runner import preflight, run

    checks = preflight(root, launch)
    for check in checks:
        print(f"PASS: {check}")
    if args.print_command:
        if launch.monitor is not None:
            print(f"monitor: {launch.monitor.shell_command()}")
        for sensor in launch.sensors:
            print(f"sensor.{sensor.process.name}: {sensor.process.shell_command()}")
        print(f"controller: {launch.backend.controller.shell_command()}")
        print(f"release_motion: {launch.release_motion.shell_command()}")
        print(f"teleop: {launch.backend.teleop.shell_command()}")
        return 0
    return run(root, launch)


if __name__ == "__main__":
    raise SystemExit(main())
