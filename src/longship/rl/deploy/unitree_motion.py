#!/usr/bin/env python3
"""Inspect or release Unitree's high-level motion service before LowCmd control."""

from __future__ import annotations

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if not 0 <= args.domain_id <= 232 or args.timeout <= 0:
        parser.error("domain-id must be 0..232 and timeout must be positive")

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(args.domain_id, args.interface)
    client = MotionSwitcherClient()
    client.SetTimeout(args.timeout)
    client.Init()
    status, result = client.CheckMode()
    if status != 0:
        raise RuntimeError(f"Unitree CheckMode failed with status {status}: {result}")
    active = str((result or {}).get("name", ""))
    print(f"UNITREE MOTION MODE: {active or 'released'}", flush=True)
    if not args.release:
        return 0
    deadline = time.monotonic() + args.timeout
    while active and time.monotonic() < deadline:
        status, release_result = client.ReleaseMode()
        if status != 0:
            raise RuntimeError(
                f"Unitree ReleaseMode failed with status {status}: {release_result}"
            )
        time.sleep(0.2)
        status, result = client.CheckMode()
        if status != 0:
            raise RuntimeError(f"Unitree CheckMode failed with status {status}: {result}")
        active = str((result or {}).get("name", ""))
    if active:
        raise RuntimeError(f"Unitree motion mode remained active: {active}")
    print("UNITREE MOTION MODE RELEASED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
