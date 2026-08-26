#!/usr/bin/env python3
"""ZMQ keyboard command publisher for interactive Sim2Sim validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
import termios
import time
import tty
from typing import Iterator

import zmq


DEFAULT_ENDPOINT = "tcp://127.0.0.1:5560"


@dataclass(frozen=True, slots=True)
class TeleopCapabilities:
    keys: frozenset[str]
    descriptions: tuple[str, ...]


CAPABILITIES = {
    "holosoma": TeleopCapabilities(
        frozenset("i]=wasdqe"),
        (
            "=: toggle stand/walk after policy enable",
            "W/S: forward/backward",
            "A/D: lateral left/right",
            "Q/E: turn left/right in place",
        ),
    ),
    "sonic": TeleopCapabilities(
        frozenset("i]1234567890np-=rwasdqe"),
        (
            "N/P: next/previous SONIC mode set",
            "1-8: select a mode inside the current set",
            "Default set: 1 slow, 2 walk, 3 run, 4 jump, 5 stealth, 6 injured",
            "9/0: decrease/increase planner speed",
            "-/=: decrease/increase squat/crawl height",
            "R: SONIC planner emergency stop",
            "W/S: SONIC planner forward/backward direction",
            "A/D: SONIC planner pure lateral direction",
            "Q/E: SONIC planner facing step in place",
        ),
    ),
    "instinctlab": TeleopCapabilities(
        frozenset("i]12npwqe"),
        (
            "N/P: cycle Hiking agent | 1: stand | 2: parkour",
            "W: parkour forward | Q/E: parkour turn left/right in place",
            "S/A/D: unsupported by the released parkour policy",
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class TeleopCommand:
    key: str
    sequence: int
    timestamp: float

    def encode(self) -> bytes:
        return json.dumps(
            {"version": 1, "key": self.key, "sequence": self.sequence, "timestamp": self.timestamp},
            separators=(",", ":"),
        ).encode()

    @classmethod
    def decode(cls, payload: bytes) -> "TeleopCommand":
        values = json.loads(payload)
        if values.get("version") != 1:
            raise ValueError(f"unsupported teleop protocol version {values.get('version')!r}")
        key = str(values["key"]).lower()
        if len(key) != 1:
            raise ValueError("teleop key must contain exactly one character")
        return cls(key, int(values["sequence"]), float(values["timestamp"]))


class TeleopSubscriber:
    def __init__(self, endpoint: str, backend: str) -> None:
        self.capabilities = CAPABILITIES[backend]
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(endpoint)

    def poll(self) -> tuple[TeleopCommand, ...]:
        commands: list[TeleopCommand] = []
        while True:
            try:
                payload = self.socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            command = TeleopCommand.decode(payload)
            if command.key in self.capabilities.keys:
                commands.append(command)
            else:
                print(f"teleop: key {command.key!r} is unsupported by this policy; ignored", flush=True)
        return tuple(commands)


def _terminal_keys() -> Iterator[str]:
    if not sys.stdin.isatty():
        raise RuntimeError("interactive teleop requires a TTY; use --script for automated checks")
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        while True:
            yield sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=sorted(CAPABILITIES))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--script", help="publish these keys instead of reading a terminal")
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.interval < 0:
        raise ValueError("--interval must be non-negative")
    capabilities = CAPABILITIES[args.backend]
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.bind(args.endpoint)
    print(f"Longship teleop: backend={args.backend} endpoint={args.endpoint}")
    if args.backend == "sonic":
        print("I: initialize SONIC | ]: enable tracking (a motion key starts its planner)")
    else:
        print("I: initialize slowly | ]: enable policy (queued if initialization is still running)")
    for description in capabilities.descriptions:
        print(description)
    print("Unsupported keys are published for explicit adapter-side rejection.")
    time.sleep(0.3)  # Allow SUB sockets to finish their subscription handshake.
    source = iter(args.script) if args.script is not None else _terminal_keys()
    try:
        for sequence, raw_key in enumerate(source, 1):
            key = raw_key.lower()
            if key == "\x03":
                break
            if key.isspace() and key != "]":
                continue
            socket.send(TeleopCommand(key, sequence, time.time()).encode())
            supported = "sent" if key in capabilities.keys else "sent (policy will ignore)"
            print(f"key={key!r}: {supported}", flush=True)
            if args.script is not None:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
