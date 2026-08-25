from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
from typing import Any, Mapping


LOWSTATE_TOPIC = "rt/lowstate"
LOWCMD_TOPIC = "rt/lowcmd"
DEPTH_TOPIC = "rt/camera/depth"
SECONDARY_IMU_TOPIC = "rt/secondary_imu"
SIM_CONTROL_TOPIC = "rt/longship/sim_control"
DEFAULT_INTERFACE = "lo"
DEFAULT_DOMAIN_ID = 0

G1_29DOF_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True, slots=True)
class DdsContract:
    """Wire contract shared by all G1 Sim2Sim backends."""

    domain_id: int = DEFAULT_DOMAIN_ID
    interface: str = DEFAULT_INTERFACE
    lowstate_topic: str = LOWSTATE_TOPIC
    lowcmd_topic: str = LOWCMD_TOPIC
    secondary_imu_topic: str = SECONDARY_IMU_TOPIC
    depth_topic: str | None = None
    state_frequency_hz: int = 500
    control_frequency_hz: int = 50
    command_frequency_hz: int = 200

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DdsContract":
        values = dict(values)
        transport_type = values.pop("type", "unitree_sdk2_dds")
        if transport_type != "unitree_sdk2_dds":
            raise ValueError(f"unsupported Sim2Sim transport {transport_type!r}")
        allowed = {field for field in cls.__dataclass_fields__}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown DDS contract fields: {sorted(unknown)}")
        contract = cls(**values)
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.domain_id < 0 or self.domain_id > 232:
            raise ValueError("DDS domain_id must be in [0, 232]")
        if not self.interface:
            raise ValueError("DDS interface must not be empty")
        if self.interface != DEFAULT_INTERFACE:
            raise ValueError(
                "Sim2Sim DDS is restricted to the loopback interface 'lo'; "
                "use deploy configuration for a physical network interface"
            )
        for name in ("state_frequency_hz", "control_frequency_hz", "command_frequency_hz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.command_frequency_hz < self.control_frequency_hz:
            raise ValueError("command_frequency_hz must be >= control_frequency_hz")


@dataclass(frozen=True, slots=True)
class DdsHostCheck:
    ready: bool
    checks: tuple[str, ...]
    blockers: tuple[str, ...]


def check_host(contract: DdsContract) -> DdsHostCheck:
    """Check OS capabilities without initializing CycloneDDS global state."""

    checks: list[str] = []
    blockers: list[str] = []
    try:
        interfaces = {name for _, name in socket.if_nameindex()}
    except OSError as exc:
        blockers.append(f"cannot enumerate network interfaces: {exc}")
    else:
        if contract.interface not in interfaces:
            blockers.append(f"DDS interface {contract.interface!r} does not exist")
        else:
            checks.append(f"DDS interface exists: {contract.interface}")

    try:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp.bind(("127.0.0.1", 0))
            checks.append(f"UDP loopback bind works: {udp.getsockname()[1]}")
        finally:
            udp.close()
    except OSError as exc:
        blockers.append(f"UDP loopback is unavailable: {exc}")

    return DdsHostCheck(not blockers, tuple(checks), tuple(blockers))


def sdk_pythonpath(root: str | Path) -> tuple[Path, ...]:
    """Return the vendored SDK paths needed by the Python DDS probe."""

    workspace = Path(root).resolve()
    return (
        workspace / ".runtime/python",
        workspace
        / "third_party/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python",
    )


def with_sdk_pythonpath(root: str | Path, environment: Mapping[str, str] | None = None) -> dict[str, str]:
    result = dict(os.environ if environment is None else environment)
    existing = result.get("PYTHONPATH")
    paths = [str(path) for path in sdk_pythonpath(root)]
    if existing:
        paths.append(existing)
    result["PYTHONPATH"] = os.pathsep.join(paths)
    return result
