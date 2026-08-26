from __future__ import annotations

import argparse
import threading
import time

from longship.rl.sim2sim.dds import DdsContract, check_host


def roundtrip(contract: DdsContract, timeout: float = 5.0) -> tuple[int, int, int]:
    """Exchange real Unitree HG messages through CycloneDDS.

    This intentionally uses the same message classes and topics as the three
    simulation backends. It is a transport test, not an in-memory mock.
    """

    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__IMUState_,
        unitree_hg_msg_dds__LowCmd_,
        unitree_hg_msg_dds__LowState_,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

    # Unitree SDK2 raises when initialization fails and returns ``None`` on
    # success.  Do not interpret its return value as a boolean status.
    ChannelFactoryInitialize(contract.domain_id, contract.interface)

    state_event = threading.Event()
    command_event = threading.Event()
    secondary_imu_event = threading.Event()
    received = {"state": 0, "command": 0, "secondary_imu": 0}

    def on_state(message: object) -> None:
        if int(message.tick) == 4242:  # type: ignore[attr-defined]
            received["state"] += 1
            state_event.set()

    def on_command(message: object) -> None:
        motor = message.motor_cmd[0]  # type: ignore[attr-defined]
        if abs(float(motor.q) - 0.125) < 1.0e-6:
            received["command"] += 1
            command_event.set()

    def on_secondary_imu(message: object) -> None:
        quaternion = message.quaternion  # type: ignore[attr-defined]
        if abs(float(quaternion[0]) - 1.0) < 1.0e-6:
            received["secondary_imu"] += 1
            secondary_imu_event.set()

    state_publisher = ChannelPublisher(contract.lowstate_topic, LowState_)
    state_publisher.Init()
    command_publisher = ChannelPublisher(contract.lowcmd_topic, LowCmd_)
    command_publisher.Init()
    secondary_imu_publisher = ChannelPublisher(contract.secondary_imu_topic, IMUState_)
    secondary_imu_publisher.Init()
    state_subscriber = ChannelSubscriber(contract.lowstate_topic, LowState_)
    state_subscriber.Init(on_state, 4)
    command_subscriber = ChannelSubscriber(contract.lowcmd_topic, LowCmd_)
    command_subscriber.Init(on_command, 4)
    secondary_imu_subscriber = ChannelSubscriber(contract.secondary_imu_topic, IMUState_)
    secondary_imu_subscriber.Init(on_secondary_imu, 4)

    state = unitree_hg_msg_dds__LowState_()
    state.tick = 4242
    command = unitree_hg_msg_dds__LowCmd_()
    command.motor_cmd[0].q = 0.125
    secondary_imu = unitree_hg_msg_dds__IMUState_()
    secondary_imu.quaternion[0] = 1.0

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not (
        state_event.is_set() and command_event.is_set() and secondary_imu_event.is_set()
    ):
        state_publisher.Write(state)
        command_publisher.Write(command)
        secondary_imu_publisher.Write(secondary_imu)
        time.sleep(0.02)
    if not state_event.is_set() or not command_event.is_set() or not secondary_imu_event.is_set():
        raise TimeoutError(
            "DDS roundtrip timed out: "
            f"lowstate={state_event.is_set()}, lowcmd={command_event.is_set()}, "
            f"secondary_imu={secondary_imu_event.is_set()}"
        )
    return received["state"], received["command"], received["secondary_imu"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Unitree DDS loopback roundtrip")
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    contract = DdsContract(domain_id=args.domain_id, interface=args.interface)
    contract.validate()
    host = check_host(contract)
    for message in host.checks:
        print(f"ok: {message}")
    for message in host.blockers:
        print(f"blocker: {message}")
    if not host.ready:
        return 2
    state_count, command_count, secondary_imu_count = roundtrip(contract, args.timeout)
    print(
        "DDS READY: "
        f"{contract.lowstate_topic} received={state_count}, "
        f"{contract.lowcmd_topic} received={command_count}, "
        f"{contract.secondary_imu_topic} received={secondary_imu_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
