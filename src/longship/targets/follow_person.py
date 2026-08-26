from __future__ import annotations

import time
from collections.abc import Callable

from longship.contracts.skills.follow_person import FollowCommand, MotionReceipt
from longship.targets.unitree_sdk2 import (
    LocomotionLease,
    UnitreeG1HighLevelAdapter,
    VelocitySetpoint,
)


class UnitreeFollowMotion:
    """Map approved Follow commands onto Longship's bounded G1 adapter."""

    def __init__(
        self,
        target: UnitreeG1HighLevelAdapter,
        *,
        lease_ttl_s: float = 1.0,
        lease_epoch: int = 1,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not 0.2 <= lease_ttl_s <= target.max_lease_ttl_s:
            raise ValueError("follow lease TTL is outside the target adapter bounds")
        if type(lease_epoch) is not int or lease_epoch < 0:
            raise ValueError("lease epoch must be a non-negative integer")
        self._target = target
        self._lease_ttl_ns = int(lease_ttl_s * 1_000_000_000)
        self._lease_epoch = lease_epoch
        self._clock_ns = clock_ns
        self._lease: LocomotionLease | None = None
        self._session_id: str | None = None

    def acquire(self, session_id: str, now_ns: int) -> MotionReceipt:
        if self._lease is not None:
            return MotionReceipt(False, "follow motion authority is already held")
        lease = LocomotionLease(
            lease_id=f"{session_id}:whole-body",
            epoch=self._lease_epoch,
            issued_monotonic_ns=now_ns,
            expires_monotonic_ns=now_ns + self._lease_ttl_ns,
        )
        try:
            self._target.arm(lease)
        except Exception as exc:
            return MotionReceipt(False, f"target arm failed: {type(exc).__name__}")
        self._lease = lease
        self._session_id = session_id
        return MotionReceipt(True, "whole-body motion lease acquired")

    def apply(self, command: FollowCommand) -> MotionReceipt:
        lease = self._lease
        if lease is None or command.session_id != self._session_id:
            return MotionReceipt(False, "command does not own the follow lease")
        if command.issued_monotonic_ns >= lease.expires_monotonic_ns - (
            self._lease_ttl_ns // 3
        ):
            refreshed = LocomotionLease(
                lease_id=lease.lease_id,
                epoch=lease.epoch,
                issued_monotonic_ns=command.issued_monotonic_ns,
                expires_monotonic_ns=command.issued_monotonic_ns + self._lease_ttl_ns,
                actuator_scope=lease.actuator_scope,
            )
            try:
                self._target.refresh_lease(refreshed)
            except Exception as exc:
                return MotionReceipt(
                    False, f"lease refresh failed: {type(exc).__name__}"
                )
            self._lease = refreshed
            lease = refreshed
        setpoint = VelocitySetpoint(
            command_id=f"{command.session_id}:{command.sequence}",
            lease_id=lease.lease_id,
            lease_epoch=lease.epoch,
            sequence=command.sequence,
            issued_monotonic_ns=command.issued_monotonic_ns,
            expires_monotonic_ns=command.expires_monotonic_ns,
            vx_mps=command.forward_mps,
            vy_mps=0.0,
            yaw_rate_radps=command.yaw_rate_radps,
        )
        try:
            receipt = self._target.command(setpoint)
        except Exception as exc:
            return MotionReceipt(False, f"target command failed: {type(exc).__name__}")
        return MotionReceipt(receipt.accepted, receipt.reason)

    def protective_stop(self, reason: str) -> MotionReceipt:
        try:
            receipt = self._target.stop(reason)
        except Exception as exc:
            return MotionReceipt(False, f"target stop failed: {type(exc).__name__}")
        self._lease = None
        return MotionReceipt(
            receipt.accepted,
            (
                "zero velocity accepted, but a separately qualified motion monitor "
                "must still verify stopped state"
            ),
            verified_stopped=False,
        )

    def close(self) -> None:
        self._target.close()


class RecordingMotion:
    """Deterministic target fake for simulation and downstream tests."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.commands: list[FollowCommand] = []
        self.stop_reasons: list[str] = []
        self.current_command: FollowCommand | None = None

    def acquire(self, session_id: str, now_ns: int) -> MotionReceipt:
        del now_ns
        if self.session_id is not None:
            return MotionReceipt(False, "motion is already acquired")
        self.session_id = session_id
        return MotionReceipt(True, "mock lease acquired")

    def apply(self, command: FollowCommand) -> MotionReceipt:
        if command.session_id != self.session_id:
            return MotionReceipt(False, "mock lease mismatch")
        self.commands.append(command)
        self.current_command = command
        return MotionReceipt(True, "mock command accepted")

    def protective_stop(self, reason: str) -> MotionReceipt:
        self.stop_reasons.append(reason)
        self.current_command = None
        return MotionReceipt(True, "mock zero-motion evidence", verified_stopped=True)
