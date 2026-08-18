from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Protocol


class UnitreeAdapterError(RuntimeError):
    pass


class HardwareDisabledError(UnitreeAdapterError):
    pass


class CommandRejectedError(UnitreeAdapterError):
    pass


class _LocoClient(Protocol):
    def SetVelocity(
        self, vx: float, vy: float, omega: float, duration: float = 1.0
    ) -> int:
        ...


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True, slots=True)
class VelocityLimits:
    max_abs_vx_mps: float = 0.0
    max_abs_vy_mps: float = 0.0
    max_abs_yaw_rate_radps: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.max_abs_vx_mps,
            self.max_abs_vy_mps,
            self.max_abs_yaw_rate_radps,
        )
        if not all(_finite_number(value) and value >= 0.0 for value in values):
            raise ValueError("velocity limits must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LocomotionLease:
    """Process-local Longship ownership token, not a Unitree service lease."""

    lease_id: str
    epoch: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    actuator_scope: str = "whole_body_motion"


@dataclass(frozen=True, slots=True)
class VelocitySetpoint:
    command_id: str
    lease_id: str
    lease_epoch: int
    sequence: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    vx_mps: float
    vy_mps: float
    yaw_rate_radps: float
    frame_id: str = "base_link"


@dataclass(frozen=True, slots=True)
class StopEpisode:
    request_id: str
    generation: int
    lease_id: str
    lease_epoch: int
    requested_monotonic_ns: int
    transport_quiesced_monotonic_ns: int | None = None
    zero_velocity_accepted: bool = False


@dataclass(frozen=True, slots=True)
class StoppedEvidence:
    """Post-barrier whole-body evidence from a separately qualified monitor."""

    evidence_id: str
    stop_request_id: str
    stop_generation: int
    target_id: str
    boot_id: str
    clock_domain: str
    lease_id: str
    lease_epoch: int
    window_start_monotonic_ns: int
    window_end_monotonic_ns: int
    max_abs_linear_speed_mps: float
    max_abs_yaw_rate_radps: float
    max_abs_joint_velocity_radps: float
    max_abs_roll_pitch_rate_radps: float


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    command_id: str
    accepted: bool
    sdk_code: int | None
    reason: str
    submitted_monotonic_ns: int
    completed_monotonic_ns: int
    stop_request_id: str | None = None
    stop_generation: int | None = None
    lease_epoch: int | None = None
    executed_or_stopped: bool = False


class UnitreeG1HighLevelAdapter:
    """Experimental wrapper around the official G1 high-level LocoClient.

    This is not a safety-rated stop channel. The SDK timeout is only a transport
    hint and is not a wall-clock bound. Local epochs, a watchdog, correlated
    post-stop evidence, an independent target monitor, and a physical E-stop are
    all required because a synchronous SDK call cannot be preempted here.
    """

    def __init__(
        self,
        client: _LocoClient,
        *,
        target_id: str,
        boot_id: str,
        clock_domain: str = "host.monotonic",
        hardware_enabled: bool = False,
        limits: VelocityLimits | None = None,
        max_command_ttl_s: float = 0.25,
        max_lease_ttl_s: float = 2.0,
        transport_timeout_s: float = 0.05,
        stop_command_duration_s: float = 0.1,
        max_evidence_age_s: float = 0.5,
        min_evidence_window_s: float = 0.1,
        stopped_linear_threshold_mps: float = 0.02,
        stopped_yaw_threshold_radps: float = 0.02,
        stopped_joint_threshold_radps: float = 0.05,
        stopped_roll_pitch_threshold_radps: float = 0.02,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (target_id, boot_id, clock_domain)
        ):
            raise ValueError("target_id, boot_id, and clock_domain are required")
        if type(hardware_enabled) is not bool:
            raise ValueError("hardware_enabled must be boolean")
        if limits is not None and type(limits) is not VelocityLimits:
            raise ValueError("limits must be a VelocityLimits instance")
        self._require_range("max_command_ttl_s", max_command_ttl_s, 0.05, 0.25)
        self._require_range("max_lease_ttl_s", max_lease_ttl_s, 0.05, 10.0)
        self._require_range(
            "transport_timeout_s", transport_timeout_s, 0.001, max_command_ttl_s
        )
        self._require_range(
            "stop_command_duration_s",
            stop_command_duration_s,
            0.05,
            max_command_ttl_s,
        )
        self._require_range("max_evidence_age_s", max_evidence_age_s, 0.05, 2.0)
        self._require_range(
            "min_evidence_window_s",
            min_evidence_window_s,
            0.05,
            max_evidence_age_s,
        )
        threshold_bounds = (
            ("stopped_linear_threshold_mps", stopped_linear_threshold_mps, 0.2),
            ("stopped_yaw_threshold_radps", stopped_yaw_threshold_radps, 0.5),
            ("stopped_joint_threshold_radps", stopped_joint_threshold_radps, 0.5),
            (
                "stopped_roll_pitch_threshold_radps",
                stopped_roll_pitch_threshold_radps,
                0.5,
            ),
        )
        for name, value, upper in threshold_bounds:
            self._require_range(name, value, 0.0, upper)

        self._client = client
        self._target_id = target_id
        self._boot_id = boot_id
        self._clock_domain = clock_domain
        self._hardware_enabled = hardware_enabled
        self._limits = limits or VelocityLimits()
        self._max_command_ttl_s = max_command_ttl_s
        self._max_lease_ttl_s = max_lease_ttl_s
        self._transport_timeout_s = transport_timeout_s
        self._stop_command_duration_s = stop_command_duration_s
        self._max_evidence_age_s = max_evidence_age_s
        self._min_evidence_window_s = min_evidence_window_s
        self._stopped_linear_threshold_mps = stopped_linear_threshold_mps
        self._stopped_yaw_threshold_radps = stopped_yaw_threshold_radps
        self._stopped_joint_threshold_radps = stopped_joint_threshold_radps
        self._stopped_roll_pitch_threshold_radps = (
            stopped_roll_pitch_threshold_radps
        )
        self._clock_ns = clock_ns
        self._active_lease: LocomotionLease | None = None
        self._retained_lease: LocomotionLease | None = None
        self._last_lease_epoch = -1
        self._last_sequence = -1
        self._stop_latched = False
        self._stop_generation = 0
        self._stop_episode: StopEpisode | None = None
        self._pending_stop_calls = 0
        self._watchdog: threading.Timer | None = None
        self._command_generation = 0
        self._state_lock = threading.RLock()
        self._rpc_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self.last_fault: str | None = None
        self.last_stop_receipt: CommandReceipt | None = None

    @staticmethod
    def _require_range(name: str, value: object, lower: float, upper: float) -> None:
        if not _finite_number(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be finite and between {lower} and {upper}")

    @property
    def hardware_enabled(self) -> bool:
        return self._hardware_enabled

    @property
    def limits(self) -> VelocityLimits:
        return self._limits

    @property
    def max_command_ttl_s(self) -> float:
        return self._max_command_ttl_s

    @property
    def max_lease_ttl_s(self) -> float:
        return self._max_lease_ttl_s

    @property
    def transport_timeout_s(self) -> float:
        return self._transport_timeout_s

    @property
    def stop_command_duration_s(self) -> float:
        return self._stop_command_duration_s

    @property
    def max_evidence_age_s(self) -> float:
        return self._max_evidence_age_s

    @property
    def min_evidence_window_s(self) -> float:
        return self._min_evidence_window_s

    @property
    def stopped_linear_threshold_mps(self) -> float:
        return self._stopped_linear_threshold_mps

    @property
    def stopped_yaw_threshold_radps(self) -> float:
        return self._stopped_yaw_threshold_radps

    @property
    def stopped_joint_threshold_radps(self) -> float:
        return self._stopped_joint_threshold_radps

    @property
    def stopped_roll_pitch_threshold_radps(self) -> float:
        return self._stopped_roll_pitch_threshold_radps

    @property
    def armed_lease_id(self) -> str | None:
        with self._state_lock:
            return self._active_lease.lease_id if self._active_lease else None

    @property
    def retained_lease_id(self) -> str | None:
        with self._state_lock:
            return self._retained_lease.lease_id if self._retained_lease else None

    @property
    def stop_latched(self) -> bool:
        with self._state_lock:
            return self._stop_latched

    @property
    def current_stop_episode(self) -> StopEpisode | None:
        with self._state_lock:
            return self._stop_episode

    def arm(self, lease: LocomotionLease) -> None:
        with self._state_lock:
            now = self._clock_ns()
            if not self._hardware_enabled:
                raise HardwareDisabledError("real Unitree commands are disabled by default")
            if self._closed or self._closing:
                raise UnitreeAdapterError("adapter is closed or closing")
            if self._stop_latched:
                raise CommandRejectedError("stop is latched; verified reset is required")
            self._validate_lease(lease, now)
            if self._active_lease is not None:
                raise CommandRejectedError("another process-local ownership token is active")
            if lease.epoch <= self._last_lease_epoch:
                raise CommandRejectedError("lease epoch must increase monotonically")
            self._active_lease = lease
            self._last_lease_epoch = lease.epoch
            self._last_sequence = -1
            self.last_stop_receipt = None

    def command(self, setpoint: VelocitySetpoint) -> CommandReceipt:
        validation_error: CommandRejectedError | HardwareDisabledError | None = None
        validation_episode: StopEpisode | None = None
        with self._state_lock:
            submitted = self._clock_ns()
            try:
                self._validate_setpoint(setpoint, submitted)
            except (CommandRejectedError, HardwareDisabledError) as exc:
                validation_error = exc
                if (
                    self._hardware_enabled
                    and self._active_lease is not None
                    and not self._closed
                    and not self._closing
                ):
                    validation_episode = self._begin_stop_locked(
                        f"invalid command rejected: {type(exc).__name__}"
                    )
            else:
                self._last_sequence = setpoint.sequence
                self._command_generation += 1
                generation = self._command_generation
                self._schedule_watchdog_locked(setpoint.expires_monotonic_ns, generation)

        if validation_error is not None:
            if validation_episode is not None:
                with self._rpc_lock:
                    receipt = self._send_zero_locked(
                        "invalid command rejected", validation_episode, submitted=submitted
                    )
                    with self._state_lock:
                        self._finish_stop_locked(validation_episode, receipt)
            raise validation_error

        needs_zero = False
        episode: StopEpisode | None = None
        code: int | None = None
        with self._rpc_lock:
            with self._state_lock:
                now = self._clock_ns()
                if (
                    self._stop_latched
                    or generation != self._command_generation
                    or now >= setpoint.expires_monotonic_ns
                ):
                    episode = self._begin_stop_locked(
                        "setpoint expired or was revoked before SDK submission"
                    )
                    needs_zero = True
                    duration_s = self.stop_command_duration_s
                else:
                    duration_s = min(
                        (setpoint.expires_monotonic_ns - now) / 1_000_000_000,
                        self.max_command_ttl_s,
                    )
            if not needs_zero:
                try:
                    code = self._client.SetVelocity(
                        setpoint.vx_mps,
                        setpoint.vy_mps,
                        setpoint.yaw_rate_radps,
                        duration_s,
                    )
                except Exception as exc:
                    with self._state_lock:
                        episode = self._begin_stop_locked(
                            "Unitree SetVelocity raised an exception"
                        )
                    receipt = self._send_zero_locked("command exception", episode)
                    with self._state_lock:
                        self._finish_stop_locked(episode, receipt)
                    raise UnitreeAdapterError("Unitree SetVelocity call failed") from exc

                returned = self._clock_ns()
                with self._state_lock:
                    late = returned >= setpoint.expires_monotonic_ns
                    superseded = (
                        self._stop_latched or generation != self._command_generation
                    )
                    sdk_success = type(code) is int and code == 0
                    if not sdk_success or late or superseded:
                        reason = (
                            "sdk rejected or returned an invalid code"
                            if not sdk_success
                            else "late or superseded sdk acknowledgement"
                        )
                        episode = self._begin_stop_locked(reason)
                        needs_zero = True
            if needs_zero:
                assert episode is not None
                receipt = self._send_zero_locked("command rejected or expired", episode)
                with self._state_lock:
                    self._finish_stop_locked(episode, receipt)

        accepted = type(code) is int and code == 0 and not needs_zero
        completed = self._clock_ns()
        return CommandReceipt(
            command_id=setpoint.command_id,
            accepted=accepted,
            sdk_code=code if type(code) is int else None,
            reason="sdk accepted" if accepted else "command rejected and stop latched",
            submitted_monotonic_ns=submitted,
            completed_monotonic_ns=completed,
        )

    def stop(self, reason: str = "runtime stop") -> CommandReceipt:
        with self._state_lock:
            if not self._hardware_enabled:
                raise HardwareDisabledError("real Unitree commands are disabled by default")
            if self._closed or self._closing:
                raise UnitreeAdapterError("adapter is closed or closing")
            submitted = self._clock_ns()
            self._pending_stop_calls += 1
            episode = self._begin_stop_locked(reason, force_new=True)
        with self._rpc_lock:
            receipt = self._send_zero_locked(reason, episode, submitted=submitted)
            with self._state_lock:
                self._finish_stop_locked(episode, receipt)
                self._pending_stop_calls -= 1
        return receipt

    def reset_after_verified_stop(self, evidence: StoppedEvidence) -> None:
        # The RPC barrier prevents reset while a non-zero call or zero fallback
        # is still in flight. State is always acquired second to avoid inversion.
        with self._rpc_lock:
            with self._state_lock:
                if self._closed or self._closing:
                    raise UnitreeAdapterError("adapter is closed or closing")
                if not self._stop_latched:
                    raise CommandRejectedError("adapter is not stop-latched")
                if self._pending_stop_calls:
                    raise CommandRejectedError("a stop transport attempt is still pending")
                lease = self._retained_lease
                episode = self._stop_episode
                if lease is None or episode is None:
                    raise CommandRejectedError("no correlated stop episode is available")
                if (
                    episode.lease_id != lease.lease_id
                    or episode.lease_epoch != lease.epoch
                ):
                    raise CommandRejectedError("stop episode does not match retained lease")
                if (
                    episode.transport_quiesced_monotonic_ns is None
                    or not episode.zero_velocity_accepted
                ):
                    raise CommandRejectedError(
                        "zero-velocity transport has not quiesced successfully"
                    )
                now = self._clock_ns()
                self._validate_stopped_evidence(evidence, lease, episode, now)
                self._retained_lease = None
                self._stop_episode = None
                self._stop_latched = False
                self.last_fault = None

    def close(self) -> None:
        with self._state_lock:
            if self._closed or self._closing:
                return
            self._closing = True
            submitted = self._clock_ns()
            episode = (
                self._begin_stop_locked("adapter closed", force_new=True)
                if self._hardware_enabled
                else None
            )
        try:
            if episode is not None:
                with self._rpc_lock:
                    receipt = self._send_zero_locked(
                        "adapter closed", episode, submitted=submitted
                    )
                    with self._state_lock:
                        self._finish_stop_locked(episode, receipt)
        finally:
            with self._state_lock:
                self._cancel_watchdog_locked()
                self._closed = True
                self._closing = False

    def _validate_lease(self, lease: LocomotionLease, now: int) -> None:
        if not isinstance(lease, LocomotionLease):
            raise CommandRejectedError("lease must be a LocomotionLease")
        if not isinstance(lease.lease_id, str) or not lease.lease_id:
            raise CommandRejectedError("lease_id must be a non-empty string")
        if type(lease.epoch) is not int or lease.epoch < 0:
            raise CommandRejectedError("lease epoch must be a non-negative integer")
        if lease.actuator_scope != "whole_body_motion":
            raise CommandRejectedError(
                "G1 high-level locomotion conservatively owns whole body motion"
            )
        if type(lease.issued_monotonic_ns) is not int or type(
            lease.expires_monotonic_ns
        ) is not int:
            raise CommandRejectedError(
                "lease timestamps must be integer monotonic nanoseconds"
            )
        if lease.issued_monotonic_ns > now or lease.expires_monotonic_ns <= now:
            raise CommandRejectedError("lease is not currently valid")
        if lease.expires_monotonic_ns - lease.issued_monotonic_ns > int(
            self.max_lease_ttl_s * 1_000_000_000
        ):
            raise CommandRejectedError("lease TTL exceeds the adapter limit")

    def _validate_setpoint(self, setpoint: VelocitySetpoint, now: int) -> None:
        if self._closed or self._closing:
            raise CommandRejectedError("adapter is closed or closing")
        if not self._hardware_enabled:
            raise HardwareDisabledError("real Unitree commands are disabled by default")
        if self._stop_latched:
            raise CommandRejectedError("stop is latched")
        if not isinstance(setpoint, VelocitySetpoint):
            raise CommandRejectedError("setpoint must be a VelocitySetpoint")
        lease = self._active_lease
        if lease is None:
            raise CommandRejectedError(
                "no process-local locomotion ownership token is active"
            )
        if not isinstance(setpoint.lease_id, str) or not setpoint.lease_id:
            raise CommandRejectedError("setpoint lease_id must be a non-empty string")
        if type(setpoint.lease_epoch) is not int or setpoint.lease_epoch < 0:
            raise CommandRejectedError("setpoint lease epoch must be non-negative")
        if setpoint.lease_id != lease.lease_id or setpoint.lease_epoch != lease.epoch:
            raise CommandRejectedError("setpoint does not match the active lease epoch")
        if now >= lease.expires_monotonic_ns:
            raise CommandRejectedError("locomotion lease expired")
        if setpoint.frame_id != "base_link":
            raise CommandRejectedError("only base_link velocity commands are accepted")
        if not isinstance(setpoint.command_id, str) or not setpoint.command_id:
            raise CommandRejectedError("command_id must be a non-empty string")
        if type(setpoint.sequence) is not int or setpoint.sequence <= self._last_sequence:
            raise CommandRejectedError("setpoint sequence must increase monotonically")
        if type(setpoint.issued_monotonic_ns) is not int or type(
            setpoint.expires_monotonic_ns
        ) is not int:
            raise CommandRejectedError(
                "setpoint timestamps must be integer monotonic nanoseconds"
            )
        values = (setpoint.vx_mps, setpoint.vy_mps, setpoint.yaw_rate_radps)
        if not all(_finite_number(value) for value in values):
            raise CommandRejectedError("velocity values must be finite numbers")
        if setpoint.issued_monotonic_ns > now or setpoint.expires_monotonic_ns <= now:
            raise CommandRejectedError("setpoint is not currently valid")
        if setpoint.issued_monotonic_ns < lease.issued_monotonic_ns:
            raise CommandRejectedError("setpoint was issued before its locomotion lease")
        if setpoint.expires_monotonic_ns > lease.expires_monotonic_ns:
            raise CommandRejectedError("setpoint outlives its locomotion lease")
        if setpoint.expires_monotonic_ns - setpoint.issued_monotonic_ns > int(
            self.max_command_ttl_s * 1_000_000_000
        ):
            raise CommandRejectedError("setpoint TTL exceeds the adapter limit")
        if abs(setpoint.vx_mps) > self._limits.max_abs_vx_mps:
            raise CommandRejectedError("vx exceeds the configured target limit")
        if abs(setpoint.vy_mps) > self._limits.max_abs_vy_mps:
            raise CommandRejectedError("vy exceeds the configured target limit")
        if abs(setpoint.yaw_rate_radps) > self._limits.max_abs_yaw_rate_radps:
            raise CommandRejectedError("yaw rate exceeds the configured target limit")

    def _validate_stopped_evidence(
        self,
        evidence: StoppedEvidence,
        lease: LocomotionLease,
        episode: StopEpisode,
        now: int,
    ) -> None:
        if not isinstance(evidence, StoppedEvidence):
            raise CommandRejectedError("stopped evidence has the wrong type")
        if not isinstance(evidence.evidence_id, str) or not evidence.evidence_id:
            raise CommandRejectedError("stopped evidence ID is required")
        if (
            evidence.stop_request_id != episode.request_id
            or evidence.stop_generation != episode.generation
            or evidence.target_id != self._target_id
            or evidence.boot_id != self._boot_id
            or evidence.clock_domain != self._clock_domain
            or evidence.lease_id != lease.lease_id
            or evidence.lease_epoch != lease.epoch
        ):
            raise CommandRejectedError("stopped evidence correlation does not match")
        if type(evidence.window_start_monotonic_ns) is not int or type(
            evidence.window_end_monotonic_ns
        ) is not int:
            raise CommandRejectedError(
                "evidence window must use integer monotonic nanoseconds"
            )
        barrier = episode.transport_quiesced_monotonic_ns
        assert barrier is not None
        if (
            evidence.window_start_monotonic_ns < barrier
            or evidence.window_end_monotonic_ns < evidence.window_start_monotonic_ns
            or evidence.window_end_monotonic_ns > now
        ):
            raise CommandRejectedError("evidence window is not post-stop and monotonic")
        if evidence.window_end_monotonic_ns - evidence.window_start_monotonic_ns < int(
            self.min_evidence_window_s * 1_000_000_000
        ):
            raise CommandRejectedError("stopped evidence window is too short")
        if now - evidence.window_end_monotonic_ns > int(
            self.max_evidence_age_s * 1_000_000_000
        ):
            raise CommandRejectedError("stopped evidence is stale")
        speeds = (
            evidence.max_abs_linear_speed_mps,
            evidence.max_abs_yaw_rate_radps,
            evidence.max_abs_joint_velocity_radps,
            evidence.max_abs_roll_pitch_rate_radps,
        )
        if not all(_finite_number(value) and value >= 0.0 for value in speeds):
            raise CommandRejectedError("stopped evidence speeds are invalid")
        checks = (
            (
                evidence.max_abs_linear_speed_mps,
                self.stopped_linear_threshold_mps,
                "linear speed",
            ),
            (
                evidence.max_abs_yaw_rate_radps,
                self.stopped_yaw_threshold_radps,
                "yaw rate",
            ),
            (
                evidence.max_abs_joint_velocity_radps,
                self.stopped_joint_threshold_radps,
                "joint velocity",
            ),
            (
                evidence.max_abs_roll_pitch_rate_radps,
                self.stopped_roll_pitch_threshold_radps,
                "roll/pitch rate",
            ),
        )
        for measured, threshold, label in checks:
            if measured > threshold:
                raise CommandRejectedError(f"{label} does not prove a stop")

    def _begin_stop_locked(
        self, reason: str, *, force_new: bool = False
    ) -> StopEpisode:
        if force_new or not self._stop_latched or self._stop_episode is None:
            lease = self._active_lease or self._retained_lease
            lease_id = lease.lease_id if lease is not None else "no-active-lease"
            lease_epoch = lease.epoch if lease is not None else self._last_lease_epoch
            self._stop_generation += 1
            self._stop_episode = StopEpisode(
                request_id=f"unitree-stop-{self._stop_generation}",
                generation=self._stop_generation,
                lease_id=lease_id,
                lease_epoch=lease_epoch,
                requested_monotonic_ns=self._clock_ns(),
            )
        self._stop_latched = True
        if self._active_lease is not None:
            self._retained_lease = self._active_lease
            self._active_lease = None
        self._command_generation += 1
        self._cancel_watchdog_locked()
        self.last_fault = reason
        return self._stop_episode

    def _finish_stop_locked(
        self, episode: StopEpisode, receipt: CommandReceipt
    ) -> None:
        current = self._stop_episode
        if current is None or current.generation != episode.generation:
            return
        self._stop_episode = replace(
            current,
            transport_quiesced_monotonic_ns=receipt.completed_monotonic_ns,
            zero_velocity_accepted=receipt.accepted,
        )
        self.last_stop_receipt = receipt

    def _schedule_watchdog_locked(self, expires_ns: int, generation: int) -> None:
        self._cancel_watchdog_locked()
        delay_s = max(0.0, (expires_ns - self._clock_ns()) / 1_000_000_000)
        timer = threading.Timer(delay_s, self._watchdog_expired, args=(generation,))
        timer.daemon = True
        self._watchdog = timer
        timer.start()

    def _cancel_watchdog_locked(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    def _watchdog_expired(self, generation: int) -> None:
        with self._state_lock:
            if (
                generation != self._command_generation
                or self._stop_latched
                or self._closed
                or self._closing
            ):
                return
            self._watchdog = None
            episode = self._begin_stop_locked("velocity command watchdog expired")
            submitted = self._clock_ns()
        with self._rpc_lock:
            receipt = self._send_zero_locked(
                "watchdog expired", episode, submitted=submitted
            )
            with self._state_lock:
                self._finish_stop_locked(episode, receipt)

    def _send_zero_locked(
        self,
        reason: str,
        episode: StopEpisode,
        *,
        submitted: int | None = None,
    ) -> CommandReceipt:
        submitted = self._clock_ns() if submitted is None else submitted
        try:
            code = self._client.SetVelocity(
                0.0, 0.0, 0.0, self.stop_command_duration_s
            )
        except Exception as exc:
            completed = self._clock_ns()
            failure = f"{reason}: zero-velocity RPC raised {type(exc).__name__}"
            with self._state_lock:
                self.last_fault = failure
            return CommandReceipt(
                "stop",
                False,
                None,
                failure,
                submitted,
                completed,
                episode.request_id,
                episode.generation,
                episode.lease_epoch,
            )
        completed = self._clock_ns()
        accepted = type(code) is int and code == 0
        if not accepted:
            with self._state_lock:
                self.last_fault = (
                    f"{reason}: zero-velocity RPC rejected or returned an invalid code"
                )
        return CommandReceipt(
            "stop",
            accepted,
            code if type(code) is int else None,
            reason,
            submitted,
            completed,
            episode.request_id,
            episode.generation,
            episode.lease_epoch,
        )


def connect_unitree_g1(
    network_interface: str,
    *,
    target_id: str = "",
    boot_id: str = "",
    clock_domain: str = "host.monotonic",
    hardware_enabled: bool = False,
    limits: VelocityLimits | None = None,
    sdk_timeout_s: float = 0.05,
    max_command_ttl_s: float = 0.25,
) -> UnitreeG1HighLevelAdapter:
    """Create the official-SDK shim after an explicit supervised hardware gate."""

    if type(hardware_enabled) is not bool or hardware_enabled is not True:
        raise HardwareDisabledError(
            "pass hardware_enabled=True only in a supervised deployment"
        )
    if not isinstance(network_interface, str) or not network_interface.strip():
        raise ValueError("network_interface must be a non-empty string")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (target_id, boot_id, clock_domain)
    ):
        raise ValueError(
            "target_id, boot_id, and clock_domain are required for evidence correlation"
        )
    if limits is not None and type(limits) is not VelocityLimits:
        raise ValueError("limits must be a VelocityLimits instance")
    UnitreeG1HighLevelAdapter._require_range(
        "max_command_ttl_s", max_command_ttl_s, 0.05, 0.25
    )
    if (
        not _finite_number(sdk_timeout_s)
        or not 0.001 <= sdk_timeout_s <= max_command_ttl_s
    ):
        raise ValueError("SDK timeout hint must be positive and no longer than command TTL")
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    except ImportError as exc:
        raise UnitreeAdapterError(
            "Install Unitree's official unitree_sdk2_python from its reviewed revision"
        ) from exc

    ChannelFactoryInitialize(0, network_interface)
    client = LocoClient()
    client.SetTimeout(sdk_timeout_s)
    client.Init()
    return UnitreeG1HighLevelAdapter(
        client,
        target_id=target_id,
        boot_id=boot_id,
        clock_domain=clock_domain,
        hardware_enabled=True,
        limits=limits,
        max_command_ttl_s=max_command_ttl_s,
        transport_timeout_s=sdk_timeout_s,
    )
