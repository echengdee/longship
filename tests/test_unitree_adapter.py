from __future__ import annotations

import math
import threading
import unittest

from longship.contracts.skills.follow_person import FollowCommand
from longship.targets.follow_person import UnitreeFollowMotion
from longship.targets.unitree_sdk2 import (
    CommandRejectedError,
    HardwareDisabledError,
    LocomotionLease,
    StoppedEvidence,
    UnitreeAdapterError,
    UnitreeG1HighLevelAdapter,
    VelocityLimits,
    VelocitySetpoint,
    connect_unitree_g1,
)


class FakeClock:
    def __init__(self, now_ns: int = 5_000_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, delta_ns: int) -> None:
        self.now_ns += delta_ns


class FakeLocoClient:
    def __init__(self, code: int = 0, *, clock=None, command_delay_ns: int = 0) -> None:
        self.code = code
        self.clock = clock
        self.command_delay_ns = command_delay_ns
        self.calls: list[tuple[float, float, float, float]] = []
        self.zero_received = threading.Event()

    def SetVelocity(self, vx, vy, omega, duration=1.0):
        self.calls.append((vx, vy, omega, duration))
        if vx == vy == omega == 0.0:
            self.zero_received.set()
        elif self.clock is not None and self.command_delay_ns:
            self.clock.advance(self.command_delay_ns)
        return self.code


class BlockingLocoClient(FakeLocoClient):
    def __init__(self) -> None:
        super().__init__()
        self.nonzero_started = threading.Event()
        self.release_nonzero = threading.Event()

    def SetVelocity(self, vx, vy, omega, duration=1.0):
        self.calls.append((vx, vy, omega, duration))
        if vx == vy == omega == 0.0:
            self.zero_received.set()
            return self.code
        self.nonzero_started.set()
        self.release_nonzero.wait(timeout=1.0)
        return self.code


class SequencedZeroClient(FakeLocoClient):
    def __init__(self, zero_codes) -> None:
        super().__init__()
        self.zero_codes = iter(zero_codes)

    def SetVelocity(self, vx, vy, omega, duration=1.0):
        self.calls.append((vx, vy, omega, duration))
        if vx == vy == omega == 0.0:
            code = next(self.zero_codes)
            if code == 0:
                self.zero_received.set()
            return code
        return 0


def lease(clock: FakeClock, *, epoch: int = 1) -> LocomotionLease:
    return LocomotionLease(
        lease_id=f"lease-{epoch}",
        epoch=epoch,
        issued_monotonic_ns=clock(),
        expires_monotonic_ns=clock() + 1_000_000_000,
    )


def setpoint(clock: FakeClock, **overrides) -> VelocitySetpoint:
    values = {
        "command_id": "cmd-1",
        "lease_id": "lease-1",
        "lease_epoch": 1,
        "sequence": 1,
        "issued_monotonic_ns": clock(),
        "expires_monotonic_ns": clock() + 40_000_000,
        "vx_mps": 0.1,
        "vy_mps": 0.0,
        "yaw_rate_radps": 0.0,
        "frame_id": "base_link",
    }
    values.update(overrides)
    return VelocitySetpoint(**values)


def adapter(client=None, *, enabled=True, clock=None) -> UnitreeG1HighLevelAdapter:
    clock = clock or FakeClock()
    return UnitreeG1HighLevelAdapter(
        client or FakeLocoClient(),
        target_id="g1-test",
        boot_id="boot-test",
        clock_domain="test.monotonic",
        hardware_enabled=enabled,
        limits=VelocityLimits(0.2, 0.1, 0.3),
        max_command_ttl_s=0.05,
        transport_timeout_s=0.05,
        stop_command_duration_s=0.05,
        clock_ns=clock,
    )


def stopped_evidence(
    target: UnitreeG1HighLevelAdapter,
    clock: FakeClock,
    *,
    evidence_id: str = "evidence-1",
    lease_id: str = "lease-1",
    linear_speed: float = 0.0,
    window_start_ns: int | None = None,
) -> StoppedEvidence:
    episode = target.current_stop_episode
    assert episode is not None
    barrier = episode.transport_quiesced_monotonic_ns
    assert barrier is not None
    start = barrier if window_start_ns is None else window_start_ns
    end = start + 100_000_000
    return StoppedEvidence(
        evidence_id=evidence_id,
        stop_request_id=episode.request_id,
        stop_generation=episode.generation,
        target_id="g1-test",
        boot_id="boot-test",
        clock_domain="test.monotonic",
        lease_id=lease_id,
        lease_epoch=1,
        window_start_monotonic_ns=start,
        window_end_monotonic_ns=end,
        max_abs_linear_speed_mps=linear_speed,
        max_abs_yaw_rate_radps=0.0,
        max_abs_joint_velocity_radps=0.0,
        max_abs_roll_pitch_rate_radps=0.0,
    )


class UnitreeAdapterTests(unittest.TestCase):
    def test_hardware_is_disabled_before_sdk_import(self) -> None:
        with self.assertRaises(HardwareDisabledError):
            connect_unitree_g1("eth0")
        for invalid_gate in (1, "false", object()):
            with self.subTest(invalid_gate=invalid_gate):
                with self.assertRaises(HardwareDisabledError):
                    connect_unitree_g1(
                        "eth0", hardware_enabled=invalid_gate  # type: ignore[arg-type]
                    )
        invalid_enabled_configs = (
            {"network_interface": " "},
            {"target_id": " "},
            {"max_command_ttl_s": math.nan},
            {"limits": object()},
        )
        for override in invalid_enabled_configs:
            values = {
                "network_interface": "eth0",
                "target_id": "g1-test",
                "boot_id": "boot-test",
                "hardware_enabled": True,
            }
            values.update(override)
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    connect_unitree_g1(**values)  # type: ignore[arg-type]

    def test_disabled_adapter_stop_never_calls_sdk(self) -> None:
        client = FakeLocoClient()
        target = adapter(client, enabled=False)

        with self.assertRaises(HardwareDisabledError):
            target.stop("disabled")

        self.assertEqual(client.calls, [])

    def test_command_requires_scoped_lease_and_valid_setpoint(self) -> None:
        invalid_overrides = (
            {"lease_id": "other"},
            {"lease_epoch": 2},
            {"frame_id": "map"},
            {"vx_mps": math.nan},
            {"vx_mps": 0.21},
            {"expires_monotonic_ns": 5_000_000_000},
            {"expires_monotonic_ns": 5_060_000_000},
            {"issued_monotonic_ns": 4_999_999_999},
            {"lease_epoch": True},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                clock = FakeClock()
                target = adapter(clock=clock)
                target.arm(lease(clock))
                with self.assertRaises(CommandRejectedError):
                    target.command(setpoint(clock, **overrides))
                target.close()

    def test_active_lease_can_only_be_refreshed_by_same_owner(self) -> None:
        clock = FakeClock()
        target = adapter(clock=clock)
        target.arm(lease(clock))
        clock.advance(100_000_000)
        refreshed = LocomotionLease(
            lease_id="lease-1",
            epoch=1,
            issued_monotonic_ns=clock(),
            expires_monotonic_ns=clock() + 1_000_000_000,
        )

        target.refresh_lease(refreshed)
        receipt = target.command(setpoint(clock))

        self.assertTrue(receipt.accepted)
        clock.advance(100_000_000)
        with self.assertRaises(CommandRejectedError):
            target.refresh_lease(
                LocomotionLease(
                    lease_id="different-owner",
                    epoch=1,
                    issued_monotonic_ns=clock(),
                    expires_monotonic_ns=clock() + 1_000_000_000,
                )
            )
        clock.advance(1_000_000_000)
        with self.assertRaises(CommandRejectedError):
            target.refresh_lease(
                LocomotionLease(
                    lease_id="lease-1",
                    epoch=1,
                    issued_monotonic_ns=clock(),
                    expires_monotonic_ns=clock() + 1_000_000_000,
                )
            )
        target.close()

    def test_follow_motion_maps_commands_and_refreshes_same_lease(self) -> None:
        clock = FakeClock()
        client = FakeLocoClient()
        target = adapter(client, clock=clock)
        motion = UnitreeFollowMotion(target, lease_ttl_s=1.0, clock_ns=clock)
        acquired = motion.acquire("follow-session", clock())
        clock.advance(800_000_000)
        command = FollowCommand(
            session_id="follow-session",
            sequence=1,
            issued_monotonic_ns=clock(),
            expires_monotonic_ns=clock() + 50_000_000,
            forward_mps=0.1,
            yaw_rate_radps=0.2,
            reason="test follow command",
        )

        receipt = motion.apply(command)

        self.assertTrue(acquired.accepted)
        self.assertTrue(receipt.accepted)
        self.assertEqual(client.calls[0][:3], (0.1, 0.0, 0.2))
        stopped = motion.protective_stop("test complete")
        self.assertFalse(stopped.verified_stopped)
        motion.close()

    def test_short_deadline_watchdog_latches_and_sends_zero(self) -> None:
        clock = FakeClock()
        client = FakeLocoClient()
        target = adapter(client, clock=clock)
        target.arm(lease(clock))

        receipt = target.command(setpoint(clock))

        self.assertTrue(receipt.accepted)
        self.assertFalse(receipt.executed_or_stopped)
        self.assertLessEqual(client.calls[0][3], 0.05)
        self.assertTrue(client.zero_received.wait(timeout=0.2))
        self.assertTrue(target.stop_latched)
        self.assertEqual(target.retained_lease_id, "lease-1")
        target.close()

    def test_slow_sdk_response_is_rejected_and_zeroed(self) -> None:
        clock = FakeClock()
        client = FakeLocoClient(clock=clock, command_delay_ns=80_000_000)
        target = adapter(client, clock=clock)
        target.arm(lease(clock))

        receipt = target.command(setpoint(clock))

        self.assertFalse(receipt.accepted)
        self.assertTrue(target.stop_latched)
        self.assertEqual(client.calls[-1][:3], (0.0, 0.0, 0.0))
        self.assertIsNotNone(target.last_stop_receipt)

    def test_nonzero_sdk_code_is_observable_and_latched(self) -> None:
        clock = FakeClock()
        client = FakeLocoClient(code=7)
        target = adapter(client, clock=clock)
        target.arm(lease(clock))

        receipt = target.command(setpoint(clock))

        self.assertFalse(receipt.accepted)
        self.assertEqual(receipt.sdk_code, 7)
        self.assertTrue(target.stop_latched)
        self.assertIsNotNone(target.last_stop_receipt)
        self.assertFalse(target.last_stop_receipt.accepted)

    def test_stop_revokes_active_epoch_and_prevents_resurrection(self) -> None:
        clock = FakeClock()
        target = adapter(clock=clock)
        target.arm(lease(clock))

        receipt = target.stop("test")

        self.assertTrue(receipt.accepted)
        self.assertTrue(target.stop_latched)
        self.assertIsNone(target.armed_lease_id)
        self.assertEqual(target.retained_lease_id, "lease-1")
        with self.assertRaises(CommandRejectedError):
            target.command(setpoint(clock))

    def test_reset_requires_fresh_correlated_stopped_evidence(self) -> None:
        clock = FakeClock()
        target = adapter(clock=clock)
        target.arm(lease(clock))
        target.stop("test")

        with self.assertRaises(CommandRejectedError):
            target.reset_after_verified_stop(stopped_evidence(target, clock, lease_id="wrong"))
        with self.assertRaises(CommandRejectedError):
            target.reset_after_verified_stop(
                stopped_evidence(target, clock, linear_speed=0.1)
            )
        with self.assertRaises(CommandRejectedError):
            target.reset_after_verified_stop(
                stopped_evidence(target, clock, window_start_ns=clock() - 1)
            )

        clock.advance(100_000_000)
        target.reset_after_verified_stop(stopped_evidence(target, clock))
        self.assertFalse(target.stop_latched)
        self.assertIsNone(target.retained_lease_id)
        target.arm(lease(clock, epoch=2))
        self.assertEqual(target.armed_lease_id, "lease-2")
        target.close()

    def test_close_latches_and_retains_lease_identity(self) -> None:
        clock = FakeClock()
        target = adapter(clock=clock)
        target.arm(lease(clock))

        target.close()

        self.assertTrue(target.stop_latched)
        self.assertEqual(target.retained_lease_id, "lease-1")

    def test_stop_thresholds_reject_nan_and_infinite_configuration(self) -> None:
        for field, value in (
            ("stopped_linear_threshold_mps", math.nan),
            ("stopped_yaw_threshold_radps", math.inf),
            ("stopped_joint_threshold_radps", math.nan),
            ("stopped_roll_pitch_threshold_radps", math.inf),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    UnitreeG1HighLevelAdapter(
                        FakeLocoClient(),
                        target_id="g1-test",
                        boot_id="boot-test",
                        **{field: value},
                    )

    def test_limits_require_the_validated_value_type(self) -> None:
        fake_limits = type(
            "NotVelocityLimits",
            (),
            {
                "max_abs_vx_mps": math.inf,
                "max_abs_vy_mps": math.inf,
                "max_abs_yaw_rate_radps": math.inf,
            },
        )()
        with self.assertRaises(ValueError):
            UnitreeG1HighLevelAdapter(
                FakeLocoClient(),
                target_id="g1-test",
                boot_id="boot-test",
                limits=fake_limits,  # type: ignore[arg-type]
            )

    def test_boolean_sdk_code_is_a_protocol_failure_and_cannot_clear_latch(self) -> None:
        clock = FakeClock()
        target = adapter(FakeLocoClient(code=False), clock=clock)
        target.arm(lease(clock))

        receipt = target.command(setpoint(clock))

        self.assertFalse(receipt.accepted)
        self.assertIsNone(receipt.sdk_code)
        self.assertTrue(target.stop_latched)
        self.assertIsNotNone(target.current_stop_episode)
        clock.advance(100_000_000)
        with self.assertRaises(CommandRejectedError):
            target.reset_after_verified_stop(stopped_evidence(target, clock))

    def test_close_prevents_latch_reset(self) -> None:
        clock = FakeClock()
        target = adapter(clock=clock)
        target.arm(lease(clock))
        target.close()
        clock.advance(100_000_000)

        with self.assertRaises(UnitreeAdapterError):
            target.reset_after_verified_stop(stopped_evidence(target, clock))

    def test_reset_cannot_pass_an_inflight_nonzero_rpc_or_pre_stop_sample(self) -> None:
        clock = FakeClock()
        client = BlockingLocoClient()
        target = adapter(client, clock=clock)
        target.arm(lease(clock))
        command_thread = threading.Thread(target=target.command, args=(setpoint(clock),))
        command_thread.start()
        self.assertTrue(client.nonzero_started.wait(timeout=0.2))

        stop_thread = threading.Thread(target=target.stop, args=("concurrent stop",))
        stop_thread.start()
        for _ in range(100):
            if target.stop_latched:
                break
            threading.Event().wait(0.001)
        episode = target.current_stop_episode
        self.assertIsNotNone(episode)
        assert episode is not None
        stale = StoppedEvidence(
            "pre-stop",
            episode.request_id,
            episode.generation,
            "g1-test",
            "boot-test",
            "test.monotonic",
            "lease-1",
            1,
            episode.requested_monotonic_ns - 100_000_000,
            episode.requested_monotonic_ns,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        reset_errors: list[Exception] = []

        def reset() -> None:
            try:
                target.reset_after_verified_stop(stale)
            except Exception as exc:
                reset_errors.append(exc)

        reset_thread = threading.Thread(target=reset)
        reset_thread.start()
        threading.Event().wait(0.02)
        self.assertTrue(reset_thread.is_alive())
        self.assertFalse(client.zero_received.is_set())
        self.assertTrue(target.stop_latched)

        client.release_nonzero.set()
        command_thread.join(timeout=0.5)
        stop_thread.join(timeout=0.5)
        reset_thread.join(timeout=0.5)

        self.assertTrue(client.zero_received.is_set())
        self.assertTrue(target.stop_latched)
        self.assertTrue(reset_errors)

    def test_new_stop_invalidates_old_evidence_before_waiting_for_rpc(self) -> None:
        clock = FakeClock()
        target = adapter(clock=clock)
        target.arm(lease(clock))
        target.stop("first")
        first_episode = target.current_stop_episode
        assert first_episode is not None
        clock.advance(100_000_000)
        old_evidence = stopped_evidence(target, clock)

        target._rpc_lock.acquire()
        stop_thread = threading.Thread(target=target.stop, args=("second",))
        stop_thread.start()
        for _ in range(100):
            current = target.current_stop_episode
            if current is not None and current.generation > first_episode.generation:
                break
            threading.Event().wait(0.001)
        reset_errors: list[Exception] = []

        def reset() -> None:
            try:
                target.reset_after_verified_stop(old_evidence)
            except Exception as exc:
                reset_errors.append(exc)

        reset_thread = threading.Thread(target=reset)
        reset_thread.start()
        threading.Event().wait(0.02)
        self.assertTrue(reset_thread.is_alive())
        target._rpc_lock.release()
        stop_thread.join(timeout=0.5)
        reset_thread.join(timeout=0.5)

        self.assertTrue(reset_errors)
        self.assertTrue(target.stop_latched)
        current = target.current_stop_episode
        self.assertIsNotNone(current)
        assert current is not None
        self.assertGreater(current.generation, first_episode.generation)

    def test_latest_repeated_stop_failure_cannot_be_overwritten(self) -> None:
        clock = FakeClock()
        client = SequencedZeroClient([0, 7])
        target = adapter(client, clock=clock)
        target.arm(lease(clock))

        first = target.stop("first")
        second = target.stop("second")

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertIsNotNone(target.current_stop_episode)
        assert target.current_stop_episode is not None
        self.assertFalse(target.current_stop_episode.zero_velocity_accepted)
        self.assertIs(target.last_stop_receipt, second)


if __name__ == "__main__":
    unittest.main()
