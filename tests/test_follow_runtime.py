from __future__ import annotations

import unittest
from pathlib import Path

from longship.contracts.skills.follow_person import FollowScene, FollowState, PersonTrack
from longship.runtime.follow_person import FollowPersonRuntime
from longship.safety.follow_obstacle import ForwardObstacleGuard
from longship.skills.follow_person.config import FollowProfile
from longship.skills.follow_person.governor import MotionGovernor
from longship.skills.follow_person.planner import LocalFollowPlanner
from longship.targets.follow_person import RecordingMotion


ROOT = Path(__file__).resolve().parents[1]


def scene(
    now_ns: int,
    sequence: int,
    *,
    tracks: tuple[PersonTrack, ...] = (),
    healthy: bool = True,
    clearance: float | None = 10.0,
) -> FollowScene:
    return FollowScene(
        sequence=sequence,
        captured_monotonic_ns=now_ns,
        received_monotonic_ns=now_ns,
        healthy=healthy,
        calibration_id="synthetic-test-calibration",
        calibration_valid=healthy,
        detector_ready=healthy,
        floor_valid=healthy,
        tracks=tracks,
        obstacles=(),
        raw_forward_clearance_m=clearance,
        detail="test scene" if healthy else "synthetic camera dropout",
    )


class FollowRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        self.motion = RecordingMotion()
        self.runtime = FollowPersonRuntime(
            self.profile,
            self.motion,
            planner=LocalFollowPlanner(self.profile.planner, self.profile.control),
            safety_guard=ForwardObstacleGuard(self.profile.safety),
            governor=MotionGovernor(self.profile.control),
            session_id="follow-test",
        )
        self.now = 2_000_000_000
        self.runtime.start(now_ns=self.now)

    def advance(self, seconds: float) -> int:
        self.now += int(seconds * 1_000_000_000)
        return self.now

    def test_acquires_target_and_emits_ttl_bounded_motion(self) -> None:
        target = PersonTrack("person-a", 2.5, 0.0, 0.9)

        snapshot = self.runtime.tick(
            scene(self.advance(0.05), 1, tracks=(target,)), now_ns=self.now
        )

        self.assertEqual(snapshot.state, FollowState.FOLLOWING)
        self.assertEqual(snapshot.locked_track_id, "person-a")
        self.assertIsNotNone(snapshot.command)
        assert snapshot.command is not None
        self.assertGreater(snapshot.command.forward_mps, 0.0)
        self.assertLessEqual(
            snapshot.command.expires_monotonic_ns
            - snapshot.command.issued_monotonic_ns,
            250_000_000,
        )

    def test_transient_scene_failure_holds_zero_then_recovers_lock(self) -> None:
        target = PersonTrack("person-a", 2.4, 0.1, 0.9)
        self.runtime.tick(
            scene(self.advance(0.05), 1, tracks=(target,)), now_ns=self.now
        )

        held = self.runtime.tick(
            scene(
                self.advance(0.1),
                2,
                healthy=False,
                clearance=None,
            ),
            now_ns=self.now,
        )
        recovered = self.runtime.tick(
            scene(self.advance(0.1), 3, tracks=(target,)), now_ns=self.now
        )

        self.assertEqual(held.state, FollowState.HOLDING)
        assert held.command is not None
        self.assertTrue(held.command.is_zero)
        self.assertEqual(recovered.state, FollowState.FOLLOWING)
        self.assertEqual(recovered.locked_track_id, "person-a")

    def test_sustained_scene_failure_fails_closed(self) -> None:
        target = PersonTrack("person-a", 2.4, 0.0, 0.9)
        self.runtime.tick(
            scene(self.advance(0.05), 1, tracks=(target,)), now_ns=self.now
        )
        self.runtime.tick(
            scene(self.advance(0.05), 2, healthy=False, clearance=None),
            now_ns=self.now,
        )

        failed = None
        for sequence in range(3, 15):
            failed = self.runtime.tick(
                scene(
                    self.advance(0.1),
                    sequence,
                    healthy=False,
                    clearance=None,
                ),
                now_ns=self.now,
            )
            if failed.state is FollowState.FAILED:
                break

        assert failed is not None
        self.assertEqual(failed.state, FollowState.FAILED)
        self.assertTrue(failed.stop_verified)
        self.assertEqual(len(self.motion.stop_reasons), 1)

    def test_nearby_new_track_can_reacquire_last_seen_target(self) -> None:
        target = PersonTrack("person-a", 2.4, 0.2, 0.9)
        self.runtime.tick(
            scene(self.advance(0.05), 1, tracks=(target,)), now_ns=self.now
        )
        lost = self.runtime.tick(
            scene(self.advance(0.05), 2), now_ns=self.now
        )
        replacement = PersonTrack("person-b", 2.35, 0.22, 0.9)
        recovered = self.runtime.tick(
            scene(self.advance(0.05), 3, tracks=(replacement,)), now_ns=self.now
        )

        self.assertEqual(lost.state, FollowState.LOST_APPROACH)
        self.assertEqual(recovered.state, FollowState.FOLLOWING)
        self.assertEqual(recovered.locked_track_id, "person-b")

    def test_persistent_raw_obstacle_requires_supervisor_restart(self) -> None:
        target = PersonTrack("person-a", 2.5, 0.0, 0.9)
        blocked = self.runtime.tick(
            scene(self.advance(0.05), 1, tracks=(target,), clearance=0.4),
            now_ns=self.now,
        )
        failed = None
        for sequence in range(2, 35):
            failed = self.runtime.tick(
                scene(
                    self.advance(0.1),
                    sequence,
                    tracks=(target,),
                    clearance=0.4,
                ),
                now_ns=self.now,
            )
            if failed.state is FollowState.FAILED:
                break

        assert failed is not None
        self.assertEqual(blocked.state, FollowState.BLOCKED)
        assert blocked.command is not None
        self.assertTrue(blocked.command.is_zero)
        self.assertEqual(failed.state, FollowState.FAILED)

    def test_observability_failure_does_not_interrupt_control(self) -> None:
        class BrokenSink:
            def publish(self, event) -> None:
                del event
                raise OSError("synthetic disk failure")

        runtime = FollowPersonRuntime(
            self.profile,
            RecordingMotion(),
            planner=LocalFollowPlanner(self.profile.planner, self.profile.control),
            safety_guard=ForwardObstacleGuard(self.profile.safety),
            governor=MotionGovernor(self.profile.control),
            event_sink=BrokenSink(),
            session_id="follow-with-broken-observer",
        )
        runtime.start(now_ns=self.now)

        snapshot = runtime.tick(
            scene(
                self.advance(0.05),
                1,
                tracks=(PersonTrack("person-a", 2.5, 0.0, 0.9),),
            ),
            now_ns=self.now,
        )

        self.assertEqual(snapshot.state, FollowState.FOLLOWING)
        self.assertEqual(runtime.last_observability_error, "OSError")

    def test_qualification_bound_calibration_cannot_change_mid_session(self) -> None:
        motion = RecordingMotion()
        runtime = FollowPersonRuntime(
            self.profile,
            motion,
            planner=LocalFollowPlanner(self.profile.planner, self.profile.control),
            safety_guard=ForwardObstacleGuard(self.profile.safety),
            governor=MotionGovernor(self.profile.control),
            session_id="calibration-bound-follow",
            required_calibration_id="reviewed-camera-a",
        )
        runtime.start(now_ns=self.now)
        mismatched = scene(
            self.advance(0.05),
            1,
            tracks=(PersonTrack("person-a", 2.5, 0.0, 0.9),),
        )

        snapshot = runtime.tick(mismatched, now_ns=self.now)

        self.assertEqual(snapshot.state, FollowState.HOLDING)
        self.assertIn("calibration identity changed", snapshot.detail)

    def test_resume_keeps_zero_until_locked_target_is_freshly_seen(self) -> None:
        target = PersonTrack("person-a", 2.5, 0.0, 0.9)
        self.runtime.tick(
            scene(self.advance(0.05), 1, tracks=(target,)), now_ns=self.now
        )
        self.runtime.pause(now_ns=self.advance(0.05))
        self.runtime.resume(now_ns=self.advance(0.05))

        waiting = self.runtime.tick(
            scene(self.advance(0.05), 2), now_ns=self.now
        )
        resumed = self.runtime.tick(
            scene(self.advance(0.05), 3, tracks=(target,)), now_ns=self.now
        )

        self.assertEqual(waiting.state, FollowState.ACQUIRING)
        assert waiting.command is not None
        self.assertTrue(waiting.command.is_zero)
        self.assertEqual(resumed.state, FollowState.FOLLOWING)

    def test_control_loop_gap_beyond_command_ttl_fails_closed(self) -> None:
        target = PersonTrack("person-a", 2.5, 0.0, 0.9)

        snapshot = self.runtime.tick(
            scene(self.advance(0.2), 1, tracks=(target,)), now_ns=self.now
        )

        self.assertEqual(snapshot.state, FollowState.FAILED)
        self.assertIn("control loop exceeded", snapshot.detail)

    def test_planner_exception_enters_owned_stop_path(self) -> None:
        class BrokenPlanner:
            def plan(self, *args, **kwargs):
                del args, kwargs
                raise RuntimeError("synthetic planner fault")

        motion = RecordingMotion()
        runtime = FollowPersonRuntime(
            self.profile,
            motion,
            planner=BrokenPlanner(),
            safety_guard=ForwardObstacleGuard(self.profile.safety),
            governor=MotionGovernor(self.profile.control),
            session_id="broken-planner-follow",
        )
        runtime.start(now_ns=self.now)

        snapshot = runtime.tick(
            scene(
                self.advance(0.05),
                1,
                tracks=(PersonTrack("person-a", 2.5, 0.0, 0.9),),
            ),
            now_ns=self.now,
        )

        self.assertEqual(snapshot.state, FollowState.FAILED)
        self.assertEqual(len(motion.stop_reasons), 1)


if __name__ == "__main__":
    unittest.main()
