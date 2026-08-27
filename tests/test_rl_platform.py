from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from longship.rl.builder import build_model
from longship.rl.compatibility import CompatibilityLock, CompatibilityLockError
from longship.rl.config import ExperimentConfig, ExperimentConfigError
from longship.rl.registry import ComponentRegistry, RegistryError
from longship.rl.sim2sim.dds import (
    G1_29DOF_JOINTS,
    SECONDARY_IMU_TOPIC,
    DdsContract,
    sdk_pythonpath,
)
from longship.rl.sim2sim.launch import backend_launch
from longship.rl.sim2sim.preflight import inspect_artifact
from longship.rl.sim2sim.control import ControlMode, PolicyControl
from longship.rl.sim2sim.adapters.holosoma_dds import _advance_phase
from longship.rl.sim2sim.profile import bundled_profile_path, load_control_profile
from longship.rl.sim2sim.hiking_pipeline import (
    POLICY_SIGNS,
    HikingModeCommand,
    dds_to_policy,
    policy_to_dds,
)
from longship.rl.sim2sim.teleop import CAPABILITIES, TeleopCommand
from longship.rl.sim2sim.simulator import (
    InteractiveControls,
    LowCommandSnapshot,
    _draw_virtual_gantry,
    _update_tracking_camera,
)
from longship.rl.sim2sim.sonic_pipeline import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    SonicMotion,
    SonicOnnxPipeline,
    SonicPlannerCommand,
    allowed_pred_num_tokens,
)
from longship.rl.training.runner import ExperimentRunner


@dataclass
class Part:
    values: dict[str, Any]

    def __init__(self, **values: Any) -> None:
        self.values = values


@dataclass
class Policy:
    encoder: Part
    backbone: Part
    actor_decoder: Part
    critic_decoder: Part


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[Mapping[str, Any], Path]] = []

    def train(self, experiment: Mapping[str, Any], output_dir: Path) -> Path:
        self.calls.append((experiment, output_dir))
        return output_dir / "checkpoints" / "last.pt"


def model_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register("encoder", "ProprioceptionEncoder")(Part)
    registry.register("backbone", "MLPBackbone")(Part)
    registry.register("decoder", "GaussianActorDecoder")(Part)
    registry.register("decoder", "ValueDecoder")(Part)
    registry.register("policy", "ActorCriticPolicy")(Policy)
    return registry


def experiment_values() -> dict[str, Any]:
    return {
        "schema_version": "longship.rl-experiment.v1",
        "name": "g1_locomotion_ppo",
        "seed": 42,
        "model": {
            "type": "ActorCriticPolicy",
            "encoder": {
                "type": "ProprioceptionEncoder",
                "history_steps": 5,
            },
            "backbone": {
                "type": "MLPBackbone",
                "hidden_dims": [512, 256, 128],
            },
            "actor_decoder": {
                "type": "GaussianActorDecoder",
                "action_dim": 29,
            },
            "critic_decoder": {"type": "ValueDecoder", "output_dim": 1},
        },
        "training": {
            "backend": {"type": "HoloSomaBackend"},
            "trainer": {"type": "PPO"},
        },
        "environment": {"robot": "unitree_g1_29dof", "task": "velocity_tracking"},
    }


class RLPlatformTests(unittest.TestCase):
    def test_dds_contract_is_unitree_g1_compatible(self) -> None:
        values = CompatibilityLock.bundled().values["contracts"]["sim2sim_transport"]
        contract = DdsContract.from_mapping(values)
        contract.validate()
        self.assertEqual(contract.interface, "lo")
        self.assertEqual(contract.lowstate_topic, "rt/lowstate")
        self.assertEqual(contract.lowcmd_topic, "rt/lowcmd")
        self.assertEqual(contract.secondary_imu_topic, SECONDARY_IMU_TOPIC)
        self.assertEqual(len(G1_29DOF_JOINTS), 29)
        self.assertEqual(len(set(G1_29DOF_JOINTS)), 29)

    def test_sim2sim_rejects_physical_network_interface(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            DdsContract(interface="eth0").validate()

    def test_all_backends_use_the_same_dds_topics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launches = [
                backend_launch(temporary, backend)
                for backend in ("holosoma", "sonic", "instinctlab", "php")
            ]
        self.assertEqual({item.contract.lowstate_topic for item in launches}, {"rt/lowstate"})
        self.assertEqual({item.contract.lowcmd_topic for item in launches}, {"rt/lowcmd"})
        self.assertEqual({item.contract.interface for item in launches}, {"lo"})
        self.assertEqual(
            {item.contract.secondary_imu_topic for item in launches}, {"rt/secondary_imu"}
        )
        self.assertEqual(launches[-1].contract.depth_topic, "rt/camera/depth")

    def test_each_backend_has_an_owned_control_profile(self) -> None:
        profiles = {
            backend: load_control_profile(bundled_profile_path(backend), backend)
            for backend in ("holosoma", "sonic", "instinctlab", "php")
        }
        self.assertEqual(profiles["holosoma"].initialization.source, "profile")
        self.assertEqual(profiles["sonic"].initialization.source, "python_pipeline")
        self.assertEqual(profiles["sonic"].policy_options["runtime"], "onnxruntime")
        self.assertEqual(profiles["instinctlab"].initialization.source, "checkpoint")
        self.assertEqual(profiles["instinctlab"].dds.depth_topic, "rt/camera/depth")
        self.assertEqual(profiles["php"].policy.source, "onnx_metadata")
        self.assertEqual(profiles["php"].dds.depth_topic, "rt/camera/depth")

    def test_all_backends_use_the_longship_mujoco_simulator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launches = [
                backend_launch(temporary, backend)
                for backend in ("holosoma", "sonic", "instinctlab", "php")
            ]
        for launch in launches:
            self.assertIn("longship.rl.sim2sim.simulator", launch.simulator.argv)
            self.assertIn("--viewer", launch.simulator.argv)
            self.assertIn("--gantry", launch.simulator.argv)
            self.assertIn("--profile", launch.controller.argv)
            state_rate = launch.simulator.argv.index("--state-frequency-hz") + 1
            self.assertEqual(launch.simulator.argv[state_rate], str(launch.contract.state_frequency_hz))
        self.assertNotIn("--depth", launches[0].simulator.argv)
        self.assertNotIn("--depth", launches[1].simulator.argv)
        self.assertIn("--depth", launches[2].simulator.argv)

    def test_every_backend_has_a_zmq_keyboard_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launches = [backend_launch(temporary, name) for name in CAPABILITIES]
        for launch in launches:
            self.assertIn("longship.rl.sim2sim.teleop", launch.teleop.argv)
            self.assertIn("--teleop-endpoint", launch.controller.argv)

    def test_teleop_capabilities_do_not_invent_hiking_commands(self) -> None:
        hiking = CAPABILITIES["instinctlab"].keys
        self.assertTrue(set("i12npwqe]").issubset(hiking))
        self.assertTrue(set("sad").isdisjoint(hiking))
        self.assertTrue(set("wasdqe").issubset(CAPABILITIES["holosoma"].keys))
        self.assertIn("=", CAPABILITIES["holosoma"].keys)
        self.assertTrue(set("wasdqe").issubset(CAPABILITIES["sonic"].keys))
        self.assertTrue(set("wasdqey").issubset(CAPABILITIES["php"].keys))

    def test_hiking_exposes_only_upstream_stand_and_parkour_agents(self) -> None:
        command = HikingModeCommand()
        self.assertEqual(command.mode, "stand")
        self.assertIn("parkour", command.handle("2"))
        self.assertEqual(command.mode, "parkour")
        self.assertIn("stand", command.handle("n"))
        self.assertEqual(command.mode, "stand")
        self.assertIn("unsupported", command.handle("3"))

    def test_hiking_preserves_source_scene_joint_signs(self) -> None:
        policy = np.arange(29, dtype=np.float64) + 1.0
        dds = policy_to_dds(policy)
        self.assertTrue(np.array_equal(dds_to_policy(dds), policy))
        self.assertTrue(np.array_equal(np.flatnonzero(POLICY_SIGNS < 0), (2, 5, 8)))
        gains = policy_to_dds(np.ones(29), signed=False)
        self.assertTrue(np.array_equal(gains, np.ones(29)))

    def test_teleop_message_roundtrip(self) -> None:
        message = TeleopCommand("w", 7, 123.5)
        self.assertEqual(TeleopCommand.decode(message.encode()), message)

    def test_python_sonic_pipeline_preserves_native_joint_mappings(self) -> None:
        self.assertTrue(
            np.array_equal(
                ISAACLAB_TO_MUJOCO[MUJOCO_TO_ISAACLAB],
                np.arange(29),
            )
        )

    def test_python_sonic_planner_keeps_its_own_command_semantics(self) -> None:
        command = SonicPlannerCommand()
        self.assertIn("SLOW_WALK", command.handle("1"))
        self.assertIn("movement", command.handle("w"))
        self.assertTrue(np.allclose(command.movement, (1.0, 0.0, 0.0)))
        self.assertEqual(command.target_speed, 0.4)
        self.assertIn("facing", command.handle("q"))
        self.assertTrue(np.allclose(command.movement, 0.0))
        self.assertEqual(command.target_speed, 0.0)
        self.assertGreater(command.facing[1], 0.0)

    def test_python_sonic_uses_native_prediction_token_masks(self) -> None:
        self.assertTrue(np.array_equal(allowed_pred_num_tokens(1)[0], [1] * 6 + [0] * 5))
        self.assertTrue(np.array_equal(allowed_pred_num_tokens(10)[0], [1] * 11))

    def test_python_sonic_exposes_native_mode_sets(self) -> None:
        command = SonicPlannerCommand()
        self.assertIn("RUN", command.handle("3"))
        self.assertEqual(command.mode, 3)
        self.assertIn("squat/crawl", command.handle("n"))
        self.assertEqual(command.mode, 4)
        self.assertEqual(command.target_height, 0.8)
        self.assertIn("unsupported", command.handle("w"))
        self.assertIn("CRAWLING", command.handle("4"))
        self.assertEqual(command.mode, 8)
        self.assertIn("movement", command.handle("w"))
        self.assertEqual(command.target_speed, 0.4)
        self.assertIn("boxing", command.handle("n"))
        self.assertIn("LEFT_HOOK", command.handle("6"))
        self.assertIn("styled walk", command.handle("n"))
        self.assertIn("SCARE_WALK", command.handle("7"))
        self.assertIn("standing", command.handle("n"))
        self.assertEqual(command.mode, 1)

    def test_python_sonic_history_matches_native_startup_padding(self) -> None:
        pipeline = SonicOnnxPipeline.__new__(SonicOnnxPipeline)
        pipeline.history = [
            (np.ones(3), np.ones(29), np.ones(29), np.ones(29), -np.ones(3))
        ]
        gravity = pipeline._history_array(4).reshape(10, 3)
        self.assertTrue(np.allclose(gravity[:9], (0.0, 0.0, 1.0)))
        self.assertTrue(np.allclose(gravity[-1], (-1.0, -1.0, -1.0)))

    def test_python_sonic_rolling_planner_uses_eight_frame_splice(self) -> None:
        old = SonicMotion(
            np.zeros((3, 29)),
            np.zeros((3, 29)),
            np.zeros((3, 3)),
            np.tile((1.0, 0.0, 0.0, 0.0), (3, 1)),
        )
        new = SonicMotion(
            np.ones((10, 29)),
            np.ones((10, 29)),
            np.ones((10, 3)),
            np.tile((1.0, 0.0, 0.0, 0.0), (10, 1)),
        )
        merged = SonicOnnxPipeline._merge_motion(old, 1, 3, new)
        self.assertEqual(merged.frames, 12)
        self.assertTrue(np.allclose(merged.joint_positions[2], 0.0))
        self.assertTrue(np.allclose(merged.joint_positions[6], 0.5))
        self.assertTrue(np.allclose(merged.joint_positions[10], 1.0))

    def test_policy_control_requires_init_before_enable(self) -> None:
        default = np.zeros(2)
        control = PolicyControl(default, init_duration=1.0)
        current = np.ones(2)
        self.assertIn("ignored", control.handle("]", current, lateral=True, backward=True))
        control.handle("i", current, lateral=True, backward=True)
        started = control._init_started
        self.assertIn("already initializing", control.handle("i", current, lateral=True, backward=True))
        self.assertEqual(control._init_started, started)
        target = control.target(current, now=control._init_started + 0.5)
        self.assertTrue(np.allclose(target, 0.5))
        control.target(current, now=control._init_started + 1.0)
        self.assertEqual(control.mode, ControlMode.READY)
        control.handle("]", current, lateral=True, backward=True)
        self.assertEqual(control.mode, ControlMode.ENABLED)
        control.handle("w", current, lateral=True, backward=True)
        self.assertGreater(control.lin_x, 0)
        control.handle("q", current, lateral=True, backward=True)
        self.assertEqual((control.lin_x, control.lin_y), (0.0, 0.0))
        self.assertGreater(control.yaw, 0)

    def test_policy_enable_is_queued_during_initialization(self) -> None:
        control = PolicyControl(np.zeros(2), init_duration=1.0)
        current = np.ones(2)
        control.handle("i", current, lateral=True, backward=True)
        message = control.handle("]", current, lateral=True, backward=True)
        self.assertIn("queued", message)
        control.target(current, now=control._init_started + 1.0)
        self.assertEqual(control.mode, ControlMode.ENABLED)

    def test_holosoma_zero_command_uses_official_standing_phase(self) -> None:
        phase, standing = _advance_phase(np.asarray((0.0, np.pi)), False, (0.0, 0.0, 0.0))
        self.assertTrue(standing)
        self.assertTrue(np.allclose(phase, np.pi))
        phase, standing = _advance_phase(phase, standing, (0.1, 0.0, 0.0))
        self.assertFalse(standing)
        self.assertTrue(np.allclose(phase, (np.pi, 0.0)))

    def test_holosoma_walk_gate_and_velocity_ramp(self) -> None:
        control = PolicyControl(
            np.zeros(2),
            init_duration=1.0,
            require_walk_enable=True,
            smooth_velocity=True,
        )
        current = np.ones(2)
        control.handle("i", current, lateral=True, backward=True)
        control.target(current, now=control._init_started + 1.0)
        control.handle("]", current, lateral=True, backward=True)
        self.assertEqual(control.mode, ControlMode.ENABLED)
        self.assertIn("ignored", control.handle("w", current, lateral=True, backward=True))
        control.handle("=", current, lateral=True, backward=True)
        self.assertEqual(control.mode, ControlMode.WALKING)
        control.handle("w", current, lateral=True, backward=True)
        control.update_velocity(now=10.0)
        control.update_velocity(now=10.25)
        self.assertAlmostEqual(control.lin_x, 0.05)
        control.handle("=", current, lateral=True, backward=True)
        self.assertEqual(control.mode, ControlMode.ENABLED)
        self.assertEqual((control.lin_x, control.lin_y, control.yaw), (0.0, 0.0, 0.0))

    def test_holosoma_hold_gains_keep_strong_ankle_support(self) -> None:
        profile = load_control_profile(bundled_profile_path("holosoma"), "holosoma")
        self.assertTrue(np.all(profile.initialization.kp[[4, 5, 10, 11]] >= 150.0))
        self.assertTrue(np.all(profile.initialization.kd[[4, 5, 10, 11]] >= 5.0))

    def test_simulator_only_advances_for_a_complete_model_command(self) -> None:
        zeros = np.zeros(29)
        inactive = LowCommandSnapshot(zeros, zeros, zeros, zeros, zeros)
        self.assertFalse(inactive.has_control_authority())
        active = LowCommandSnapshot(zeros, zeros, zeros, np.ones(29), np.ones(29))
        self.assertTrue(active.has_control_authority())
        invalid = LowCommandSnapshot(zeros, zeros, zeros, np.full(29, np.nan), np.ones(29))
        self.assertFalse(invalid.has_control_authority())

    def test_viewer_keys_control_gantry_and_reset(self) -> None:
        controls = InteractiveControls(enabled=True)
        controls.key_callback(55)
        self.assertAlmostEqual(controls.length, 0.9)
        controls.key_callback(56)
        self.assertAlmostEqual(controls.length, 1.0)
        controls.key_callback(57)
        self.assertFalse(controls.enabled)
        controls.key_callback(259)
        self.assertTrue(controls.consume_reset())
        self.assertFalse(controls.consume_reset())
        controls.key_callback(89)
        self.assertTrue(controls.tracking_enabled())
        controls.key_callback(89)
        self.assertFalse(controls.tracking_enabled())

    def test_tracking_camera_preserves_mouse_orbit_and_zoom(self) -> None:
        class Camera:
            def __init__(self) -> None:
                self.lookat = np.zeros(3)
                self.azimuth = 17.0
                self.elevation = -11.0
                self.distance = 4.2

        camera = Camera()
        controls = InteractiveControls(camera_tracking=True)
        _update_tracking_camera(camera, controls, np.asarray((1.0, 2.0, 0.8)))
        self.assertTrue(np.allclose(camera.lookat, (1.0, 2.0, 0.8)))
        self.assertEqual((camera.azimuth, camera.elevation, camera.distance), (17.0, -11.0, 4.2))

    def test_gantry_force_matches_spring_damper_model(self) -> None:
        controls = InteractiveControls(enabled=True, length=1.0, stiffness=200.0, damping=100.0)
        controls.anchor = np.asarray((0.0, 0.0, 3.0))
        force = controls.force(np.asarray((0.0, 0.0, 1.0)), np.asarray((0.0, 0.0, 0.5)))
        self.assertTrue(np.allclose(force, (0.0, 0.0, 150.0)))

    def test_gantry_trolley_stays_above_robot(self) -> None:
        controls = InteractiveControls(enabled=True)
        controls.reset_anchor(np.asarray((0.0, 0.0, 0.8)))
        controls.follow_horizontally(np.asarray((1.5, -0.4, 0.8)))
        self.assertTrue(np.allclose(controls.anchor, (1.5, -0.4, 3.0)))

    def test_gantry_starts_with_guide_rest_length(self) -> None:
        controls = InteractiveControls(enabled=True)
        position = np.asarray((0.0, 0.0, 0.8))
        controls.reset_anchor(position)
        self.assertEqual(controls.length, 1.0)
        self.assertGreater(controls.force(position, np.zeros(3))[2], 0.0)

    def test_gantry_grounded_length_supports_robot_weight(self) -> None:
        controls = InteractiveControls(enabled=True)
        position = np.asarray((0.0, 0.0, 0.8))
        controls.reset_anchor(position)
        controls.set_grounded_length(position, total_mass=40.0, gravity=10.0)
        self.assertAlmostEqual(controls.length, 0.4)
        self.assertTrue(np.allclose(controls.force(position, np.zeros(3)), (0.0, 0.0, 360.0)))

    def test_enabled_gantry_adds_visible_rope_and_anchor(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        scene = mujoco.MjvScene(model, maxgeom=4)
        controls = InteractiveControls(enabled=True)
        controls.reset_anchor(np.asarray((0.0, 0.0, 0.8)))
        _draw_virtual_gantry(scene, controls, np.asarray((0.0, 0.0, 0.8)))
        self.assertEqual(scene.ngeom, 2)
        self.assertEqual(scene.geoms[0].type, mujoco.mjtGeom.mjGEOM_CAPSULE)
        self.assertEqual(scene.geoms[1].type, mujoco.mjtGeom.mjGEOM_SPHERE)
        controls.enabled = False
        _draw_virtual_gantry(scene, controls, np.asarray((0.0, 0.0, 0.8)))
        self.assertEqual(scene.ngeom, 0)

    def test_sdk_pythonpath_is_workspace_local(self) -> None:
        root = Path("/tmp/longship-test")
        paths = sdk_pythonpath(root)
        self.assertTrue(all(str(path).startswith(str(root)) for path in paths))

    def test_bundled_compatibility_lock_is_loadable(self) -> None:
        lock = CompatibilityLock.bundled()
        self.assertEqual(lock.platform_version, "1.0.0")
        self.assertEqual(lock.values["runtime"]["python"], "3.11")
        self.assertEqual(lock.values["contracts"]["action_dimension"], 29)
        self.assertNotIn(
            "nvidia_cuda_device",
            lock.values["backends"]["sonic"]["required_capabilities"],
        )
        self.assertNotIn(
            "nvidia_cuda_device",
            lock.values["backends"]["holosoma"]["required_capabilities"],
        )

    def test_compatibility_lock_rejects_unknown_fields(self) -> None:
        values = dict(CompatibilityLock.bundled().values)
        values["unexpected"] = True
        with self.assertRaisesRegex(CompatibilityLockError, "unknown fields"):
            CompatibilityLock.from_mapping(values)

    def test_lfs_pointer_is_not_treated_as_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pointer = Path(temporary) / "policy.onnx"
            pointer.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
                "size 1234\n",
                encoding="utf-8",
            )
            ready, message = inspect_artifact(pointer)
            self.assertFalse(ready)
            self.assertIn("Git LFS pointer", message)

    def test_reference_experiment_validates(self) -> None:
        experiment = ExperimentConfig.from_mapping(experiment_values())
        self.assertEqual(experiment.name, "g1_locomotion_ppo")
        self.assertEqual(experiment.values["training"]["trainer"]["type"], "PPO")

    def test_model_is_built_from_typed_slots(self) -> None:
        experiment = ExperimentConfig.from_mapping(experiment_values())
        model = build_model(experiment.values["model"], registry=model_registry())
        self.assertIsInstance(model, Policy)
        self.assertEqual(model.backbone.values["hidden_dims"], [512, 256, 128])
        self.assertEqual(model.actor_decoder.values["action_dim"], 29)

    def test_unknown_component_reports_registered_names(self) -> None:
        with self.assertRaisesRegex(RegistryError, "registered: none"):
            ComponentRegistry().create("encoder", {"type": "MissingEncoder"})

    def test_unknown_experiment_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ExperimentConfigError, "unknown fields"):
            ExperimentConfig.from_mapping(
                {
                    "schema_version": "longship.rl-experiment.v1",
                    "name": "bad",
                    "model": {"type": "Policy"},
                    "training": {
                        "backend": {"type": "Backend"},
                        "trainer": {"type": "PPO"},
                    },
                    "environment": {"robot": "g1", "task": "walk"},
                    "sim2sim": {},
                }
            )

    def test_runner_writes_resolved_config_before_dispatch(self) -> None:
        experiment = ExperimentConfig.from_mapping(experiment_values())
        registry = ComponentRegistry()
        backend = RecordingBackend()
        registry.register("training_backend", "HoloSomaBackend")(lambda **_: backend)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run-001"
            checkpoint = ExperimentRunner(registry).run(experiment, output)
            self.assertTrue((output / "resolved.yaml").is_file())
            self.assertEqual(checkpoint, output / "checkpoints" / "last.pt")
            self.assertEqual(len(backend.calls), 1)
            with self.assertRaises(FileExistsError):
                ExperimentRunner(registry).run(experiment, output)


if __name__ == "__main__":
    unittest.main()
