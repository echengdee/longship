from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from longship.artifacts import sha256_file
from longship.policies import UNITREE_RL_GYM_G1_POLICY_SHA256


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "plugins/targets/mujoco_g1_policy/runner.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "longship_mujoco_g1_policy_runner", MODULE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load external G1 policy runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _config_payload() -> dict[str, object]:
    return {
        "policy_path": "external-policy.pt",
        "xml_path": "external-scene.xml",
        "simulation_duration": 10.0,
        "simulation_dt": 0.002,
        "control_decimation": 10,
        "kps": [80.0] * 12,
        "kds": [3.0] * 12,
        "default_angles": [0.0] * 12,
        "ang_vel_scale": 0.2,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 0.1,
        "action_scale": 0.2,
        "cmd_scale": [1.0, 1.0, 1.0],
        "num_actions": 12,
        "num_obs": 47,
        "cmd_init": [0.0, 0.0, 0.0],
    }


class ExternalG1PolicyPluginTests(unittest.TestCase):
    def test_external_config_contract_is_strict(self) -> None:
        module = _load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text(json.dumps(_config_payload()), encoding="utf-8")
            with patch.dict(sys.modules, {"yaml": None}):
                config = module.ExternalPolicyConfig.load(path)

            self.assertEqual(config.policy_period_s, 0.02)
            payload = _config_payload()
            payload["unexpected"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(sys.modules, {"yaml": None}):
                with self.assertRaisesRegex(ValueError, "unexpected shape"):
                    module.ExternalPolicyConfig.load(path)

    def test_target_clamps_only_machine_precision_boundary_error(self) -> None:
        module = _load_runner()
        target = module.G1PolicyVelocityTarget(
            maximum_forward_mps=0.15,
            maximum_yaw_rate_radps=0.2,
        )
        self.assertTrue(target.acquire("session", 100).accepted)
        command = module.FollowCommand(
            session_id="session",
            sequence=1,
            issued_monotonic_ns=100,
            expires_monotonic_ns=200,
            forward_mps=0.1500000005,
            yaw_rate_radps=-0.2000000005,
            reason="bounded numerical input",
        )

        self.assertTrue(target.apply(command).accepted)
        self.assertEqual(target.desired_velocity(), (0.15, 0.0, -0.2))
        target.set_time(200)
        self.assertEqual(target.desired_velocity(), (0.0, 0.0, 0.0))

    def test_artifact_verification_rejects_scene_outside_bundle(self) -> None:
        module = _load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "asset.bin").write_bytes(b"asset")
            scene = root / "outside.xml"
            scene.write_text("<mujoco/>", encoding="utf-8")
            policy = root / "policy.pt"
            policy.write_bytes(b"policy")
            config = root / "config.yaml"
            config.write_text("config", encoding="utf-8")
            license_file = root / "LICENSE"
            license_file.write_text(
                "BSD 3-Clause License\nexternal record\n", encoding="utf-8"
            )
            bundle_digest, _, _ = module.sha256_directory(bundle)
            args = argparse.Namespace(
                scene=scene,
                scene_bundle_root=bundle,
                policy=policy,
                policy_config=config,
                license=license_file,
                expected_scene_bundle_sha256=bundle_digest,
                expected_policy_sha256=sha256_file(policy),
                expected_config_sha256=sha256_file(config),
                expected_license_sha256=sha256_file(license_file),
            )

            with self.assertRaisesRegex(ValueError, "outside"):
                module.verify_artifacts(args)

    def test_interactive_report_requires_real_target_commands(self) -> None:
        module = _load_runner()
        pipeline = SimpleNamespace(
            exit_reason="operator_stop",
            terminal_state=module.FollowState.STOPPED,
            brain_requests=1,
            accepted_skill_calls=1,
            control_steps=2,
            target_command_steps=0,
        )
        artifacts = module.ArtifactIdentity("manifest", "m", "a", "b", "c", "d")
        world = SimpleNamespace(fallen=False, contact_steps=0)
        target = SimpleNamespace(stop_verified=True)
        report = module.G1DynamicReport(
            "stack", pipeline, artifacts, world, target
        )

        self.assertFalse(report.passed)
        pipeline.target_command_steps = 1
        self.assertTrue(report.passed)

    def test_hud_sink_publishes_initial_camera_before_runtime_events(self) -> None:
        module = _load_runner()
        published: list[tuple[int, bytes, str]] = []
        dashboard = SimpleNamespace(
            publish_camera_frame=lambda sequence, jpeg, source: published.append(
                (sequence, jpeg, source)
            )
        )
        world = SimpleNamespace(
            render_camera_jpeg=lambda: (0, b"\xff\xd8initial\xff\xd9")
        )
        sink = module.G1HudEventSink(
            SimpleNamespace(publish=lambda event: None),
            dashboard,
            world,
            brain_provider="deterministic",
            brain_model=None,
            brain_reasoning_effort=None,
        )

        self.assertTrue(sink.publish_current_camera())
        self.assertEqual(
            published,
            [(0, b"\xff\xd8initial\xff\xd9", "g1-pelvis-sim-camera")],
        )

    def test_model_manifest_uses_new_artifact_contract_and_blocks_prefetch(
        self,
    ) -> None:
        module = _load_runner()
        manifest = module.load_model_artifact_manifest(
            ROOT
            / "plugins/targets/mujoco_g1_policy/model-artifacts.experimental.json"
        )

        self.assertEqual(manifest.manifest_id, module._MODEL_MANIFEST_ID)
        self.assertFalse(manifest.prefetch_eligible)
        self.assertEqual(
            manifest.artifact("motion.pt").sha256,
            UNITREE_RL_GYM_G1_POLICY_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
