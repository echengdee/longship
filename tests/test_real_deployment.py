from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from longship.rl.deploy.launch import build_deployment_launch
from longship.rl.deploy.profile import DeploymentProfile


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src/longship/rl/deploy/profiles/hiking_g1.yaml"


def _profile() -> DeploymentProfile:
    return DeploymentProfile.load(PROFILE, ROOT)


def test_registered_profile_reuses_sim2sim_controller_and_policy_profile() -> None:
    profile = _profile()
    launch = build_deployment_launch(ROOT, "/runtime/python", profile)

    assert profile.control_profile == (
        ROOT / "src/longship/rl/sim2sim/profiles/instinctlab.yaml"
    )
    assert str(ROOT / "src/longship/rl/sim2sim/adapters/instinctlab_dds.py") in (
        launch.backend.controller.argv
    )
    assert "--real-robot" in launch.backend.controller.argv
    clock = launch.backend.controller.argv.index("--clock")
    assert launch.backend.controller.argv[clock + 1] == "wall"
    assert "longship.rl.sim2sim.teleop" in launch.backend.teleop.argv


def test_sensor_is_configuration_not_a_model_specific_runner() -> None:
    profile = _profile()
    launch = build_deployment_launch(ROOT, "/runtime/python", profile)

    assert len(profile.sensors) == 1
    sensor_profile = profile.sensors[0]
    sensor = launch.sensors[0].process.argv
    assert sensor_profile.type == "realsense_depth_dds"
    assert (sensor_profile.config["raw_width"], sensor_profile.config["raw_height"]) == (
        848,
        480,
    )
    assert (
        sensor_profile.config["output_width"],
        sensor_profile.config["output_height"],
    ) == (480, 270)
    assert "longship.rl.deploy.realsense_depth_dds" in sensor


def test_visual_monitor_is_shared_and_both_frame_producers_connect_to_it() -> None:
    profile = _profile()
    launch = build_deployment_launch(ROOT, "/runtime/python", profile)

    assert profile.visualization is not None
    assert launch.monitor is not None
    assert "longship.rl.deploy.web_monitor" in launch.monitor.argv
    endpoint = profile.visualization.frame_endpoint
    assert endpoint in launch.backend.controller.argv
    assert endpoint in launch.sensors[0].process.argv
    assert "--debug-frame-fps" in launch.backend.controller.argv


def test_deployment_rejects_loopback_as_robot_interface() -> None:
    profile = _profile()

    with pytest.raises(ValueError, match="non-loopback"):
        replace(profile, dds=replace(profile.dds, interface="lo")).validate()


def test_deployment_enables_crc_path_and_input_watchdog() -> None:
    launch = build_deployment_launch(ROOT, "/runtime/python", _profile())

    assert "--input-timeout" in launch.backend.controller.argv
    assert "longship.rl.deploy.unitree_motion" in launch.release_motion.argv
    assert "--release" in launch.release_motion.argv


def test_runner_and_launch_are_not_named_after_a_model() -> None:
    runner = (ROOT / "src/longship/rl/deploy/runner.py").read_text(encoding="utf-8").lower()
    launch = (ROOT / "src/longship/rl/deploy/launch.py").read_text(encoding="utf-8").lower()

    assert "hiking" not in runner
    assert "hiking" not in launch


def test_unknown_backend_fails_at_adapter_registry_boundary() -> None:
    profile = replace(_profile(), backend="future_policy")

    with pytest.raises(ValueError, match="no physical deployment adapter"):
        build_deployment_launch(ROOT, "/runtime/python", profile)
