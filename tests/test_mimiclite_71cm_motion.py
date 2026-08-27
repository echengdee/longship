from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from longship_runtime.runtime.motion_reference import load_g1_reference
from longship_runtime.runtime.profile import load_control_profile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/training/build_mimiclite_71cm_motion.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_mimiclite_71cm_motion", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_motion_has_continuous_normalized_qpos() -> None:
    builder = _load_builder()
    source = ROOT / "third_party/OmniRetarget_Dataset/robot-terrain/climb_12_z_scale_1.0.npz"
    qpos, phases = builder.build_motion(
        source,
        climb_end_frame=240,
        cargo_floor_height_m=0.71,
        target_fps=50.0,
        settle_s=0.25,
        sit_s=3.0,
        hold_s=1.0,
        move_inside_m=0.25,
        turn_deg=180.0,
    )

    assert qpos.ndim == 2 and qpos.shape[1] == 36
    assert phases["climb_end"] < phases["settle_end"] < phases["turn_sit_end"] < phases["motion_end"]
    np.testing.assert_allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, atol=1e-5)
    assert np.isfinite(qpos).all()
    assert float(qpos[-1, 2]) == np.float32(0.88)
    assert float(np.max(np.abs(np.diff(qpos[:, :3], axis=0)), axis=0).max()) < 0.04
    # End pose is symmetric and bent, unlike the rejected two-knee kneel sample.
    np.testing.assert_allclose(qpos[-1, [7, 13]], -2.40, atol=1e-6)
    np.testing.assert_allclose(qpos[-1, [10, 16]], 1.97, atol=1e-6)


def test_generated_motion_loads_in_sim2sim_and_truck_is_aligned() -> None:
    dataset = ROOT / "third_party/mimiclite-assets/g1-climb-turn-sit-71cm"
    motion = dataset / "motions/climb_turn_floor_sit_71cm.npz"
    if not motion.exists():
        _load_builder().main()
    reference = load_g1_reference(motion)
    assert reference.frames == 611
    assert reference.fps == 50.0
    np.testing.assert_allclose(reference.root_positions[-1, 2], 0.88, atol=1e-6)

    profile = load_control_profile(
        ROOT
        / "modules/longship-sim2real/src/longship_runtime/runtime/profiles/mimiclite_box_truck.yaml",
        "mimiclite",
    )
    assert profile.simulator.box_truck is not None
    assert profile.simulator.box_truck.cargo_floor_height_m == 0.71
    assert profile.policy_options["motion"].endswith("climb_turn_floor_sit_71cm.npz")
