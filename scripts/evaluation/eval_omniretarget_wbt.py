#!/usr/bin/env python3
"""Headless, metric-based evaluation for an object-aware HoloSoma WBT checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Callable

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.video import CartesianCameraConfig, VideoConfig
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.helpers import get_class
from holosoma.utils.rotations import quat_error_magnitude
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=8, help="Independent noisy evaluation trials.")
    parser.add_argument(
        "--simulator",
        choices=("saved", "mujoco"),
        default="saved",
        help="Use the checkpoint's training simulator or CPU MuJoCo for non-disruptive monitoring.",
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        default=None,
        metavar="DIR",
        help="Record env 0 to DIR and continue through bad-tracking for the full reference clip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/omniretarget_wbt_eval.json"),
        help="JSON report path, relative to the current working directory.",
    )
    return parser.parse_args()


def scalar_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def evaluation_config(
    saved: ExperimentConfig,
    num_envs: int,
    video_dir: Path | None,
    simulator: str,
) -> ExperimentConfig:
    cfg = saved.get_eval_config()
    if simulator == "mujoco":
        from holosoma.config_values.simulator import mujoco

        rigid_objects = {
            name: (
                dataclasses.replace(
                    obj,
                    xml_file="holosoma/data/scene_objects/boxes/large_box.xml",
                )
                if obj.xml_file is None and obj.urdf_file and obj.urdf_file.endswith("/large_box.urdf")
                else obj
            )
            for name, obj in cfg.scene.rigid_objects.items()
        }
        cfg = dataclasses.replace(
            cfg,
            scene=dataclasses.replace(cfg.scene, rigid_objects=rigid_objects),
            randomization=dataclasses.replace(
                cfg.randomization,
                # The joint action term expects this setup hook to create its
                # delay state even when delay randomization is disabled.
                setup_terms={
                    "setup_action_delay_buffers": cfg.randomization.setup_terms[
                        "setup_action_delay_buffers"
                    ]
                },
                reset_terms={},
                step_terms={},
                ignore_unsupported=True,
            ),
            simulator=dataclasses.replace(
                mujoco,
                config=dataclasses.replace(
                    mujoco.config,
                    sim=dataclasses.replace(
                        mujoco.config.sim,
                        max_episode_length_s=cfg.simulator.config.sim.max_episode_length_s,
                    ),
                ),
            ),
        )
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(
            cfg.training,
            headless=True,
            num_envs=num_envs,
            export_onnx=False,
            max_eval_steps=None,
        ),
    )
    if video_dir is None:
        return cfg

    video = VideoConfig(
        enabled=True,
        interval=1,
        # The stock G1 MJCF exposes a 640 px offscreen framebuffer.
        width=640 if simulator == "mujoco" else 960,
        height=360 if simulator == "mujoco" else 540,
        playback_rate=1.0,
        output_format="h264",
        save_dir=str(video_dir.resolve()),
        upload_to_wandb=False,
        show_command_overlay=False,
        record_env_id=0,
        camera=CartesianCameraConfig(
            offset=[2.8, 2.8, 1.8],
            target_offset=[0.0, 0.0, 0.8],
            smoothing=0.85,
            tracking_body_name="pelvis",
        ),
    )
    logger = dataclasses.replace(cfg.logger, video=video, headless_recording=True)
    # Keep simulating after the configured failure threshold so the recording
    # shows the complete physical outcome instead of ending at the first bad frame.
    termination = dataclasses.replace(
        cfg.termination,
        terms={name: term for name, term in cfg.termination.terms.items() if name != "bad_tracking"},
    )
    return dataclasses.replace(cfg, logger=logger, termination=termination)


def configure_streaming_video(algo, video_dir: Path | None) -> tuple[Path | None, Callable[[], None]]:
    """Stream frames to a separate FFmpeg process so native simulator exits cannot corrupt MP4."""
    if video_dir is None:
        return None, lambda: None

    import numpy as np

    recorder = algo.unwrapped_env.simulator.video_recorder
    if recorder is None:
        raise RuntimeError("Video recording was requested but the simulator has no recorder.")

    video_dir.mkdir(parents=True, exist_ok=True)
    output = video_dir / f"omniretarget_wbt_{int(algo.global_step):07d}_h264.mp4"
    config = recorder.config
    sim_config = algo.unwrapped_env.simulator.simulator_config.sim
    fps = float(sim_config.fps / sim_config.control_decimation_steps * config.playback_rate)
    ffmpeg = os.environ.get(
        "LONGSHIP_FFMPEG",
        "/home/qcraft/miniconda3/envs/gmr/lib/python3.10/site-packages/"
        "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2",
    )
    if not Path(ffmpeg).is_file():
        raise RuntimeError(f"FFmpeg executable not found: {ffmpeg}")
    process = subprocess.Popen(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{config.width}x{config.height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    closed = False

    def write_frame(frame) -> None:
        frame_uint8 = np.ascontiguousarray(frame, dtype=np.uint8)
        if process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        process.stdin.write(frame_uint8.tobytes())

    def finish() -> None:
        nonlocal closed
        if not closed:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=30)
            closed = True

    recorder._add_frame = write_frame
    recorder._encode_and_save_video = finish
    return output, finish


@torch.no_grad()
def evaluate(algo, checkpoint: Path) -> dict:
    algo._pre_evaluate_policy()
    obs = algo.env.reset()
    command = algo.unwrapped_env.command_manager.get_state("motion_command")
    if command is None or not command.motion.has_object:
        raise RuntimeError("The checkpoint is not configured with an object-aware motion command.")

    num_envs = algo.env.num_envs
    total_motion_frames = int(command.motion.time_step_total)
    active = torch.ones(num_envs, dtype=torch.bool, device=algo.device)
    completed = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    survival_frames = torch.zeros(num_envs, dtype=torch.long, device=algo.device)
    failure_reference_frames = torch.full((num_envs,), -1, dtype=torch.long, device=algo.device)
    failure_causes: list[list[str]] = [[] for _ in range(num_envs)]
    previous_time = command.time_steps.clone()

    tracked: dict[str, list[float]] = {
        "reward": [],
        "reference_position_error_m": [],
        "reference_rotation_error_rad": [],
        "body_position_error_m": [],
        "joint_position_l2_error_rad": [],
        "object_position_error_m": [],
        "object_rotation_error_rad": [],
    }

    # The extra margin is only a guard. A trial normally exits at a bad-tracking
    # termination or when MotionCommand wraps after the final reference frame.
    for _ in range(total_motion_frames + 8):
        if not bool(active.any()):
            break

        command.update_metrics()
        metric_map = command.metrics
        object_position_error = torch.linalg.vector_norm(
            command.object_pos_w - command.simulator_object_pos_w, dim=-1
        )
        object_rotation_error = quat_error_magnitude(
            command.object_quat_w, command.simulator_object_quat_w
        )

        if algo.obs_normalization:
            actor_obs = algo.obs_normalizer(obs, update=False)
        else:
            actor_obs = obs
        actions = algo.actor(actor_obs)[0]
        obs, rewards, dones, _ = algo.env.step(actions)

        active_indices = torch.where(active)[0]
        survival_frames[active_indices] += 1
        tensor_metrics = {
            "reward": rewards,
            "reference_position_error_m": metric_map["motion/error_ref_pos"],
            "reference_rotation_error_rad": metric_map["motion/error_ref_rot"],
            "body_position_error_m": metric_map["motion/error_body_pos"],
            "joint_position_l2_error_rad": metric_map["motion/error_joint_pos"],
            "object_position_error_m": object_position_error,
            "object_rotation_error_rad": object_rotation_error,
        }
        for name, tensor in tensor_metrics.items():
            tracked[name].extend(tensor[active_indices].detach().cpu().tolist())

        done_now = dones.bool() & active
        if bool(done_now.any()):
            termination_manager = algo.unwrapped_env.termination_manager
            bad_tracking = termination_manager._term_instances.get("bad_tracking")
            if bad_tracking is not None:
                for cause, mask in getattr(bad_tracking, "last_failures", {}).items():
                    for env_id in torch.where(done_now & mask)[0].detach().cpu().tolist():
                        failure_causes[env_id].append(cause)
            failed |= done_now
            failure_reference_frames[done_now] = previous_time[done_now]
            active &= ~done_now

        current_time = command.time_steps.clone()
        wrapped_now = (current_time < previous_time) & active
        if bool(wrapped_now.any()):
            completed |= wrapped_now
            active &= ~wrapped_now
        previous_time = current_time

    unfinished = active.clone()
    algo._post_evaluate_policy()

    results = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": int(algo.global_step),
        "evaluation_mode": "checkpoint_randomization",
        "reference_frames": total_motion_frames,
        "reference_fps": int(command.motion.fps.item()),
        "trials": num_envs,
        "completed_trials": int(completed.sum().item()),
        "failed_trials": int(failed.sum().item()),
        "unfinished_trials": int(unfinished.sum().item()),
        "completion_rate": float(completed.float().mean().item()),
        "survival_frames": survival_frames.detach().cpu().tolist(),
        "failure_reference_frames": failure_reference_frames.detach().cpu().tolist(),
        "failure_causes": failure_causes,
        "metrics": {name: scalar_summary(values) for name, values in tracked.items()},
    }
    return results


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")

    saved_config, saved_wandb_path = load_saved_experiment_config(
        CheckpointConfig(checkpoint=str(checkpoint))
    )
    video_dir = args.record_video.resolve() if args.record_video is not None else None
    if video_dir is not None and args.num_envs != 1:
        raise ValueError("Video recording requires --num-envs 1")
    config = evaluation_config(saved_config, args.num_envs, video_dir, args.simulator)
    simulation_app = None
    algo = None
    finish_video: Callable[[], None] = lambda: None
    try:
        env, device, simulation_app = setup_simulation_environment(config)
        algo_class = get_class(config.algo._target_)
        log_dir = args.output.resolve().parent / "runtime"
        log_dir.mkdir(parents=True, exist_ok=True)
        algo = algo_class(device=device, env=env, config=config.algo.config, log_dir=str(log_dir), multi_gpu_cfg=None)
        algo.setup()
        algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
        algo.load(str(checkpoint))
        video_path, finish_video = configure_streaming_video(algo, video_dir)

        report = evaluate(algo, checkpoint)
        report["evaluation_simulator"] = config.simulator.config.name
        if video_path is not None:
            report["video_path"] = str(video_path)
            report["bad_tracking_termination_enabled"] = False
            report["completion_semantics"] = "full reference playback for visualization; not policy success"
        else:
            report["bad_tracking_termination_enabled"] = True
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Evaluation report: {output}")
    finally:
        finish_video()
        if algo is not None and hasattr(algo, "writer"):
            algo.writer.close()
        if algo is not None:
            algo.unwrapped_env.simulator.close()
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
