"""Pure-Python SONIC planner/encoder/decoder pipeline using ONNX Runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

import numpy as np

from longship.rl.runtime import OnnxEngine


ISAACLAB_TO_MUJOCO = np.asarray(
    (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28),
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.asarray(
    (0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28),
    dtype=np.int64,
)

_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_MOTOR_TYPES = (
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020", "5020", "5020", "5020", "5020", "5020",
    "4010", "4010", "5020", "5020", "5020", "5020", "5020", "4010", "4010",
)
_OMEGA = 10.0 * 2.0 * math.pi
KP = np.asarray([_ARMATURE[kind] * _OMEGA**2 for kind in _MOTOR_TYPES], dtype=np.float64)
KP[[4, 5, 10, 11, 13, 14]] *= 2.0
KD = np.asarray([2.0 * 2.0 * _ARMATURE[kind] * _OMEGA for kind in _MOTOR_TYPES], dtype=np.float64)
KD[[4, 5, 10, 11, 13, 14]] *= 2.0
ACTION_SCALE = np.asarray(
    [0.25 * _EFFORT[kind] / (_ARMATURE[kind] * _OMEGA**2) for kind in _MOTOR_TYPES],
    dtype=np.float64,
)
DEFAULT_Q = np.asarray(
    (-0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0),
    dtype=np.float64,
)


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1.0e-8 else np.asarray((1.0, 0.0, 0.0, 0.0))


def _quat_conjugate(value: np.ndarray) -> np.ndarray:
    return np.asarray((value[0], -value[1], -value[2], -value[3]))


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        (w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
         w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2)
    )


def _quat_matrix(value: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(value)
    return np.asarray(
        ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
         (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
         (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    )


def _projected_gravity(value: np.ndarray) -> np.ndarray:
    return _quat_matrix(value).T @ np.asarray((0.0, 0.0, -1.0))


def _orientation_6d(base: np.ndarray, reference: np.ndarray) -> np.ndarray:
    relative = _quat_mul(_quat_conjugate(_normalize_quaternion(base)), _normalize_quaternion(reference))
    return _quat_matrix(relative)[:, :2].reshape(-1)


def allowed_pred_num_tokens(mode: int) -> np.ndarray:
    """Return SONIC's released 6..16-token selection mask for a mode."""
    allowed = np.zeros((1, 11), dtype=np.int64)
    allowed[:, :6] = 1
    if mode in (10, 14):  # walk-boxing and elbow-crawling
        allowed[:] = 1
    return allowed


@dataclass(frozen=True, slots=True)
class SonicRobotState:
    tick: int
    q: np.ndarray
    dq: np.ndarray
    quaternion: np.ndarray
    gyroscope: np.ndarray
    torso_quaternion: np.ndarray
    torso_gyroscope: np.ndarray


@dataclass(frozen=True, slots=True)
class SonicMotion:
    joint_positions: np.ndarray  # [T, 29], IsaacLab order
    joint_velocities: np.ndarray  # [T, 29], IsaacLab order
    root_positions: np.ndarray  # [T, 3], world frame
    root_quaternions: np.ndarray  # [T, 4], wxyz

    @property
    def frames(self) -> int:
        return int(self.joint_positions.shape[0])


SONIC_MODE_NAMES = {
    0: "IDLE",
    1: "SLOW_WALK",
    2: "WALK",
    3: "RUN",
    4: "IDLE_SQUAT",
    5: "IDLE_KNEEL_TWO_LEGS",
    6: "IDLE_KNEEL",
    7: "IDLE_LYING_FACE_DOWN",
    8: "CRAWLING",
    9: "IDLE_BOXING",
    10: "WALK_BOXING",
    11: "LEFT_PUNCH",
    12: "RIGHT_PUNCH",
    13: "RANDOM_PUNCH",
    14: "ELBOW_CRAWLING",
    15: "LEFT_HOOK",
    16: "RIGHT_HOOK",
    17: "FORWARD_JUMP",
    18: "STEALTH_WALK",
    19: "INJURED_WALK",
    20: "LEDGE_WALKING",
    21: "OBJECT_CARRYING",
    22: "STEALTH_WALK_2",
    23: "HAPPY_DANCE_WALK",
    24: "ZOMBIE_WALK",
    25: "GUN_WALK",
    26: "SCARE_WALK",
}
SONIC_MODE_SETS = (
    ("standing", (1, 2, 3, 17, 18, 19)),
    ("squat/crawl", (4, 5, 6, 8, 14)),
    ("boxing", (9, 10, 11, 12, 13, 15, 16)),
    ("styled walk", (20, 21, 22, 23, 24, 25, 26)),
)
SONIC_STATIC_MODES = frozenset((0, 4, 5, 6, 7, 9))
SONIC_SQUAT_MODES = frozenset((4, 5, 6, 7, 8, 14))
SONIC_DEFAULT_SPEEDS = {
    1: 0.4,
    2: 0.6,
    3: 1.5,
    8: 0.4,
    10: 0.7,
    11: 0.7,
    12: 0.7,
    13: 0.7,
    14: 0.7,
    15: 0.7,
    16: 0.7,
}


@dataclass(slots=True)
class SonicPlannerCommand:
    mode: int = 0
    mode_set: int = 0
    movement: np.ndarray | None = None
    facing_angle: float = 0.0
    target_speed: float = -1.0
    target_height: float = -1.0

    def __post_init__(self) -> None:
        if self.movement is None:
            self.movement = np.zeros(3, dtype=np.float32)

    @property
    def facing(self) -> np.ndarray:
        return np.asarray((math.cos(self.facing_angle), math.sin(self.facing_angle), 0.0), dtype=np.float32)

    @property
    def mode_name(self) -> str:
        return SONIC_MODE_NAMES[self.mode]

    def _select_mode(self, mode: int) -> str:
        self.mode = mode
        self.movement.fill(0.0)
        self.target_speed = -1.0 if mode in SONIC_STATIC_MODES else 0.0
        self.target_height = 0.8 if mode in SONIC_SQUAT_MODES else -1.0
        return f"SONIC planner mode={self.mode_name} ({mode})"

    def handle(self, key: str) -> str:
        if key in "np":
            delta = 1 if key == "n" else -1
            self.mode_set = (self.mode_set + delta) % len(SONIC_MODE_SETS)
            set_name, modes = SONIC_MODE_SETS[self.mode_set]
            selected = self._select_mode(modes[0])
            options = ", ".join(
                f"{index + 1}:{SONIC_MODE_NAMES[mode]}" for index, mode in enumerate(modes)
            )
            return f"SONIC mode set={set_name}; {selected}; choices=[{options}]"
        if key in "12345678":
            set_name, modes = SONIC_MODE_SETS[self.mode_set]
            index = int(key) - 1
            if index >= len(modes):
                return f"SONIC mode key {key} unsupported in set={set_name}"
            return self._select_mode(modes[index])
        if key in "90":
            if self.mode in SONIC_STATIC_MODES or self.target_speed < 0.0:
                return f"SONIC speed key {key} ignored for mode={self.mode_name}"
            delta = -0.1 if key == "9" else 0.1
            self.target_speed = max(0.0, self.target_speed + delta)
            return f"SONIC planner speed={self.target_speed:.1f} m/s"
        if key in "-=":
            if self.target_height < 0.0:
                return f"SONIC height key {key} ignored for mode={self.mode_name}"
            delta = -0.1 if key == "-" else 0.1
            self.target_height = min(0.8, max(0.2, self.target_height + delta))
            return f"SONIC planner height={self.target_height:.1f} m"
        if key == "r":
            self.movement = np.zeros(3, dtype=np.float32)
            self.target_speed = -1.0 if self.mode in SONIC_STATIC_MODES else 0.0
            return "SONIC planner emergency stop"
        if key in "qe":
            self.movement = np.zeros(3, dtype=np.float32)
            self.target_speed = -1.0 if self.mode in SONIC_STATIC_MODES else 0.0
            self.facing_angle += math.pi / 6.0 if key == "q" else -math.pi / 6.0
            return f"SONIC planner facing={math.degrees(self.facing_angle):.0f} deg"
        if key not in "wasd":
            return f"SONIC planner key {key!r} unsupported"
        if self.mode == 0 and key in "wasd":
            self._select_mode(1)
        if key in "wasd" and self.mode in SONIC_STATIC_MODES:
            return f"SONIC motion key {key!r} unsupported for static mode={self.mode_name}"
        self.target_speed = SONIC_DEFAULT_SPEEDS.get(self.mode, 0.4)
        facing = self.facing
        if key == "w":
            self.movement = facing.copy()
        elif key == "s":
            self.movement = -facing
        elif key == "a":
            self.movement = np.asarray((-facing[1], facing[0], 0.0), dtype=np.float32)
        elif key == "d":
            self.movement = np.asarray((facing[1], -facing[0], 0.0), dtype=np.float32)
        return (
            f"SONIC planner mode={self.mode_name} ({self.mode}) "
            f"speed={self.target_speed:.1f} movement={self.movement.tolist()}"
        )


class SonicOnnxPlanner:
    def __init__(self, model: Path, provider: str) -> None:
        self.engine = OnnxEngine(model, provider=provider, intra_op_threads=4)

    def infer(self, context: np.ndarray, command: SonicPlannerCommand) -> SonicMotion:
        context = np.asarray(context, dtype=np.float32).reshape(1, 4, 36)
        # Indices map to 6..16 predicted tokens.  SONIC's released G1 command
        # table permits 6..11 tokens for ordinary locomotion.  This mask is
        # model-visible: selecting the trailing entries changes the generated
        # gait distribution, not merely the planner latency.
        allowed = allowed_pred_num_tokens(command.mode)
        feeds = {
            "context_mujoco_qpos": context,
            "target_vel": np.asarray((command.target_speed,), dtype=np.float32),
            "mode": np.asarray((command.mode,), dtype=np.int64),
            "movement_direction": np.asarray(command.movement, dtype=np.float32)[None],
            "facing_direction": command.facing[None],
            "random_seed": np.asarray((1234,), dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": allowed,
            "height": np.asarray((command.target_height,), dtype=np.float32),
        }
        predicted, count = self.engine.infer(feeds, ("mujoco_qpos", "num_pred_frames"))
        frames_30 = int(np.asarray(count).reshape(-1)[0])
        if not 2 <= frames_30 <= 64:
            raise RuntimeError(f"SONIC planner returned invalid frame count {frames_30}")
        qpos = np.asarray(predicted, dtype=np.float64).reshape(-1, 36)[:frames_30]
        frames_50 = max(2, int(math.floor(frames_30 / 30.0 * 50.0)))
        sample = np.arange(frames_50, dtype=np.float64) * 30.0 / 50.0
        low = np.floor(sample).astype(np.int64)
        high = np.minimum(low + 1, frames_30 - 1)
        weight = sample - low
        hardware_positions = (1.0 - weight[:, None]) * qpos[low, 7:] + weight[:, None] * qpos[high, 7:]
        positions = hardware_positions[:, MUJOCO_TO_ISAACLAB]
        velocities = np.empty_like(positions)
        velocities[:-1] = (positions[1:] - positions[:-1]) * 50.0
        velocities[-1] = velocities[-2]
        root_positions = (1.0 - weight[:, None]) * qpos[low, :3] + weight[:, None] * qpos[high, :3]
        quaternions = np.empty((frames_50, 4), dtype=np.float64)
        for index, (f0, f1, alpha) in enumerate(zip(low, high, weight, strict=True)):
            q0, q1 = qpos[f0, 3:7], qpos[f1, 3:7]
            if np.dot(q0, q1) < 0:
                q1 = -q1
            quaternions[index] = _normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)
        return SonicMotion(positions, velocities, root_positions, quaternions)


class SonicOnnxPipeline:
    """SONIC's model-specific pipeline behind Longship's common runtime API."""

    def __init__(self, decoder: Path, encoder: Path, planner: Path, provider: str = "auto") -> None:
        self.decoder = OnnxEngine(decoder, provider=provider, intra_op_threads=4)
        self.encoder = OnnxEngine(encoder, provider=provider, intra_op_threads=4)
        self.planner = SonicOnnxPlanner(planner, provider)
        encoder_shape = self.encoder.session.get_inputs()[0].shape
        self.encoder_dimension = int(encoder_shape[-1])
        if self.encoder_dimension not in (1247, 1762):
            raise ValueError(f"unsupported SONIC encoder dimension {self.encoder_dimension}")
        self.command = SonicPlannerCommand()
        self.motion: SonicMotion | None = None
        self.motion_frame = 0
        self.last_action = np.zeros(29, dtype=np.float64)
        self.last_policy_observation: np.ndarray | None = None
        self.history: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=10)
        self._motion_lock = threading.Lock()
        self._planning = False
        self._planner_thread: threading.Thread | None = None
        self._pending_replan_state: SonicRobotState | None = None
        self._command_generation = 0
        self._closed = False
        self.planner_active = False
        self._plan_error: BaseException | None = None
        self.last_plan_started = 0.0
        self.sim_time = 0.0

    @property
    def providers(self) -> tuple[str, ...]:
        return self.decoder.providers

    def initialize_planner(self, state: SonicRobotState) -> None:
        # Native SONIC does not start planner playback on `]`.  It seeds a
        # measured one-frame reference and lets the tracking policy settle;
        # planner playback starts only after a planner command is received.
        seed = SonicMotion(
            state.q[MUJOCO_TO_ISAACLAB][None].copy(),
            np.zeros((1, 29), dtype=np.float64),
            np.asarray(((0.0, 0.0, 0.788740),), dtype=np.float64),
            np.asarray(((1.0, 0.0, 0.0, 0.0),), dtype=np.float64),
        )
        with self._motion_lock:
            self.motion = seed
            self.motion_frame = 0
        self.last_plan_started = self.sim_time

    def request_replan(self, state: SonicRobotState, *, force: bool = False) -> bool:
        now = self.sim_time
        if self._closed:
            return False
        if self._planning:
            if force:
                self._pending_replan_state = state
            return False
        if not force and now - self.last_plan_started < 1.0:
            return False
        with self._motion_lock:
            source_motion = self.motion
            source_frame = self.motion_frame
        if source_motion is None:
            return False
        gen_frame = source_frame + 2
        context = self._planner_context(source_motion, gen_frame)
        command = SonicPlannerCommand(
            mode=self.command.mode,
            movement=self.command.movement.copy(),
            facing_angle=self.command.facing_angle,
            target_speed=self.command.target_speed,
            target_height=self.command.target_height,
        )
        command_generation = self._command_generation
        self._planning = True
        self.last_plan_started = now

        def worker() -> None:
            try:
                generated = self.planner.infer(context, command)
                with self._motion_lock:
                    if (
                        self.motion is source_motion
                        and command_generation == self._command_generation
                    ):
                        self.motion = self._merge_motion(
                            source_motion, self.motion_frame, gen_frame, generated
                        )
                        self.motion_frame = 0
            except BaseException as exc:  # propagated on the control thread
                self._plan_error = exc
            finally:
                self._planning = False
                pending = self._pending_replan_state
                self._pending_replan_state = None
                if pending is not None and not self._closed:
                    self.request_replan(pending, force=True)

        self._planner_thread = threading.Thread(
            target=worker, daemon=False, name="sonic-onnx-planner"
        )
        self._planner_thread.start()
        return True

    @staticmethod
    def _planner_context(motion: SonicMotion, gen_frame: int) -> np.ndarray:
        context = np.empty((4, 36), dtype=np.float32)
        for index in range(4):
            sample = gen_frame + index * 50.0 / 30.0
            low = min(int(math.floor(sample)), motion.frames - 1)
            high = min(low + 1, motion.frames - 1)
            weight = sample - math.floor(sample)
            context[index, :3] = (
                (1.0 - weight) * motion.root_positions[low]
                + weight * motion.root_positions[high]
            )
            q0, q1 = motion.root_quaternions[low], motion.root_quaternions[high]
            if np.dot(q0, q1) < 0.0:
                q1 = -q1
            context[index, 3:7] = _normalize_quaternion((1.0 - weight) * q0 + weight * q1)
            isaac_q = (
                (1.0 - weight) * motion.joint_positions[low]
                + weight * motion.joint_positions[high]
            )
            context[index, 7:] = isaac_q[ISAACLAB_TO_MUJOCO]
        return context

    @staticmethod
    def _merge_motion(
        old: SonicMotion, current_frame: int, gen_frame: int, new: SonicMotion
    ) -> SonicMotion:
        """Port SONIC's eight-frame rolling planner splice."""
        length = gen_frame - current_frame + new.frames
        if length <= 0:
            return old
        positions = np.empty((length, 29), dtype=np.float64)
        velocities = np.empty((length, 29), dtype=np.float64)
        roots = np.empty((length, 3), dtype=np.float64)
        quats = np.empty((length, 4), dtype=np.float64)
        blend_start = max(0, gen_frame - current_frame)
        for frame in range(length):
            old_frame = min(max(frame + current_frame, 0), old.frames - 1)
            new_frame = min(max(frame + current_frame - gen_frame, 0), new.frames - 1)
            weight = min(1.0, max(0.0, (frame - blend_start) / 8.0))
            positions[frame] = (1.0 - weight) * old.joint_positions[old_frame] + weight * new.joint_positions[new_frame]
            velocities[frame] = (1.0 - weight) * old.joint_velocities[old_frame] + weight * new.joint_velocities[new_frame]
            roots[frame] = (1.0 - weight) * old.root_positions[old_frame] + weight * new.root_positions[new_frame]
            q0, q1 = old.root_quaternions[old_frame], new.root_quaternions[new_frame]
            if np.dot(q0, q1) < 0.0:
                q1 = -q1
            quats[frame] = _normalize_quaternion((1.0 - weight) * q0 + weight * q1)
        return SonicMotion(positions, velocities, roots, quats)

    def close(self) -> None:
        self._closed = True
        thread = self._planner_thread
        if thread is not None and thread.is_alive():
            thread.join()

    def handle(self, key: str, state: SonicRobotState) -> str:
        message = self.command.handle(key)
        if "unsupported" not in message and "ignored" not in message:
            self._command_generation += 1
            self.planner_active = True
            self.request_replan(state, force=True)
        return message

    def _history_array(self, item: int) -> np.ndarray:
        values = list(self.history)
        if not values:
            raise RuntimeError("SONIC history is empty")
        while len(values) < 10:
            # StateLogger::GetLatest() left-pads an incomplete native history
            # with its default entry.  Its zero quaternion projects gravity to
            # +Z (not a zero vector), an odd but model-visible startup contract.
            padding = [np.zeros_like(value) for value in values[0]]
            padding[4] = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            values.insert(0, tuple(padding))
        return np.concatenate([entry[item] for entry in values[-10:]])

    def _motion_window(self, motion: SonicMotion, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.minimum(self.motion_frame + np.arange(10) * step, motion.frames - 1)
        return (
            motion.joint_positions[indices].reshape(-1),
            motion.joint_velocities[indices].reshape(-1),
            motion.root_quaternions[indices],
        )

    def _encoder_observation(self, state: SonicRobotState, motion: SonicMotion) -> np.ndarray:
        step = 1 if self.encoder_dimension == 1247 else 5
        positions, velocities, roots = self._motion_window(motion, step)
        orientations = np.concatenate([_orientation_6d(state.quaternion, value) for value in roots])
        if self.encoder_dimension == 1247:
            fields = (
                np.zeros(4), positions, velocities, orientations,
                np.zeros(6 + 120 + 120 + 9 + 12 + 288 + 24 + 24),
            )
        else:
            fields = (
                np.zeros(4), positions, velocities, np.zeros(10 + 1 + 6), orientations,
                np.zeros(120 + 120 + 9 + 12 + 720 + 60 + 60),
            )
        observation = np.concatenate(fields).astype(np.float32)
        if observation.size != self.encoder_dimension:
            raise RuntimeError(
                f"SONIC encoder observation has {observation.size} values; expected {self.encoder_dimension}"
            )
        return observation[None]

    def infer(self, state: SonicRobotState) -> np.ndarray:
        self.sim_time = float(state.tick) * 0.001
        if self._plan_error is not None:
            error, self._plan_error = self._plan_error, None
            raise RuntimeError("asynchronous SONIC planner failed") from error
        with self._motion_lock:
            motion = self.motion
            frame = self.motion_frame
        if motion is None:
            raise RuntimeError("SONIC planner has not been initialized")
        body_q = state.q[MUJOCO_TO_ISAACLAB] - DEFAULT_Q[MUJOCO_TO_ISAACLAB]
        body_dq = state.dq[MUJOCO_TO_ISAACLAB]
        gravity = _projected_gravity(state.quaternion)
        self.history.append((state.gyroscope.copy(), body_q, body_dq, self.last_action.copy(), gravity))
        token = self.encoder.infer({"obs_dict": self._encoder_observation(state, motion)})[0].reshape(64)
        policy_observation = np.concatenate(
            (token, self._history_array(0), self._history_array(1), self._history_array(2),
             self._history_array(3), self._history_array(4))
        ).astype(np.float32)
        if policy_observation.size != 994:
            raise RuntimeError(f"SONIC policy observation has {policy_observation.size} values; expected 994")
        self.last_policy_observation = policy_observation.copy()
        action = self.decoder.infer({"obs_dict": policy_observation[None]})[0].reshape(29).astype(np.float64)
        self.last_action = action
        with self._motion_lock:
            if self.motion is motion:
                self.motion_frame = min(frame + 1, motion.frames - 1)
        if self.planner_active:
            self.request_replan(state)
        target = DEFAULT_Q + ACTION_SCALE * action[ISAACLAB_TO_MUJOCO]
        # Match the validated native workflow's --lower-body-stance-any-stand-mode.
        if np.linalg.norm(self.command.movement[:2]) <= 0.05:
            stance = np.asarray(
                (-0.136089448, -0.007990774, 0.012218388, 0.282749737, -0.164737643,
                 0.005382167, -0.128899470, 0.015063731, 0.032335667, 0.288939655,
                 -0.177327879, -0.002457094, -0.013470759, -0.003519845, 0.002550134),
                dtype=np.float64,
            )
            delta = np.clip(stance - target[:15], -0.35, 0.35)
            target[:15] += 0.35 * delta
        return target
