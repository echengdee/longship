"""Shared policy-side initialization and velocity-command state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time

import numpy as np


class ControlMode(str, Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    ENABLED = "enabled"
    WALKING = "walking"


@dataclass(slots=True)
class PolicyControl:
    default_q: np.ndarray
    init_duration: float = 1.0
    linear_step: float = 0.1
    angular_step: float = 0.1
    require_walk_enable: bool = False
    smooth_velocity: bool = False
    linear_acceleration: float = 0.2
    angular_acceleration: float = 0.5
    mode: ControlMode = ControlMode.IDLE
    lin_x: float = 0.0
    lin_y: float = 0.0
    yaw: float = 0.0
    pending_enable: bool = False
    _init_started: float = 0.0
    _init_q: np.ndarray | None = None
    _target_lin_x: float = 0.0
    _target_lin_y: float = 0.0
    _target_yaw: float = 0.0
    _velocity_updated: float | None = None

    @property
    def policy_enabled(self) -> bool:
        return self.mode in (ControlMode.ENABLED, ControlMode.WALKING)

    @property
    def motion_enabled(self) -> bool:
        return self.mode is ControlMode.WALKING or (
            self.mode is ControlMode.ENABLED and not self.require_walk_enable
        )

    def handle(self, key: str, current_q: np.ndarray, *, lateral: bool, backward: bool) -> str:
        if key == "i":
            if self.mode is ControlMode.INITIALIZING:
                remaining = max(0.0, self.init_duration - (time.monotonic() - self._init_started))
                return f"already initializing; {remaining:.1f}s remaining"
            self.mode = ControlMode.INITIALIZING
            self.pending_enable = False
            self._init_started = time.monotonic()
            self._init_q = np.asarray(current_q, dtype=np.float64).copy()
            self._zero_velocity()
            return f"initializing over {self.init_duration:.1f}s"
        if key == "]":
            if self.mode is ControlMode.INITIALIZING:
                self.pending_enable = True
                return "enable queued; policy will start when initialization completes"
            if self.mode is ControlMode.READY:
                self.mode = ControlMode.ENABLED
                if self.require_walk_enable:
                    return "policy enabled; press = to enter walk mode"
                return "policy enabled"
            return f"enable ignored while mode={self.mode.value}; press i and wait"
        if key == "=" and self.require_walk_enable:
            if self.mode is ControlMode.ENABLED:
                self.mode = ControlMode.WALKING
                self._zero_velocity()
                return "walk mode enabled"
            if self.mode is ControlMode.WALKING:
                self.mode = ControlMode.ENABLED
                self._zero_velocity()
                return "stand mode enabled"
            return f"walk enable ignored while mode={self.mode.value}; press i, then ]"
        if self.require_walk_enable and self.mode is ControlMode.ENABLED:
            return "motion ignored while mode=enabled; press = to enter walk mode"
        if not self.motion_enabled:
            return f"motion ignored while mode={self.mode.value}"
        if key == "w":
            self._target_lin_x = min(1.0, max(0.0, self._target_lin_x) + self.linear_step)
            self._target_lin_y = self._target_yaw = 0.0
        elif key == "s" and backward:
            self._target_lin_x = max(-1.0, min(0.0, self._target_lin_x) - self.linear_step)
            self._target_lin_y = self._target_yaw = 0.0
        elif key == "a" and lateral:
            self._target_lin_y = min(1.0, max(0.0, self._target_lin_y) + self.linear_step)
            self._target_lin_x = self._target_yaw = 0.0
        elif key == "d" and lateral:
            self._target_lin_y = max(-1.0, min(0.0, self._target_lin_y) - self.linear_step)
            self._target_lin_x = self._target_yaw = 0.0
        elif key == "q":
            self._target_yaw = min(1.0, max(0.0, self._target_yaw) + self.angular_step)
            self._target_lin_x = self._target_lin_y = 0.0
        elif key == "e":
            self._target_yaw = max(-1.0, min(0.0, self._target_yaw) - self.angular_step)
            self._target_lin_x = self._target_lin_y = 0.0
        else:
            return f"motion key {key!r} unsupported; ignored"
        if not self.smooth_velocity:
            self.lin_x = self._target_lin_x
            self.lin_y = self._target_lin_y
            self.yaw = self._target_yaw
        return (
            f"velocity target=({self._target_lin_x:+.2f}, "
            f"{self._target_lin_y:+.2f}, {self._target_yaw:+.2f})"
        )

    @staticmethod
    def _approach(current: float, target: float, maximum_delta: float) -> float:
        return current + float(np.clip(target - current, -maximum_delta, maximum_delta))

    def update_velocity(self, now: float | None = None) -> tuple[float, float, float]:
        """Advance the command toward its target without a step input to the actor."""
        timestamp = time.monotonic() if now is None else now
        if not self.smooth_velocity:
            return self.lin_x, self.lin_y, self.yaw
        if self._velocity_updated is None:
            self._velocity_updated = timestamp
            return self.lin_x, self.lin_y, self.yaw
        elapsed = max(0.0, timestamp - self._velocity_updated)
        self._velocity_updated = timestamp
        self.lin_x = self._approach(
            self.lin_x, self._target_lin_x, self.linear_acceleration * elapsed
        )
        self.lin_y = self._approach(
            self.lin_y, self._target_lin_y, self.linear_acceleration * elapsed
        )
        self.yaw = self._approach(
            self.yaw, self._target_yaw, self.angular_acceleration * elapsed
        )
        return self.lin_x, self.lin_y, self.yaw

    def _zero_velocity(self) -> None:
        self.lin_x = self.lin_y = self.yaw = 0.0
        self._target_lin_x = self._target_lin_y = self._target_yaw = 0.0
        self._velocity_updated = None

    def target(self, current_q: np.ndarray, now: float | None = None) -> np.ndarray | None:
        if self.mode is ControlMode.IDLE:
            return None
        if self.mode is ControlMode.INITIALIZING:
            elapsed = (time.monotonic() if now is None else now) - self._init_started
            ratio = np.clip(elapsed / self.init_duration, 0.0, 1.0)
            if ratio >= 1.0:
                if self.pending_enable:
                    self.mode = ControlMode.ENABLED
                    self.pending_enable = False
                    suffix = "; press = to enter walk mode" if self.require_walk_enable else ""
                    print(f"teleop: initialization complete; policy enabled{suffix}", flush=True)
                else:
                    self.mode = ControlMode.READY
                    print("teleop: initialization complete; press ] to enable policy", flush=True)
                return self.default_q.copy()
            return self._init_q * (1.0 - ratio) + self.default_q * ratio
        return self.default_q.copy()
