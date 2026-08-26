from __future__ import annotations

from .config import ControlSettings


class MotionGovernor:
    """Bounds acceleration and jerk for ordinary (non-emergency) commands."""

    def __init__(self, settings: ControlSettings) -> None:
        self.settings = settings
        self.forward_mps = 0.0
        self.yaw_rate_radps = 0.0
        self._forward_accel = 0.0
        self._yaw_accel = 0.0

    def apply(
        self, desired_forward_mps: float, desired_yaw_rate_radps: float, dt_s: float
    ) -> tuple[float, float]:
        if dt_s <= 0.0:
            return self.forward_mps, self.yaw_rate_radps
        forward, forward_accel = self._axis(
            current=self.forward_mps,
            current_accel=self._forward_accel,
            desired=desired_forward_mps,
            dt_s=dt_s,
            max_accel=self.settings.maximum_linear_accel_mps2,
            max_jerk=self.settings.maximum_linear_jerk_mps3,
        )
        yaw, yaw_accel = self._axis(
            current=self.yaw_rate_radps,
            current_accel=self._yaw_accel,
            desired=desired_yaw_rate_radps,
            dt_s=dt_s,
            max_accel=self.settings.maximum_yaw_accel_radps2,
            max_jerk=self.settings.maximum_yaw_jerk_radps3,
        )
        if forward < 0.0:
            forward = 0.0
            forward_accel = 0.0
        self.forward_mps = forward
        self.yaw_rate_radps = yaw
        self._forward_accel = forward_accel
        self._yaw_accel = yaw_accel
        return forward, yaw

    def emergency_zero(self) -> tuple[float, float]:
        self.forward_mps = 0.0
        self.yaw_rate_radps = 0.0
        self._forward_accel = 0.0
        self._yaw_accel = 0.0
        return 0.0, 0.0

    @staticmethod
    def _axis(
        *,
        current: float,
        current_accel: float,
        desired: float,
        dt_s: float,
        max_accel: float,
        max_jerk: float,
    ) -> tuple[float, float]:
        requested_accel = max(-max_accel, min(max_accel, (desired - current) / dt_s))
        accel_delta = max_jerk * dt_s
        acceleration = max(
            current_accel - accel_delta,
            min(current_accel + accel_delta, requested_accel),
        )
        updated = current + acceleration * dt_s
        if (desired - current) * (desired - updated) <= 0.0:
            updated = desired
            acceleration = (updated - current) / dt_s
        return updated, acceleration
