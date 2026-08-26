from __future__ import annotations

from dataclasses import dataclass, fields

from longship.contracts.skills.follow_person import (
    PlanDecision,
    SafetyDecision,
    finite_number,
)


@dataclass(frozen=True, slots=True)
class SafetySettings:
    emergency_stop_distance_m: float
    nominal_stop_distance_m: float
    slow_distance_m: float
    maximum_deceleration_mps2: float
    system_latency_s: float
    distance_margin_m: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not finite_number(value) or value <= 0.0:
                raise ValueError(f"{field.name} must be positive")
        if not (
            self.emergency_stop_distance_m
            <= self.nominal_stop_distance_m
            < self.slow_distance_m
        ):
            raise ValueError("obstacle distances must increase from emergency to slow")


class ForwardObstacleGuard:
    """Independent last-mile limiter over raw forward-corridor clearance."""

    def __init__(self, settings: SafetySettings) -> None:
        self.settings = settings

    def apply(
        self,
        plan: PlanDecision,
        *,
        raw_clearance_m: float | None,
        current_forward_mps: float,
    ) -> SafetyDecision:
        if raw_clearance_m is None or not finite_number(raw_clearance_m):
            return SafetyDecision(0.0, 0.0, True, "raw obstacle clearance unavailable")
        speed = max(0.0, current_forward_mps)
        braking = speed * speed / (2.0 * self.settings.maximum_deceleration_mps2)
        latency = speed * self.settings.system_latency_s
        dynamic_stop = max(
            self.settings.nominal_stop_distance_m,
            braking + latency + self.settings.distance_margin_m,
        )
        stop_threshold = max(self.settings.emergency_stop_distance_m, dynamic_stop)
        if raw_clearance_m <= stop_threshold:
            return SafetyDecision(
                0.0,
                0.0,
                True,
                f"obstacle at {raw_clearance_m:.2f} m; "
                f"stop threshold {stop_threshold:.2f} m",
            )
        if plan.forward_mps <= 0.0:
            return SafetyDecision(
                plan.forward_mps, plan.yaw_rate_radps, False, plan.detail
            )
        if raw_clearance_m >= self.settings.slow_distance_m:
            return SafetyDecision(
                plan.forward_mps, plan.yaw_rate_radps, False, plan.detail
            )
        span = self.settings.slow_distance_m - stop_threshold
        scale = 0.0 if span <= 0.0 else (raw_clearance_m - stop_threshold) / span
        return SafetyDecision(
            plan.forward_mps * max(0.0, min(1.0, scale)),
            plan.yaw_rate_radps,
            False,
            f"speed limited by obstacle at {raw_clearance_m:.2f} m",
        )
