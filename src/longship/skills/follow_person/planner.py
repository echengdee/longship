from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from longship.contracts.skills.follow_person import ObstaclePoint, PlanDecision

from .config import ControlSettings, PlannerSettings

GridCell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _SearchNode:
    cell: GridCell
    cost: float


class LocalFollowPlanner:
    """Build a fresh base-local route from every synchronized scene.

    The planner deliberately has no global pose or map. Its output is a short
    target-independent velocity suggestion that still passes through Runtime,
    the obstacle guard, command arbitration, and the target adapter.
    """

    _NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    )

    def __init__(
        self, settings: PlannerSettings, control: ControlSettings
    ) -> None:
        self.settings = settings
        self.control = control
        self._forward_cells = int(
            settings.forward_extent_m / settings.grid_resolution_m
        )
        self._side_cells = int(settings.side_extent_m / settings.grid_resolution_m)

    def plan(
        self,
        target_robot_xy_m: tuple[float, float],
        obstacles: tuple[ObstaclePoint, ...],
        *,
        standoff_m: float,
        goal_tolerance_m: float,
    ) -> PlanDecision:
        target_forward, target_left = target_robot_xy_m
        target_distance = math.hypot(target_forward, target_left)
        remaining = target_distance - standoff_m
        if remaining <= goal_tolerance_m:
            return PlanDecision(0.0, 0.0, ((0.0, 0.0),), True, False, "at standoff")

        if target_distance <= 1e-9:
            return PlanDecision(0.0, 0.0, (), True, False, "target is at origin")

        goal_scale = max(0.0, remaining) / target_distance
        goal_xy = (target_forward * goal_scale, target_left * goal_scale)
        if goal_xy[0] < 0.0:
            heading = math.atan2(target_left, target_forward)
            yaw = self._clamp(
                self.control.heading_gain * heading,
                self.control.maximum_yaw_rate_radps,
            )
            return PlanDecision(
                0.0,
                yaw,
                ((0.0, 0.0),),
                False,
                False,
                "target is outside the forward planning field",
            )

        occupied = self._occupied_cells(obstacles, target_robot_xy_m)
        start = (0, 0)
        occupied.discard(start)
        requested_goal = self._to_cell(goal_xy)
        goal = self._nearest_free_goal(requested_goal, occupied)
        if goal is None:
            return PlanDecision(0.0, 0.0, (), False, True, "goal region is occupied")

        cells = self._search(start, goal, occupied)
        if not cells:
            return PlanDecision(0.0, 0.0, (), False, True, "no local route")
        path = tuple(self._to_xy(cell) for cell in cells)
        aim = path[-1]
        for point in path[1:]:
            aim = point
            if math.hypot(*point) >= self.settings.lookahead_distance_m:
                break
        heading = math.atan2(aim[1], max(1e-9, aim[0]))
        yaw_rate = self._clamp(
            self.control.heading_gain * heading,
            self.control.maximum_yaw_rate_radps,
        )
        forward = self._clamp(
            self.control.distance_gain * max(0.0, remaining),
            self.control.maximum_forward_speed_mps,
        )
        if abs(heading) >= self.control.forward_disable_angle_rad:
            forward = 0.0
        else:
            forward *= max(0.0, math.cos(heading))
        if target_distance < self.control.minimum_distance_m:
            forward = 0.0
        return PlanDecision(forward, yaw_rate, path, False, False, "local route ready")

    def _occupied_cells(
        self,
        obstacles: tuple[ObstaclePoint, ...],
        target_xy: tuple[float, float],
    ) -> set[GridCell]:
        occupied: set[GridCell] = set()
        resolution = self.settings.grid_resolution_m
        for obstacle in obstacles:
            if math.dist(
                (obstacle.forward_m, obstacle.left_m), target_xy
            ) <= self.settings.target_exclusion_radius_m:
                continue
            inflated = (
                obstacle.radius_m
                + self.settings.robot_radius_m
                + self.settings.clearance_margin_m
            )
            centre = self._to_cell((obstacle.forward_m, obstacle.left_m))
            radius_cells = math.ceil(inflated / resolution)
            for forward_index in range(
                centre[0] - radius_cells, centre[0] + radius_cells + 1
            ):
                for side_index in range(
                    centre[1] - radius_cells, centre[1] + radius_cells + 1
                ):
                    cell = (forward_index, side_index)
                    if not self._in_bounds(cell):
                        continue
                    point = self._to_xy(cell)
                    if math.dist(
                        point, (obstacle.forward_m, obstacle.left_m)
                    ) <= inflated:
                        occupied.add(cell)
        return occupied

    def _nearest_free_goal(
        self, requested: GridCell, occupied: set[GridCell]
    ) -> GridCell | None:
        if self._in_bounds(requested) and requested not in occupied:
            return requested
        maximum_ring = max(2, math.ceil(0.75 / self.settings.grid_resolution_m))
        candidates: list[tuple[float, GridCell]] = []
        for radius in range(1, maximum_ring + 1):
            for dx in range(-radius, radius + 1):
                for dy in (-radius, radius):
                    cell = (requested[0] + dx, requested[1] + dy)
                    if self._in_bounds(cell) and cell not in occupied:
                        candidates.append((math.hypot(dx, dy), cell))
            for dy in range(-radius + 1, radius):
                for dx in (-radius, radius):
                    cell = (requested[0] + dx, requested[1] + dy)
                    if self._in_bounds(cell) and cell not in occupied:
                        candidates.append((math.hypot(dx, dy), cell))
            if candidates:
                return min(candidates, key=lambda item: item[0])[1]
        return None

    def _search(
        self, start: GridCell, goal: GridCell, occupied: set[GridCell]
    ) -> tuple[GridCell, ...]:
        queue: list[tuple[float, int, GridCell]] = []
        serial = 0
        heapq.heappush(queue, (self._heuristic(start, goal), serial, start))
        parents: dict[GridCell, GridCell] = {}
        costs = {start: 0.0}
        while queue:
            _, _, current = heapq.heappop(queue)
            if current == goal:
                return self._reconstruct(parents, current)
            for dx, dy, step_cost in self._NEIGHBOURS:
                neighbour = (current[0] + dx, current[1] + dy)
                if not self._in_bounds(neighbour) or neighbour in occupied:
                    continue
                if dx and dy:
                    if (
                        (current[0] + dx, current[1]) in occupied
                        or (current[0], current[1] + dy) in occupied
                    ):
                        continue
                candidate = costs[current] + step_cost
                if candidate >= costs.get(neighbour, math.inf):
                    continue
                costs[neighbour] = candidate
                parents[neighbour] = current
                serial += 1
                priority = candidate + self._heuristic(neighbour, goal)
                heapq.heappush(queue, (priority, serial, neighbour))
        return ()

    @staticmethod
    def _reconstruct(
        parents: dict[GridCell, GridCell], current: GridCell
    ) -> tuple[GridCell, ...]:
        reversed_path = [current]
        while current in parents:
            current = parents[current]
            reversed_path.append(current)
        return tuple(reversed(reversed_path))

    @staticmethod
    def _heuristic(first: GridCell, second: GridCell) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def _to_cell(self, point: tuple[float, float]) -> GridCell:
        resolution = self.settings.grid_resolution_m
        return (round(point[0] / resolution), round(point[1] / resolution))

    def _to_xy(self, cell: GridCell) -> tuple[float, float]:
        resolution = self.settings.grid_resolution_m
        return (cell[0] * resolution, cell[1] * resolution)

    def _in_bounds(self, cell: GridCell) -> bool:
        return (
            0 <= cell[0] <= self._forward_cells
            and -self._side_cells <= cell[1] <= self._side_cells
        )

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))
