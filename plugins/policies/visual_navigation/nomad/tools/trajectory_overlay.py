"""Diagnostic rendering for policy-native NoMaD trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from PIL import Image, ImageDraw, ImageFont

from longship.navigation.ports.trajectory_policy import (
    TrajectoryCandidateSet,
)
from tools.trajectory_stitching import (
    StitchedPose,
    TrajectoryStitchUpdate,
)


_CANDIDATE_COLORS = (
    (255, 219, 77),
    (255, 105, 180),
    (80, 220, 255),
    (118, 255, 128),
    (255, 150, 60),
    (190, 130, 255),
)


@dataclass(frozen=True, slots=True)
class TrajectoryOverlayState:
    source_timestamp_s: float
    phase: str
    current_node: str | None
    target_node: str | None
    status_detail: str
    candidate_set: TrajectoryCandidateSet | None = None
    goal_image: Image.Image | None = None
    model_crop_aspect: float | None = None
    stitched_path: tuple[StitchedPose, ...] = ()
    stitch_update: TrajectoryStitchUpdate | None = None
    stitch_panel_title: str = "Stitched path (diagnostic)"
    stitch_panel_detail: str = "fit view"


@dataclass(frozen=True, slots=True)
class TrajectoryPlotRange:
    reverse_extent: float = 4.0
    forward_extent: float = 20.0
    lateral_extent: float = 10.0

    def validate(self) -> None:
        values = (
            self.reverse_extent,
            self.forward_extent,
            self.lateral_extent,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("trajectory plot extents must be finite and positive")


def draw_trajectory_overlay(
    frame: Image.Image,
    state: TrajectoryOverlayState,
    *,
    plot_range: TrajectoryPlotRange = TrajectoryPlotRange(),
) -> Image.Image:
    """Returns an RGB frame with status, goal, and robot-frame trajectories."""

    plot_range.validate()
    output = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(output, "RGBA")
    status_font = _load_font(max(17, round(output.height / 36)))
    small_font = _load_font(max(14, round(output.height / 48)))
    _draw_model_crop(draw, small_font, output.size, state.model_crop_aspect)
    _draw_status(draw, status_font, output.size, state)
    _draw_goal(draw, small_font, output, state)
    _draw_robot_frame_plot(
        draw,
        small_font,
        output.size,
        state,
        plot_range,
    )
    _draw_stitched_path(draw, small_font, output.size, state)
    return output


def _draw_model_crop(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    size: tuple[int, int],
    aspect: float | None,
) -> None:
    if aspect is None:
        return
    width, height = size
    source_aspect = width / height
    if source_aspect > aspect:
        crop_width = min(width, max(1, int(height * aspect)))
        left = (width - crop_width) // 2
        right = left + crop_width
        draw.rectangle((0, 0, left, height), fill=(0, 0, 0, 85))
        draw.rectangle((right, 0, width, height), fill=(0, 0, 0, 85))
        draw.line((left, 0, left, height), fill=(80, 220, 255, 220), width=2)
        draw.line(
            (right, 0, right, height),
            fill=(80, 220, 255, 220),
            width=2,
        )
        draw.text(
            (left + crop_width // 2 - 95, height - 28),
            f"model center crop {aspect:.3f}:1",
            fill=(80, 220, 255, 255),
            font=font,
        )


def _draw_status(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    size: tuple[int, int],
    state: TrajectoryOverlayState,
) -> None:
    width, _ = size
    left = 18
    top = 16
    right = min(width - 18, left + 700)
    bottom = top + 104
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=9,
        fill=(0, 0, 0, 178),
        outline=(255, 255, 255, 90),
        width=1,
    )
    route = f"{state.current_node or '--'} -> {state.target_node or '--'}"
    lines = (
        f"t={state.source_timestamp_s:7.2f}s  {state.phase.upper()}",
        f"route: {route}",
        state.status_detail,
    )
    for index, line in enumerate(lines):
        draw.text(
            (left + 12, top + 10 + index * 27),
            line,
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )


def _draw_goal(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    output: Image.Image,
    state: TrajectoryOverlayState,
) -> None:
    if state.goal_image is None or state.target_node is None:
        return
    width, _ = output.size
    box_width = min(300, max(180, width // 4))
    goal_width, goal_height = state.goal_image.size
    box_height = min(
        round(box_width * 3 / 4),
        max(1, round(box_width * goal_height / goal_width)),
    )
    right = width - 18
    left = right - box_width
    top = 16
    bottom = top + box_height
    goal = state.goal_image.convert("RGB").resize(
        (box_width, box_height),
        Image.Resampling.BILINEAR,
    )
    output.paste(goal, (left, top))
    draw.rectangle(
        (left, top, right, bottom),
        outline=(255, 255, 255, 230),
        width=2,
    )
    label = f"Map goal: {state.target_node}"
    label_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    label_height = label_box[3] - label_box[1] + 10
    draw.rectangle(
        (left, top, right, top + label_height),
        fill=(0, 0, 0, 175),
    )
    draw.text(
        (left + 7, top + 5),
        label,
        fill=(255, 255, 255, 255),
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0, 255),
    )


def _draw_robot_frame_plot(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    size: tuple[int, int],
    state: TrajectoryOverlayState,
    plot_range: TrajectoryPlotRange,
) -> None:
    width, height = size
    panel_width = min(340, max(240, width // 4))
    panel_height = min(360, max(260, height // 2))
    left = 18
    right = left + panel_width
    bottom = height - 18
    top = bottom - panel_height
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=10,
        fill=(0, 0, 0, 185),
        outline=(255, 255, 255, 100),
        width=1,
    )
    draw.text(
        (left + 10, top + 8),
        "NoMaD robot-frame candidates (raw)",
        fill=(255, 255, 255, 255),
        font=font,
    )
    range_label = (
        f"fixed view: x [-{plot_range.reverse_extent:g}, "
        f"{plot_range.forward_extent:g}], "
        f"y +/-{plot_range.lateral_extent:g}"
    )
    draw.text(
        (left + 10, top + 26),
        range_label,
        fill=(210, 210, 210, 235),
        font=font,
    )

    plot_left = left + 24
    plot_right = right - 24
    plot_top = top + 56
    plot_bottom = bottom - 32
    origin = _project_point(
        0.0,
        0.0,
        (plot_left, plot_top, plot_right, plot_bottom),
        plot_range,
    )
    draw.line(
        (plot_left, origin[1], plot_right, origin[1]),
        fill=(255, 255, 255, 90),
        width=1,
    )
    draw.line(
        (origin[0], plot_top, origin[0], plot_bottom),
        fill=(255, 255, 255, 90),
        width=1,
    )
    draw.text(
        (plot_left, bottom - 23),
        "+y left",
        fill=(220, 220, 220, 220),
        font=font,
    )
    draw.text(
        (right - 72, bottom - 23),
        "x forward",
        fill=(220, 220, 220, 220),
        font=font,
    )

    candidate_set = state.candidate_set
    if candidate_set is None:
        draw.text(
            (plot_left + 8, plot_top + 12),
            "No fresh route-step trajectory",
            fill=(255, 205, 80, 255),
            font=font,
        )
        return

    for candidate_index, candidate in enumerate(candidate_set.candidates):
        color = _CANDIDATE_COLORS[
            candidate_index % len(_CANDIDATE_COLORS)
        ]
        points = [origin]
        points.extend(
            _project_point(
                waypoint.x,
                waypoint.y,
                (plot_left, plot_top, plot_right, plot_bottom),
                plot_range,
            )
            for waypoint in candidate.waypoints
        )
        draw.line(points, fill=(*color, 245), width=4, joint="curve")
        for point in points[1:]:
            radius = 3
            draw.ellipse(
                (
                    point[0] - radius,
                    point[1] - radius,
                    point[0] + radius,
                    point[1] + radius,
                ),
                fill=(*color, 255),
            )
    if state.stitch_update is not None:
        representative_points = [origin]
        representative_points.extend(
            _project_point(
                point.x,
                point.y,
                (plot_left, plot_top, plot_right, plot_bottom),
                plot_range,
            )
            for point in state.stitch_update.representative_path
        )
        draw.line(
            representative_points,
            fill=(255, 255, 255, 230),
            width=2,
            joint="curve",
        )
        highlighted = (
            state.stitch_update.control_waypoint
            if state.stitch_update.control_waypoint is not None
            else state.stitch_update.local_step
        )
        step = _project_point(
            highlighted.x,
            highlighted.y,
            (plot_left, plot_top, plot_right, plot_bottom),
            plot_range,
        )
        draw.line((origin, step), fill=(255, 80, 70, 255), width=6)
        draw.ellipse(
            (step[0] - 5, step[1] - 5, step[0] + 5, step[1] + 5),
            fill=(255, 80, 70, 255),
            outline=(255, 255, 255, 255),
            width=1,
        )
        if state.stitch_update.control_waypoint is not None:
            selection = (
                f"selected sample-{state.stitch_update.selected_candidate_index} "
                f"/ waypoint-{state.stitch_update.selected_waypoint_index}"
            )
            draw.text(
                (plot_left + 4, plot_top + 4),
                selection,
                fill=(255, 80, 70, 255),
                font=font,
            )
    draw.ellipse(
        (origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5),
        fill=(255, 255, 255, 255),
        outline=(0, 0, 0, 255),
        width=1,
    )


def _draw_stitched_path(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    size: tuple[int, int],
    state: TrajectoryOverlayState,
) -> None:
    if not state.stitched_path:
        return
    width, height = size
    panel_width = min(360, max(260, width // 4))
    panel_height = min(360, max(260, height // 2))
    left = max(18, (width - panel_width) // 2)
    right = left + panel_width
    bottom = height - 18
    top = bottom - panel_height
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=10,
        fill=(0, 0, 0, 185),
        outline=(255, 255, 255, 100),
        width=1,
    )
    draw.text(
        (left + 10, top + 8),
        state.stitch_panel_title,
        fill=(255, 255, 255, 255),
        font=font,
    )
    draw.text(
        (left + 10, top + 27),
        f"{state.stitch_panel_detail}; fit view",
        fill=(210, 210, 210, 235),
        font=font,
    )

    bounds = (left + 24, top + 56, right - 24, bottom - 28)
    points = _project_stitched_path(state.stitched_path, bounds)
    if len(points) >= 2:
        draw.line(points, fill=(70, 225, 255, 245), width=4, joint="curve")
    start = points[0]
    draw.ellipse(
        (start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5),
        fill=(80, 255, 120, 255),
        outline=(255, 255, 255, 255),
        width=1,
    )
    current = points[-1]
    draw.ellipse(
        (
            current[0] - 6,
            current[1] - 6,
            current[0] + 6,
            current[1] + 6,
        ),
        fill=(255, 80, 70, 255),
        outline=(255, 255, 255, 255),
        width=2,
    )
    draw.text(
        (left + 10, bottom - 22),
        "start=green  current=red  x-forward starts up",
        fill=(220, 220, 220, 220),
        font=font,
    )


def _project_stitched_path(
    path: tuple[StitchedPose, ...],
    bounds: tuple[int, int, int, int],
) -> tuple[tuple[int, int], ...]:
    left, top, right, bottom = bounds
    display_x = tuple(-pose.y for pose in path)
    display_y = tuple(-pose.x for pose in path)
    min_x, max_x = min(display_x), max(display_x)
    min_y, max_y = min(display_y), max(display_y)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((right - left) / span_x, (bottom - top) / span_y) * 0.9
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    pixel_center_x = (left + right) / 2.0
    pixel_center_y = (top + bottom) / 2.0
    return tuple(
        (
            round(pixel_center_x + (x - center_x) * scale),
            round(pixel_center_y + (y - center_y) * scale),
        )
        for x, y in zip(display_x, display_y, strict=True)
    )


def _project_point(
    forward_x: float,
    lateral_y: float,
    bounds: tuple[int, int, int, int],
    plot_range: TrajectoryPlotRange,
) -> tuple[int, int]:
    left, top, right, bottom = bounds
    lateral = max(
        -plot_range.lateral_extent,
        min(plot_range.lateral_extent, lateral_y),
    )
    forward = max(
        -plot_range.reverse_extent,
        min(plot_range.forward_extent, forward_x),
    )
    horizontal_ratio = (
        plot_range.lateral_extent - lateral
    ) / (2.0 * plot_range.lateral_extent)
    vertical_ratio = (
        plot_range.forward_extent - forward
    ) / (plot_range.forward_extent + plot_range.reverse_extent)
    return (
        round(left + horizontal_ratio * (right - left)),
        round(top + vertical_ratio * (bottom - top)),
    )


@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()
