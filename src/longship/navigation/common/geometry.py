"""Geometry primitives shared by navigation contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pose3D:
    """A rigid pose expressed in an explicitly named reference frame."""

    frame_id: str
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
