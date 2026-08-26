"""Provider-neutral perception primitives."""

from .rgbd import BoundingBox, RigidTransform, ShortTrackAssigner, TrackedBox

__all__ = ["BoundingBox", "RigidTransform", "ShortTrackAssigner", "TrackedBox"]
