"""Longship adapters for NoMaD-produced artifacts.

This package is intentionally separate from ``nomad_runtime``. It may depend
on Longship contracts, while the model runtime remains framework-neutral.
"""

from .topomap import (
    NomadTopomapMapConfig,
    create_nomad_topomap_engine,
    load_nomad_topomap,
)
from .trajectory_policy import (
    NomadTrajectoryPolicyConfig,
    NomadTrajectorySessionPort,
    NomadVisualGoalTrajectoryPolicy,
    VisualTargetGoalBinding,
    resolve_visual_target_goal,
)
from .image_resource import (
    DecodedImage,
    GoalImageLoader,
    LocalFileGoalImageLoader,
)
from .observation import (
    DecodedObservationFrame,
    DecodedObservationSource,
    NomadObservationSink,
    NomadObservationFanout,
    NomadObservationProducer,
    NomadObservationProducerConfig,
    NomadObservationProducerState,
    NomadObservationProducerStatus,
)
from .visual_policy import (
    NomadDistanceSessionPort,
    NomadVisualGoalDistancePolicy,
    NomadVisualPolicyConfig,
)

__all__ = [
    "NomadTopomapMapConfig",
    "NomadTrajectoryPolicyConfig",
    "NomadTrajectorySessionPort",
    "DecodedImage",
    "GoalImageLoader",
    "LocalFileGoalImageLoader",
    "DecodedObservationFrame",
    "DecodedObservationSource",
    "NomadDistanceSessionPort",
    "NomadObservationSink",
    "NomadObservationFanout",
    "NomadObservationProducer",
    "NomadObservationProducerConfig",
    "NomadObservationProducerState",
    "NomadObservationProducerStatus",
    "NomadVisualGoalDistancePolicy",
    "NomadVisualGoalTrajectoryPolicy",
    "NomadVisualPolicyConfig",
    "create_nomad_topomap_engine",
    "load_nomad_topomap",
    "resolve_visual_target_goal",
    "VisualTargetGoalBinding",
]
