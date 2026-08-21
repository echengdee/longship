"""Fixed-start visual topological Localization Engine implementation."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from enum import Enum
import math

from longship.navigation.common import TimePoint
from longship.navigation.map_engine.interface import MapEngine
from longship.navigation.map_engine.models import (
    AnchorDescriptor,
    AnchorKind,
    AnchorPurpose,
    AnchorQuery,
    MapEntityKind,
    MapSnapshot,
    NodeId,
    ResourceDescriptor,
    ResourceKind,
    SegmentDescriptor,
    TopologyNode,
    TopologyQuery,
)

from .interface import LocalizationEngineError
from .models import (
    BeliefRevision,
    BeliefStreamId,
    BeliefUpdateOutcome,
    BeliefUpdateResult,
    HypothesisId,
    LocalizationCapability,
    LocalizationEngineState,
    LocalizationEngineStatus,
    LocalizationErrorCode,
    LocalizationSourceHealth,
    LocalizationSourceState,
    LocalizationStatus,
    LocationBelief,
    LocationHypothesis,
    NodeLocation,
    RelocalizationAcceptance,
    RelocalizationDisposition,
    RelocalizationRequest,
    WaitForUpdateRequest,
)
from .visual_policy import (
    VisualGoalCandidate,
    VisualGoalDistanceBatchMeasurement,
    VisualGoalDistanceBatchPolicy,
    VisualGoalDistanceBatchRequest,
    VisualPolicyError,
    VisualPolicyErrorCode,
)


FIXED_START_NODE_ID = NodeId("node-0000")


class FixedStartVisualPhase(str, Enum):
    WAIT_CONTEXT = "wait_context"
    VERIFY_START = "verify_start"
    SEARCHING_NEXT = "searching_next"
    TRACKING = "tracking"
    AT_FINAL_NODE = "at_final_node"
    LOCALIZATION_LOST = "localization_lost"
    FAULT = "fault"


@dataclass(frozen=True, slots=True)
class FixedStartVisualTrackingProfile:
    """Guards for monotonic local-candidate visual chain tracking."""

    image_profile_id: str
    close_threshold: float = 3.0
    start_close_confirmations: int = 2
    successor_close_confirmations: int = 1
    normal_distance_maximum: float = 15.0
    untrusted_distance_minimum: float = 18.0
    lost_confirmations: int = 3
    tracking_candidate_count: int = 3
    evidence_window_size: int = 3
    relative_advantage_minimum: float = 1.0
    relative_distance_maximum: float = 5.0
    relative_advance_confirmations: int = 2
    lookahead_close_confirmations: int = 2
    relocalization_candidate_count: int = 8
    relocalization_goal_backtrack_count: int = 2
    relocalization_close_confirmations: int = 2
    belief_publish_period_s: float = 0.25
    max_observation_age_s: float = 0.2
    source_id: str = "visual_goal_distance"

    def validate(self) -> None:
        """Raises ``ValueError`` for inconsistent tracking guards."""

        scalar_values = (
            self.close_threshold,
            self.normal_distance_maximum,
            self.untrusted_distance_minimum,
            self.relative_advantage_minimum,
            self.relative_distance_maximum,
            self.belief_publish_period_s,
            self.max_observation_age_s,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("tracking thresholds must be finite")
        if self.close_threshold < 0.0:
            raise ValueError("close_threshold must be non-negative")
        if self.normal_distance_maximum < self.close_threshold:
            raise ValueError(
                "normal_distance_maximum must not be below close_threshold"
            )
        if self.untrusted_distance_minimum < self.normal_distance_maximum:
            raise ValueError(
                "untrusted_distance_minimum must not be below the normal "
                "distance maximum"
            )
        if self.relative_advantage_minimum < 0.0:
            raise ValueError(
                "relative_advantage_minimum must be non-negative"
            )
        if not (
            self.close_threshold
            <= self.relative_distance_maximum
            <= self.normal_distance_maximum
        ):
            raise ValueError(
                "relative_distance_maximum must be between close and normal"
            )
        if self.belief_publish_period_s <= 0.0:
            raise ValueError("belief_publish_period_s must be positive")
        if (
            self.start_close_confirmations <= 0
            or self.successor_close_confirmations <= 0
            or self.lost_confirmations <= 0
            or self.tracking_candidate_count < 2
            or self.evidence_window_size <= 0
            or self.relative_advance_confirmations <= 0
            or self.lookahead_close_confirmations <= 0
            or self.relocalization_candidate_count
            < self.tracking_candidate_count
            or self.relocalization_goal_backtrack_count < 0
            or self.relocalization_goal_backtrack_count
            >= self.relocalization_candidate_count
            or self.relocalization_close_confirmations <= 0
        ):
            raise ValueError("tracking counts and candidate windows are invalid")
        if (
            self.successor_close_confirmations > self.evidence_window_size
            or self.relative_advance_confirmations
            > self.evidence_window_size
            or self.lookahead_close_confirmations
            > self.evidence_window_size
        ):
            raise ValueError(
                "advance confirmations must fit inside the evidence window"
            )
        if self.max_observation_age_s < 0.0:
            raise ValueError("max_observation_age_s must be non-negative")
        if not self.image_profile_id or not self.source_id:
            raise ValueError("tracking profile ids must not be empty")


@dataclass(frozen=True, slots=True)
class FixedStartVisualTrackingState:
    phase: FixedStartVisualPhase
    current_node_id: NodeId | None
    target_node_id: NodeId | None
    close_count: int
    relative_count: int
    lookahead_close_count: int
    far_count: int
    relocalization_count: int
    last_temporal_distance: float | None
    last_candidate_distances: tuple[tuple[NodeId, float], ...]
    last_observation_time: TimePoint | None


@dataclass(frozen=True, slots=True)
class _ForwardEvidence:
    absolute_close: bool
    relative_winner: bool
    later_candidate_close: bool


@dataclass(frozen=True, slots=True)
class _GoalBinding:
    node_id: NodeId
    anchor: AnchorDescriptor
    resource: ResourceDescriptor
    model_artifact_id: str
    model_artifact_digest: str


class FixedStartVisualLocalizationEngine:
    """Tracks position along one fully connected, directed visual chain.

    ``tick`` is an internal runtime method, not part of the public
    ``LocalizationEngine`` facade. A supervisor calls it at the configured
    policy rate while missions only consume the resulting belief stream.
    """

    def __init__(
        self,
        *,
        snapshot: MapSnapshot,
        policy: VisualGoalDistanceBatchPolicy,
        profile: FixedStartVisualTrackingProfile,
        bindings: tuple[_GoalBinding, ...],
        stream_id: BeliefStreamId,
        started_at: TimePoint,
    ) -> None:
        self._snapshot = snapshot
        self._policy = policy
        self._profile = profile
        self._bindings = bindings
        self._stream_id = stream_id
        self._phase = FixedStartVisualPhase.WAIT_CONTEXT
        self._current_index: int | None = None
        self._target_index: int | None = 0
        self._close_count = 0
        self._relative_count = 0
        self._lookahead_close_count = 0
        self._far_count = 0
        self._relocalization_index: int | None = None
        self._relocalization_count = 0
        self._evidence: deque[_ForwardEvidence] = deque(
            maxlen=profile.evidence_window_size
        )
        self._last_temporal_distance: float | None = None
        self._last_candidate_distances: tuple[
            tuple[NodeId, float], ...
        ] = ()
        self._last_observation_time: TimePoint | None = None
        self._sequence = 0
        self._engine_state = LocalizationEngineState.RUNNING
        self._detail_code = self._phase.value
        self._last_update_at = started_at
        self._condition = asyncio.Condition()
        self._tick_lock = asyncio.Lock()
        self._belief = LocationBelief(
            snapshot_id=snapshot.snapshot_id,
            revision=BeliefRevision(stream_id=stream_id, sequence=0),
            estimate_time=started_at,
            published_at=started_at,
            status=LocalizationStatus.INITIALIZING,
            confidence=None,
            source_health=(
                LocalizationSourceHealth(
                    source_id=profile.source_id,
                    state=LocalizationSourceState.UNAVAILABLE,
                    detail_code=FixedStartVisualPhase.WAIT_CONTEXT.value,
                ),
            ),
        )

    @classmethod
    async def create(
        cls,
        *,
        map_engine: MapEngine,
        snapshot: MapSnapshot,
        policy: VisualGoalDistanceBatchPolicy,
        profile: FixedStartVisualTrackingProfile,
        stream_id: BeliefStreamId,
        started_at: TimePoint,
    ) -> "FixedStartVisualLocalizationEngine":
        """Loads and validates the fixed directed chain from Map Engine."""

        profile.validate()
        bindings = await _load_goal_bindings(
            map_engine=map_engine,
            snapshot=snapshot,
            profile=profile,
        )
        return cls(
            snapshot=snapshot,
            policy=policy,
            profile=profile,
            bindings=bindings,
            stream_id=stream_id,
            started_at=started_at,
        )

    def get_belief(self) -> LocationBelief:
        return self._belief

    def get_status(self) -> LocalizationEngineStatus:
        return LocalizationEngineStatus(
            state=self._engine_state,
            snapshot_id=self._snapshot.snapshot_id,
            stream_id=self._stream_id,
            capabilities=frozenset(
                {
                    LocalizationCapability.TOPOLOGICAL_LOCATION,
                    LocalizationCapability.FIXED_START_TOPOLOGICAL_TRACKING,
                    LocalizationCapability.SOURCE_HEALTH,
                }
            ),
            latest_sequence=self._sequence,
            last_update_at=self._last_update_at,
            detail_code=self._detail_code,
        )

    def get_tracking_state(self) -> FixedStartVisualTrackingState:
        current = (
            None
            if self._current_index is None
            else self._bindings[self._current_index].node_id
        )
        target = (
            None
            if self._target_index is None
            else self._bindings[self._target_index].node_id
        )
        return FixedStartVisualTrackingState(
            phase=self._phase,
            current_node_id=current,
            target_node_id=target,
            close_count=self._close_count,
            relative_count=self._relative_count,
            lookahead_close_count=self._lookahead_close_count,
            far_count=self._far_count,
            relocalization_count=self._relocalization_count,
            last_temporal_distance=self._last_temporal_distance,
            last_candidate_distances=self._last_candidate_distances,
            last_observation_time=self._last_observation_time,
        )

    async def tick(self, now: TimePoint) -> LocationBelief:
        """Evaluates one local goal batch and atomically updates evidence."""

        async with self._tick_lock:
            if self._phase == FixedStartVisualPhase.FAULT:
                raise LocalizationEngineError(
                    LocalizationErrorCode.ENGINE_FAULTED,
                    "fixed-start visual localization is faulted",
                )
            if self._phase == FixedStartVisualPhase.AT_FINAL_NODE:
                return self._belief
            candidate_indices = self._candidate_indices()
            if not candidate_indices:
                return await self._fault(
                    now,
                    "missing_candidate_bindings",
                )

            first_binding = self._bindings[candidate_indices[0]]
            request = VisualGoalDistanceBatchRequest(
                snapshot_id=self._snapshot.snapshot_id,
                candidates=tuple(
                    VisualGoalCandidate(
                        target_node_id=self._bindings[index].node_id,
                        target_anchor_id=(
                            self._bindings[index].anchor.anchor_id
                        ),
                        goal_resource=self._bindings[index].resource,
                    )
                    for index in candidate_indices
                ),
                requested_at=now,
                max_observation_age_s=(self._profile.max_observation_age_s),
                expected_image_profile_id=self._profile.image_profile_id,
                expected_model_artifact_id=(
                    first_binding.model_artifact_id
                ),
                expected_model_artifact_digest=(
                    first_binding.model_artifact_digest
                ),
            )
            try:
                measurement = await self._policy.compare_goals(request)
                distances = self._validate_measurement(
                    request,
                    measurement,
                    candidate_indices,
                )
            except VisualPolicyError as error:
                return await self._handle_policy_error(now, error)
            except Exception as error:
                return await self._fault(
                    now,
                    f"policy_contract_failure:{type(error).__name__}",
                )

            self._last_observation_time = measurement.observation_time
            self._last_candidate_distances = tuple(
                (self._bindings[index].node_id, distances[index])
                for index in candidate_indices
            )
            if self._target_index is not None:
                self._last_temporal_distance = distances.get(
                    self._target_index
                )
            else:
                self._last_temporal_distance = min(distances.values())
            if self._phase == FixedStartVisualPhase.WAIT_CONTEXT:
                self._phase = FixedStartVisualPhase.VERIFY_START

            if self._phase == FixedStartVisualPhase.VERIFY_START:
                return await self._update_start_verification(
                    distances[0],
                    measurement.observation_time,
                    now,
                )
            if self._phase == FixedStartVisualPhase.LOCALIZATION_LOST:
                return await self._update_relocalization(
                    distances,
                    measurement.observation_time,
                    now,
                )
            return await self._update_topological_tracking(
                distances,
                measurement.observation_time,
                now,
            )

    def _candidate_indices(self) -> tuple[int, ...]:
        if self._current_index is None:
            start_index = 0
            candidate_count = self._profile.tracking_candidate_count
        elif self._phase == FixedStartVisualPhase.LOCALIZATION_LOST:
            if self._target_index is None:
                return ()
            start_index = max(
                0,
                self._target_index
                - self._profile.relocalization_goal_backtrack_count,
            )
            candidate_count = self._profile.relocalization_candidate_count
        else:
            start_index = self._current_index
            candidate_count = self._profile.tracking_candidate_count
        end_index = min(len(self._bindings), start_index + candidate_count)
        return tuple(range(start_index, end_index))

    async def wait_for_update(
        self,
        request: WaitForUpdateRequest,
    ) -> BeliefUpdateResult:
        if request.timeout_s is not None and request.timeout_s < 0.0:
            raise LocalizationEngineError(
                LocalizationErrorCode.INVALID_REQUEST,
                "wait timeout must be non-negative",
            )

        async with self._condition:
            immediate = self._update_after(request.after_revision)
            if immediate is not None:
                return immediate
            try:
                wait_for_update = self._condition.wait_for(
                    lambda: self._belief.revision != request.after_revision
                )
                if request.timeout_s is None:
                    await wait_for_update
                else:
                    await asyncio.wait_for(
                        wait_for_update,
                        timeout=request.timeout_s,
                    )
            except asyncio.TimeoutError:
                return BeliefUpdateResult(
                    outcome=BeliefUpdateOutcome.TIMED_OUT,
                    belief=self._belief,
                )
            updated = self._update_after(request.after_revision)
            if updated is not None:
                return updated
            return BeliefUpdateResult(
                outcome=BeliefUpdateOutcome.TIMED_OUT,
                belief=self._belief,
            )

    async def request_relocalization(
        self,
        request: RelocalizationRequest,
    ) -> RelocalizationAcceptance:
        del request
        return RelocalizationAcceptance(
            relocalization_id=None,
            disposition=RelocalizationDisposition.UNAVAILABLE,
            detail_code="fixed_start_global_relocalization_unavailable",
        )

    def _update_after(
        self,
        revision: BeliefRevision,
    ) -> BeliefUpdateResult | None:
        if revision.stream_id != self._stream_id:
            return BeliefUpdateResult(
                outcome=BeliefUpdateOutcome.STREAM_RESET,
                belief=self._belief,
            )
        if revision.sequence > self._sequence:
            raise LocalizationEngineError(
                LocalizationErrorCode.INVALID_REQUEST,
                "belief revision is ahead of the current stream",
            )
        if revision.sequence < self._sequence:
            return BeliefUpdateResult(
                outcome=BeliefUpdateOutcome.UPDATED,
                belief=self._belief,
            )
        return None

    def _validate_measurement(
        self,
        request: VisualGoalDistanceBatchRequest,
        measurement: VisualGoalDistanceBatchMeasurement,
        candidate_indices: tuple[int, ...],
    ) -> dict[int, float]:
        if measurement.snapshot_id != request.snapshot_id:
            raise ValueError("visual policy returned a mismatched snapshot")
        if len(measurement.candidate_distances) != len(request.candidates):
            raise ValueError("visual policy returned a mismatched batch size")
        distances = {}
        for index, candidate, result in zip(
            candidate_indices,
            request.candidates,
            measurement.candidate_distances,
            strict=True,
        ):
            identity_matches = (
                result.target_node_id == candidate.target_node_id
                and result.target_anchor_id == candidate.target_anchor_id
                and result.goal_resource_id
                == candidate.goal_resource.resource_id
            )
            if not identity_matches:
                raise ValueError(
                    "visual policy returned mismatched goal identity"
                )
            if not math.isfinite(result.temporal_distance):
                raise ValueError(
                    "visual policy returned a non-finite distance"
                )
            distances[index] = result.temporal_distance
        if (
            measurement.image_profile_id != request.expected_image_profile_id
            or measurement.model_artifact_id
            != request.expected_model_artifact_id
            or measurement.model_artifact_digest
            != request.expected_model_artifact_digest
        ):
            raise ValueError(
                "visual policy returned incompatible model or image profile"
            )
        if not measurement.policy_id:
            raise ValueError("visual policy returned an empty policy id")
        if (
            measurement.observation_time.clock_id
            != request.requested_at.clock_id
        ):
            raise ValueError("visual policy returned a different clock domain")
        if measurement.produced_at.clock_id != request.requested_at.clock_id:
            raise ValueError("visual policy returned a different output clock")
        age_ns = (
            request.requested_at.nanoseconds
            - measurement.observation_time.nanoseconds
        )
        maximum_age_ns = round(request.max_observation_age_s * 1_000_000_000)
        if age_ns < 0 or age_ns > maximum_age_ns:
            raise ValueError("visual policy returned a stale observation")
        return distances

    async def _update_start_verification(
        self,
        distance: float,
        observation_time: TimePoint,
        now: TimePoint,
    ) -> LocationBelief:
        if distance < self._profile.close_threshold:
            self._close_count += 1
        else:
            self._close_count = 0
        self._far_count = 0

        if self._close_count < self._profile.start_close_confirmations:
            return await self._publish_periodic(
                now=now,
                estimate_time=observation_time,
                status=LocalizationStatus.INITIALIZING,
                source_state=LocalizationSourceState.HEALTHY,
                detail_code="verifying_fixed_start",
                include_current_hypothesis=False,
            )

        self._current_index = 0
        self._reset_evidence()
        if len(self._bindings) == 1:
            self._target_index = None
            self._phase = FixedStartVisualPhase.AT_FINAL_NODE
            detail_code = "final_node_confirmed"
        else:
            self._target_index = 1
            self._phase = FixedStartVisualPhase.SEARCHING_NEXT
            detail_code = "fixed_start_verified"
        return await self._publish(
            now=now,
            estimate_time=observation_time,
            status=LocalizationStatus.TRACKING,
            source_state=LocalizationSourceState.HEALTHY,
            detail_code=detail_code,
            include_current_hypothesis=True,
        )

    async def _update_topological_tracking(
        self,
        distances: dict[int, float],
        observation_time: TimePoint,
        now: TimePoint,
    ) -> LocationBelief:
        """Accumulates local relative evidence and advances monotonically."""

        if self._current_index is None or self._target_index is None:
            return await self._fault(now, "missing_tracking_indices")
        current_index = self._current_index
        target_index = self._target_index
        if current_index not in distances or target_index not in distances:
            return await self._fault(now, "missing_tracking_candidates")

        best_index, best_distance = min(
            distances.items(),
            key=lambda item: (item[1], item[0]),
        )
        current_distance = distances[current_index]
        target_distance = distances[target_index]
        evidence = _ForwardEvidence(
            absolute_close=(
                target_distance < self._profile.close_threshold
            ),
            relative_winner=(
                best_index == target_index
                and target_distance
                <= self._profile.relative_distance_maximum
                and current_distance - target_distance
                >= self._profile.relative_advantage_minimum
            ),
            later_candidate_close=(
                best_index > target_index
                and best_distance < self._profile.close_threshold
            ),
        )
        self._evidence.append(evidence)
        self._close_count = sum(
            item.absolute_close for item in self._evidence
        )
        self._relative_count = sum(
            item.relative_winner for item in self._evidence
        )
        self._lookahead_close_count = sum(
            item.later_candidate_close for item in self._evidence
        )

        if (
            self._close_count
            >= self._profile.successor_close_confirmations
            or self._relative_count
            >= self._profile.relative_advance_confirmations
        ):
            return await self._advance_one_node(
                observation_time=observation_time,
                now=now,
            )

        if (
            self._lookahead_close_count
            >= self._profile.lookahead_close_confirmations
        ):
            self._phase = FixedStartVisualPhase.LOCALIZATION_LOST
            self._engine_state = LocalizationEngineState.DEGRADED
            self._far_count = 0
            self._relocalization_index = best_index
            self._relocalization_count = 1
            self._reset_evidence()
            return await self._publish(
                now=now,
                estimate_time=observation_time,
                status=LocalizationStatus.LOST,
                source_state=LocalizationSourceState.DEGRADED,
                detail_code="expected_successor_window_missed",
                include_current_hypothesis=False,
            )

        if best_distance >= self._profile.untrusted_distance_minimum:
            self._far_count += 1
        else:
            self._far_count = 0

        if self._far_count >= self._profile.lost_confirmations:
            self._phase = FixedStartVisualPhase.LOCALIZATION_LOST
            self._engine_state = LocalizationEngineState.DEGRADED
            self._reset_evidence()
            self._relocalization_index = None
            self._relocalization_count = 0
            return await self._publish(
                now=now,
                estimate_time=observation_time,
                status=LocalizationStatus.LOST,
                source_state=LocalizationSourceState.DEGRADED,
                detail_code="visual_target_persistently_untrusted",
                include_current_hypothesis=False,
            )

        if target_distance <= self._profile.normal_distance_maximum:
            self._phase = FixedStartVisualPhase.TRACKING
        else:
            self._phase = FixedStartVisualPhase.SEARCHING_NEXT

        if best_distance > self._profile.normal_distance_maximum:
            detail_code = (
                "visual_target_untrusted"
                if best_distance >= self._profile.untrusted_distance_minimum
                else "visual_candidates_weak"
            )
            return await self._publish_periodic(
                now=now,
                estimate_time=observation_time,
                status=LocalizationStatus.DEGRADED,
                source_state=LocalizationSourceState.DEGRADED,
                detail_code=detail_code,
                include_current_hypothesis=True,
            )

        detail_code = (
            "tracking_local_candidates"
            if self._phase == FixedStartVisualPhase.TRACKING
            else "searching_expected_successor"
        )
        return await self._publish_periodic(
            now=now,
            estimate_time=observation_time,
            status=LocalizationStatus.TRACKING,
            source_state=LocalizationSourceState.HEALTHY,
            detail_code=detail_code,
            include_current_hypothesis=True,
        )

    async def _advance_one_node(
        self,
        *,
        observation_time: TimePoint,
        now: TimePoint,
    ) -> LocationBelief:
        if self._target_index is None:
            return await self._fault(now, "missing_advance_target")
        self._current_index = self._target_index
        self._far_count = 0
        self._reset_evidence()
        if self._current_index == len(self._bindings) - 1:
            self._target_index = None
            self._phase = FixedStartVisualPhase.AT_FINAL_NODE
            detail_code = "final_node_confirmed"
        else:
            self._target_index = self._current_index + 1
            self._phase = FixedStartVisualPhase.SEARCHING_NEXT
            detail_code = "topology_node_advanced"
        return await self._publish(
            now=now,
            estimate_time=observation_time,
            status=LocalizationStatus.TRACKING,
            source_state=LocalizationSourceState.HEALTHY,
            detail_code=detail_code,
            include_current_hypothesis=True,
        )

    async def _update_relocalization(
        self,
        distances: dict[int, float],
        observation_time: TimePoint,
        now: TimePoint,
    ) -> LocationBelief:
        """Searches a goal-centered, forward-biased window after local loss."""

        best_index, best_distance = min(
            distances.items(),
            key=lambda item: (item[1], item[0]),
        )
        if best_distance < self._profile.close_threshold:
            if self._relocalization_index == best_index:
                self._relocalization_count += 1
            else:
                self._relocalization_index = best_index
                self._relocalization_count = 1
        else:
            self._relocalization_index = None
            self._relocalization_count = 0

        if (
            self._relocalization_count
            >= self._profile.relocalization_close_confirmations
        ):
            self._current_index = best_index
            self._engine_state = LocalizationEngineState.RUNNING
            self._far_count = 0
            self._relocalization_index = None
            self._relocalization_count = 0
            self._reset_evidence()
            if best_index == len(self._bindings) - 1:
                self._target_index = None
                self._phase = FixedStartVisualPhase.AT_FINAL_NODE
                detail_code = "final_node_relocalized"
            else:
                self._target_index = best_index + 1
                self._phase = FixedStartVisualPhase.SEARCHING_NEXT
                detail_code = "visual_relocalized"
            return await self._publish(
                now=now,
                estimate_time=observation_time,
                status=LocalizationStatus.TRACKING,
                source_state=LocalizationSourceState.HEALTHY,
                detail_code=detail_code,
                include_current_hypothesis=True,
            )

        return await self._publish_periodic(
            now=now,
            estimate_time=observation_time,
            status=LocalizationStatus.LOST,
            source_state=LocalizationSourceState.DEGRADED,
            detail_code="visual_relocalizing",
            include_current_hypothesis=False,
        )

    def _reset_evidence(self) -> None:
        self._evidence.clear()
        self._close_count = 0
        self._relative_count = 0
        self._lookahead_close_count = 0

    async def _handle_policy_error(
        self,
        now: TimePoint,
        error: VisualPolicyError,
    ) -> LocationBelief:
        if not error.retryable:
            return await self._fault(now, f"visual_policy:{error.code.value}")

        if self._current_index is None:
            source_state = (
                LocalizationSourceState.STALE
                if error.code == VisualPolicyErrorCode.CONTEXT_STALE
                else LocalizationSourceState.UNAVAILABLE
            )
            return await self._publish(
                now=now,
                estimate_time=now,
                status=LocalizationStatus.INITIALIZING,
                source_state=source_state,
                detail_code=error.code.value,
                include_current_hypothesis=False,
            )

        if self._phase == FixedStartVisualPhase.LOCALIZATION_LOST:
            return await self._publish_periodic(
                now=now,
                estimate_time=now,
                status=LocalizationStatus.LOST,
                source_state=LocalizationSourceState.UNAVAILABLE,
                detail_code=f"relocalization_{error.code.value}",
                include_current_hypothesis=False,
            )

        self._far_count += 1
        if self._far_count >= self._profile.lost_confirmations:
            self._phase = FixedStartVisualPhase.LOCALIZATION_LOST
            self._engine_state = LocalizationEngineState.DEGRADED
            self._reset_evidence()
            self._relocalization_index = None
            self._relocalization_count = 0
            return await self._publish(
                now=now,
                estimate_time=now,
                status=LocalizationStatus.LOST,
                source_state=LocalizationSourceState.UNAVAILABLE,
                detail_code=f"persistent_{error.code.value}",
                include_current_hypothesis=False,
            )
        source_state = (
            LocalizationSourceState.STALE
            if error.code == VisualPolicyErrorCode.CONTEXT_STALE
            else LocalizationSourceState.DEGRADED
        )
        return await self._publish(
            now=now,
            estimate_time=now,
            status=LocalizationStatus.DEGRADED,
            source_state=source_state,
            detail_code=error.code.value,
            include_current_hypothesis=True,
        )

    async def _publish_periodic(
        self,
        *,
        now: TimePoint,
        estimate_time: TimePoint,
        status: LocalizationStatus,
        source_state: LocalizationSourceState,
        detail_code: str,
        include_current_hypothesis: bool,
    ) -> LocationBelief:
        """Publishes state changes immediately and steady state near 4 Hz."""

        previous_source_state = self._belief.source_health[0].state
        state_changed = (
            status != self._belief.status
            or source_state != previous_source_state
        )
        elapsed_ns = now.nanoseconds - self._belief.published_at.nanoseconds
        publish_period_ns = round(
            self._profile.belief_publish_period_s * 1_000_000_000
        )
        if not state_changed and elapsed_ns < publish_period_ns:
            return self._belief
        return await self._publish(
            now=now,
            estimate_time=estimate_time,
            status=status,
            source_state=source_state,
            detail_code=detail_code,
            include_current_hypothesis=include_current_hypothesis,
        )

    async def _fault(self, now: TimePoint, detail_code: str) -> LocationBelief:
        self._phase = FixedStartVisualPhase.FAULT
        self._engine_state = LocalizationEngineState.FAULTED
        return await self._publish(
            now=now,
            estimate_time=now,
            status=LocalizationStatus.UNAVAILABLE,
            source_state=LocalizationSourceState.UNAVAILABLE,
            detail_code=detail_code,
            include_current_hypothesis=False,
        )

    async def _publish(
        self,
        *,
        now: TimePoint,
        estimate_time: TimePoint,
        status: LocalizationStatus,
        source_state: LocalizationSourceState,
        detail_code: str,
        include_current_hypothesis: bool,
    ) -> LocationBelief:
        self._sequence += 1
        hypotheses = ()
        if include_current_hypothesis and self._current_index is not None:
            node_id = self._bindings[self._current_index].node_id
            hypotheses = (
                LocationHypothesis(
                    hypothesis_id=HypothesisId(
                        f"{self._stream_id}:node:{node_id}"
                    ),
                    topological_location=NodeLocation(node_id=node_id),
                ),
            )
        self._detail_code = detail_code
        self._last_update_at = now
        self._belief = LocationBelief(
            snapshot_id=self._snapshot.snapshot_id,
            revision=BeliefRevision(
                stream_id=self._stream_id,
                sequence=self._sequence,
            ),
            estimate_time=estimate_time,
            published_at=now,
            status=status,
            confidence=None,
            hypotheses=hypotheses,
            source_health=(
                LocalizationSourceHealth(
                    source_id=self._profile.source_id,
                    state=source_state,
                    last_observation_at=self._last_observation_time,
                    detail_code=detail_code,
                ),
            ),
        )
        async with self._condition:
            self._condition.notify_all()
        return self._belief


async def _load_goal_bindings(
    *,
    map_engine: MapEngine,
    snapshot: MapSnapshot,
    profile: FixedStartVisualTrackingProfile,
) -> tuple[_GoalBinding, ...]:
    topology = await map_engine.query_topology(snapshot, TopologyQuery())
    chain = _directed_chain(topology.nodes, topology.segments)
    anchors_result = await map_engine.query_anchors(
        snapshot,
        AnchorQuery(
            kinds=frozenset({AnchorKind.VISUAL}),
            purposes=frozenset({AnchorPurpose.LOCALIZATION}),
            limit=max(200, len(chain) * 4),
        ),
    )
    anchors_by_node: dict[NodeId, list[AnchorDescriptor]] = {}
    for anchor in anchors_result.anchors:
        if anchor.attached_to.kind != MapEntityKind.NODE:
            continue
        node_id = NodeId(anchor.attached_to.entity_id)
        anchors_by_node.setdefault(node_id, []).append(anchor)

    selected_anchors = []
    for node_id in chain:
        candidates = anchors_by_node.get(node_id, [])
        if len(candidates) != 1:
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                f"node {node_id} must have exactly one visual localization "
                "anchor",
            )
        anchor = candidates[0]
        if len(anchor.resource_ids) != 1:
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                f"anchor {anchor.anchor_id} must reference one goal image",
            )
        selected_anchors.append(anchor)

    resource_ids = tuple(anchor.resource_ids[0] for anchor in selected_anchors)
    resource_result = await map_engine.resolve_resources(
        snapshot,
        resource_ids,
    )
    if resource_result.missing_resource_ids:
        raise LocalizationEngineError(
            LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
            "visual localization goal resources are missing",
        )
    resources = {
        resource.resource_id: resource for resource in resource_result.resources
    }

    bindings = []
    compatibility: tuple[str, str] | None = None
    for node_id, anchor, resource_id in zip(
        chain,
        selected_anchors,
        resource_ids,
        strict=True,
    ):
        resource = resources[resource_id]
        if resource.kind != ResourceKind.IMAGE:
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                f"resource {resource.resource_id} is not an image",
            )
        if resource.content_digest is None:
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                f"resource {resource.resource_id} has no content digest",
            )
        image_profile_id = resource.attributes.get("image_profile_id")
        model_artifact_id = resource.attributes.get("model_artifact_id")
        model_artifact_digest = resource.attributes.get(
            "model_artifact_digest"
        )
        if image_profile_id != profile.image_profile_id:
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                f"resource {resource.resource_id} has an incompatible image "
                "profile",
            )
        if not isinstance(model_artifact_id, str) or not isinstance(
            model_artifact_digest, str
        ):
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                f"resource {resource.resource_id} lacks model provenance",
            )
        current_compatibility = (
            model_artifact_id,
            model_artifact_digest,
        )
        if compatibility is None:
            compatibility = current_compatibility
        elif compatibility != current_compatibility:
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                "visual goal resources use different model artifacts",
            )
        bindings.append(
            _GoalBinding(
                node_id=node_id,
                anchor=anchor,
                resource=resource,
                model_artifact_id=model_artifact_id,
                model_artifact_digest=model_artifact_digest,
            )
        )
    return tuple(bindings)


def _directed_chain(
    nodes: tuple[TopologyNode, ...],
    segments: tuple[SegmentDescriptor, ...],
) -> tuple[NodeId, ...]:
    start_node_id = FIXED_START_NODE_ID
    node_ids = {node.node_id for node in nodes}
    if start_node_id not in node_ids:
        raise LocalizationEngineError(
            LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
            f"fixed start node is missing: {start_node_id}",
        )
    outgoing: dict[NodeId, list[SegmentDescriptor]] = {}
    incoming_count = {node_id: 0 for node_id in node_ids}
    for segment in segments:
        if (
            segment.source_node_id not in node_ids
            or segment.target_node_id not in node_ids
        ):
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                f"segment {segment.segment_id} references a missing node",
            )
        outgoing.setdefault(segment.source_node_id, []).append(segment)
        incoming_count[segment.target_node_id] += 1
    if incoming_count[start_node_id] != 0:
        raise LocalizationEngineError(
            LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
            "fixed start node must be the root of the directed chain",
        )

    chain = []
    visited = set()
    current = start_node_id
    while True:
        if current in visited:
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                "fixed-start topology contains a cycle",
            )
        visited.add(current)
        chain.append(current)
        candidates = outgoing.get(current, [])
        if not candidates:
            break
        if len(candidates) != 1:
            raise LocalizationEngineError(
                LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
                f"node {current} does not have exactly one successor",
            )
        current = candidates[0].target_node_id

    if visited != node_ids or len(segments) != max(0, len(nodes) - 1):
        raise LocalizationEngineError(
            LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
            "fixed-start topology must be one complete directed chain",
        )
    if any(
        count != 1
        for node_id, count in incoming_count.items()
        if node_id != start_node_id
    ):
        raise LocalizationEngineError(
            LocalizationErrorCode.INCOMPATIBLE_SNAPSHOT,
            "every non-start node must have one predecessor",
        )
    return tuple(chain)
