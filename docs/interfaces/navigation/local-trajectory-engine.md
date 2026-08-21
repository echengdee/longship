# Local Trajectory Engine Public Interface Specification v0.1

> Status: initial version
> Owner: Navigation Harness Local Trajectory Engine
> Consumers: safety, tracking, and platform-integration layers outside the Harness

## 1. Role

`LocalTrajectoryEngine` is the fifth top-level Navigation Harness engine. It is
route-bound and stateful: it combines one pinned `RoutePlan`, continuous
`LocationBelief` updates, Map goal resources, and trajectory-policy results into
immutable `LocalTrajectoryStream` publications.

It owns active-traversal resolution, monotonic route progress, goal binding,
asynchronous policy-result validation, candidate selection, stream revision,
and expiry semantics. It does not search for the global route; that remains the
stateless Planning Engine's responsibility.

It publishes short-lived motion proposals, not control commands. It does not
claim that a trajectory has passed collision, safety, or platform-dynamics
checks.

## 2. Public Facade and Stream

```python
class LocalTrajectoryStream(Protocol):
    def get_latest(self) -> LocalTrajectoryPublication: ...

    async def wait_for_update(
        self,
        request: WaitForLocalTrajectoryRequest,
    ) -> LocalTrajectoryUpdateResult: ...

class LocalTrajectoryEngine(LocalTrajectoryStream, Protocol):
    pass
```

`get_latest()` reads the latest publication without blocking.
`wait_for_update()` uses a complete revision as its cursor and returns
`UPDATED`, `STREAM_RESET`, or `TIMED_OUT`. A wait timeout is a normal result.

The public facade is intentionally read-only. Runtime scheduling calls internal
`tick()`, `stop()`, and `fault()` operations through a narrow runtime protocol;
those methods are not exposed to Mission or external consumers.

The initial `RouteBoundLocalTrajectoryEngine` instance is single-use and binds
exactly one `MapSnapshot` and immutable `RoutePlan` at construction. Mission
decides when a new plan becomes active, while the composition/runtime layer
constructs and schedules the concrete engine. Replanning stops the old instance
and creates a new stream rather than mutating or replacing the bound route.

## 3. Revision

```text
LocalTrajectoryRevision
├── stream_id
└── sequence
```

- `sequence` is strictly increasing within one stream;
- an engine restart, new `RoutePlan`, or `MapSnapshot` change requires a
  new `stream_id`;
- waiting with a cursor from another `stream_id` returns `STREAM_RESET`;
- a revision identifies publication order but does not replace `valid_until`.

## 4. Publication

```text
LocalTrajectoryPublication
├── revision
├── route_id / snapshot_id
├── state
├── published_at
├── belief_revision
├── traversal_sequence
├── segment_id / source_node_id / target_node_id
├── target_anchor_id / goal_resource_id
├── observation_time / generated_at / valid_until
├── trajectory
├── hold_reason
└── detail_code
```

### 4.1 State Invariants

| State | `trajectory` | Meaning |
|---|---:|---|
| `INITIALIZING` | absent | Active output has not started |
| `HOLDING` | absent | Current inputs do not permit publication |
| `ACTIVE` | required | Complete local trajectory for this revision |
| `ROUTE_COMPLETED` | absent | Localization confirmed the `RoutePlan` goal node |
| `FAULTED` | absent | Runtime fault |
| `STOPPED` | absent | Service stopped and previous trajectories are invalid |

Only `ACTIVE` may be forwarded for processing outside the Harness. Every other
state invalidates the previous active trajectory and must never mean "continue
executing the last one."

### 4.2 Hold Reasons

The initial contract distinguishes:

- localization unavailable or outside the `RoutePlan`;
- observation context not ready or stale;
- Map goal resource unavailable;
- policy unavailable;
- traversal changed during inference, making the result stale;
- service stopped.

`detail_code` is diagnostic information and must not become the consumer's
primary control protocol.

## 5. LocalTrajectory

```text
LocalTrajectory
├── trajectory_id
├── source_candidate_id
├── waypoints[]
├── coordinate_frame / coordinate_units
├── selection_policy_id
├── source_candidate_index / source_candidate_count
├── sampling_seed
├── temporal_distance
├── policy_id / image_profile_id
└── model_artifact_id / model_artifact_digest
```

The initial NoMaD-backed engine always selects candidate `0` and preserves all
eight two-dimensional waypoints from that candidate. Coordinates remain labeled
`nomad.policy_native.robot_frame.v1` and `nomad.policy_native.v1`; the interface
does not claim that they are meters or a time-parameterized path.

## 6. Freshness and Asynchronous Results

Every `ACTIVE` publication carries:

- `observation_time`: the latest observation actually consumed by the policy;
- `generated_at`: when the policy result was produced;
- `published_at`: when the Harness published it;
- `valid_until`: the latest time at which an external consumer may use it.

All four timestamps must use the same clock domain. The engine reads
localization again after asynchronous inference. If the active traversal has
changed, it publishes `HOLDING / STALE_POLICY_RESULT` instead of publishing the
stale policy result as a trajectory.

Runtime composition injects one clock-domain `TimeSource` into the policy and
the engine. The policy records `generated_at` after inference completes, and
the engine records `published_at` after its result checks. If the resulting
proposal has already passed `valid_until`, the engine publishes `HOLDING`
instead of an expired `ACTIVE` trajectory.

## 7. External Consumer Requirements

An external consumer must, at minimum:

1. accept only `ACTIVE`;
2. validate `stream_id` and monotonic sequence order;
3. validate the expected `route_id` and `snapshot_id`;
4. check `valid_until` against its own clock;
5. revoke the previous proposal on every non-`ACTIVE` publication;
6. perform scale conversion, feasibility, safety, and control processing before
   commanding a robot.

The Harness does not mandate ROS 2, RPC, shared memory, or in-process
subscriptions for external consumers.

## 8. Distinction from Related Interfaces

- `VisualGoalTrajectoryPolicy`: plugin SPI that returns every raw candidate;
- `LocalTrajectoryEngine`: route-bound stateful engine that selects and
  publishes local motion proposals;
- `LocalTrajectoryStream`: read-only Harness output containing one selected, complete
  candidate and its associated identities;
- `RouteExecutionPort`: future contract for external execution commands,
  progress, pause, and cancellation;
- controller/chassis API: platform motion-control interface outside the Harness.
