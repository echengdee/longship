# Route Execution Port Public Interface Specification v0.1

> Status: reserved draft for a later phase; not instantiated by the current
> initial version
> Scope: future boundary for external route-execution state, pause, resume, and
> cancellation
> Dependencies: public domain models from Map Engine v0.1, Localization Engine
> v0.1, and Planning Engine v0.1
> Contract language: Python types; independent of ROS 2, RPC, local-navigation
> algorithms, controllers, and chassis protocols

## 1. Definition

`RouteExecutionPort` is an optional external capability for the next phase. The
Harness may use it to submit an immutable `RouteCommand`, read
`RouteExecutionStatus` published by an external system, and request pause,
resume, or cancellation.

It is not an internal Harness `Execution Engine` and contains no route-execution
implementation.

The current runnable version does not call this Port. It publishes
`LocalTrajectoryStream` from `RoutePlan` and `LocationBelief`; an external
integration layer consumes trajectories and performs safety, tracking, and
platform control. This document reserves the domain contract for adding command
lifecycle and execution feedback later.

## 2. Boundary Decision

The Navigation Harness contains only:

```text
Mission Engine
Map Engine
Localization Engine
Planning Engine
Local Trajectory Engine
```

The external route-execution system is responsible for:

- consuming the Harness `LocalTrajectoryStream`;
- checking trajectory expiry and revoking old proposals on hold or stop;
- real-time obstacle avoidance, trajectory tracking, and platform adaptation;
- consuming localization, observations, and platform feedback needed for
  execution;
- interacting with safety, controllers, and chassis;
- maintaining active commands, execution progress, and terminal states;
- returning structured state through this Port.

The Navigation Harness is responsible for:

- generating the global route;
- publishing complete local trajectories from the route, localization, Map goal,
  and policy candidates;
- after this Port is enabled, submitting route commands and consuming execution
  state;
- deciding whether to relocalize, replan, retry, or terminate from task budgets
  and failure evidence;
- verifying final success from the latest `LocationBelief` and task criteria.

## 3. This Project Defines Only the Port

The interface package contains only:

```text
route_execution_port/
├── interface.py
├── models.py
├── interface_spec.md
└── __init__.py
```

It does not provide:

```text
implementation.py
executor.py
route_follower.py
segment_executor.py
ros2_adapter.py
rpc_adapter.py
controller.py
```

An external route-execution system or integrator supplies the implementation.
The Harness depends only on the `Protocol` and never imports a concrete external
executor.

### 3.1 Relationship to the Trajectory Policy Port

`longship.navigation.ports.trajectory_policy` is the plugin SPI used by the
Harness `LocalTrajectoryEngine`; it is not part of `RouteExecutionPort`:

- Mission and the external executor do not request trajectory candidates
  directly;
- the Local Trajectory Engine passes the active segment, Map goal resource, and latest
  observations to `VisualGoalTrajectoryPolicy`;
- the result is a raw candidate set bound to snapshot, segment, target, and
  observation time;
- the Local Trajectory Engine selects a candidate, while unit calibration, control,
  safety, and platform dispatch remain external responsibilities.

The two Ports use separate packages. `route_execution.__init__` does not export
trajectory-policy interfaces, preventing accidental mixing of Mission and local
policy boundaries.

## 4. Public Protocol

```python
class RouteExecutionPort(Protocol):
    async def submit_route(
        self,
        command: RouteCommand,
    ) -> RouteSubmissionResult: ...

    def get_status(
        self,
        command_id: RouteCommandId,
    ) -> RouteExecutionStatus: ...

    async def wait_for_update(
        self,
        request: RouteExecutionUpdateRequest,
    ) -> RouteExecutionUpdateResult: ...

    async def control(
        self,
        request: RouteControlRequest,
    ) -> RouteControlResult: ...
```

| Method | Meaning |
|---|---|
| `submit_route()` | Submit an immutable route and wait only for acceptance or rejection |
| `get_status()` | Read the latest immutable state of one command immediately |
| `wait_for_update()` | Wait for a new revision, stable terminal state, or timeout |
| `control()` | Request pause, resume, or controlled cancellation |

v0.1 has no `replace_route()`. Replanning first moves the old command to a
terminal state, then submits a new `command_id` and `route_id`.

## 5. RouteCommand

```text
RouteCommand
├── command_id                 # idempotency key
├── mission_ref                # Navigation Mission reference
├── skill_call_id              # Longship SkillCall reference
├── resource_lease_id          # chassis lease granted by Runtime
├── cancellation_epoch         # prevents an old generation from resuming motion
├── expected_state_version     # Runtime state version at dispatch
├── issued_at
├── valid_until                # executor rejects or stops after expiry
├── snapshot                   # version token, not complete map data
├── route_plan                 # immutable global route
└── limits                     # mission-level runtime limits
```

### 5.1 Consistency Rules

Before submission:

```text
command.snapshot.snapshot_id
== command.route_plan.snapshot_id
```

Before execution, the external system also validates the resource lease,
cancellation epoch, Runtime state version, and expiry.

The external system must not:

- add, remove, or reorder `route_plan.traversals` in place;
- write a local detour back into the original `RoutePlan`;
- call the Harness Planning Engine itself;
- reuse one `command_id` for a semantically different route.

If local recovery cannot continue within the same route semantics, it returns
`FAILED` and Mission decides whether to replan.

### 5.2 Idempotency

- a new `command_id` may be accepted or rejected;
- resubmitting the same semantic command returns `ALREADY_EXISTS` or the
  original acceptance result;
- reusing `command_id` with a different `RoutePlan` or limits raises
  `IDEMPOTENCY_CONFLICT`;
- idempotency survives ordinary network retries.

## 6. RouteSubmissionResult

```text
RouteSubmissionResult
├── command_id
├── route_id
├── outcome
├── decided_at
├── status                     # optional for an accepted command
└── rejection                  # present for rejection
```

| Outcome | Meaning |
|---|---|
| `ACCEPTED` | External system took ownership of the command |
| `ALREADY_EXISTS` | Idempotent retry of an existing command |
| `REJECTED` | External system rejected ownership with a structured reason |

Normal rejection does not raise. Examples include executor busy, expired route,
insufficient capability, unavailable resource, or unmet start conditions.

## 7. RouteExecutionStatus

```text
RouteExecutionStatus
├── command_id
├── route_id
├── snapshot_id
├── revision
│   ├── command_id
│   └── sequence
├── state
├── created_at / updated_at
├── progress
├── completion                 # present for SUCCEEDED
├── failure                    # present for FAILED
├── cancellation_reason
└── detail_code
```

### 7.1 Lifecycle

```text
ACCEPTED → RUNNING → SUCCEEDED
                   → FAILED
                   → PAUSING → PAUSED → RESUMING → RUNNING
                   → CANCELLING → CANCELLED
```

Terminal states are `SUCCEEDED`, `FAILED`, and `CANCELLED`. They never
transition. `SUCCEEDED` has `completion`; `FAILED` has `failure`; other states
cannot carry those fields.

External `SUCCEEDED` means the route command finished, not that the Mission
succeeded. Mission may still verify the latest localization and task criteria.

### 7.2 Revision

- `sequence` increases strictly within one `command_id`;
- one revision's state is immutable;
- revision changes only for externally observable state, not at control rate;
- `revision.command_id` equals the Status `command_id`;
- `wait_for_update()` uses the complete revision as its cursor.

## 8. Execution Progress

```text
RouteExecutionProgress
├── completed_traversal_count
├── total_traversal_count
├── active_traversal_sequence
├── active_segment_id
├── segment_progress
├── route_progress
├── latest_belief_revision
└── last_progress_at
```

Rules:

- `completed_traversal_count` never regresses;
- present percentages are in `[0, 1]`;
- unreliable values use `None` rather than fabricated linear progress;
- `active_traversal_sequence` references `RoutePlan.traversals[].sequence`;
- `latest_belief_revision` is optional evidence and does not require the
  executor to use the Harness's localization implementation.

## 9. Failure Model

```text
RouteExecutionFailure
├── reason
├── failed_at
├── active_segment_id
├── related_segment_ids
├── last_belief_revision
├── retryable_same_route
├── replan_recommended
└── detail_code
```

| Reason | Possible Harness task-level response |
|---|---|
| `BLOCKED` / `NO_PROGRESS` | Add related segment to temporary unavailable context and replan |
| `OFF_ROUTE` | Read new location and replan |
| `LOCALIZATION_UNAVAILABLE` | Wait or request relocalization |
| `MAP_SNAPSHOT_CHANGED` | Terminate old plan, rebind map, and plan again |
| `PLAN_INVALIDATED` | Update `PlanningContext` from evidence and replan |
| `SAFETY_STOPPED` | Remain stopped until Mission or an upper layer decides |
| `PLATFORM_FAULT` | Normally terminate and report platform failure |
| `GOAL_NOT_REACHED` | Relocalize, plan a corrective route, or fail the task |

`replan_recommended` is advice from the external system. Mission still decides
from task budget, failure history, and current location.

## 10. Pause, Resume, and Cancel

```text
RouteControlRequest
├── control_command_id         # control-request idempotency key
├── route_command_id
├── skill_call_id
├── resource_lease_id
├── cancellation_epoch
├── requested_at
├── action                     # PAUSE / RESUME / CANCEL
└── reason
```

`control()` waits only for acceptance or rejection, not physical completion.
Callers confirm through subsequent Status updates:

- `PAUSE`: `PAUSING → PAUSED`
- `RESUME`: `RESUMING → RUNNING`
- `CANCEL`: `CANCELLING → CANCELLED`

`RESUME` revalidates the current resource lease and cancellation epoch. A
request from an older generation cannot resume a route cancelled by Longship
Runtime.

Pause and cancel are normal controlled route operations, not emergency stop.
Physical emergency and safety stops always belong to the external Safety and
Platform systems.

## 11. No Implementation Commitment

This Port does not prescribe:

- a single-process or distributed external system;
- ROS 2 Action, Topic, Service, RPC, or shared memory;
- visual navigation, local planning, tracking, or control algorithms;
- how the external system obtains images, point clouds, localization, or maps;
- persistence or disaster recovery;
- status publication frequency.

Implementations need only satisfy domain semantics, idempotency, immutable
state, and terminal-state rules.

## 12. Normal Results and Exceptions

Structured results, not exceptions, represent:

- normal route rejection;
- running, pausing, or cancelling commands;
- route failure;
- status-wait timeout;
- a control action disallowed in the current state.

Only contract, addressing, transport, or infrastructure failures raise
`RouteExecutionPortError`:

```text
INVALID_REQUEST
COMMAND_NOT_FOUND
IDEMPOTENCY_CONFLICT
PORT_UNAVAILABLE
EXTERNAL_EXECUTOR_UNAVAILABLE
TRANSPORT_FAILURE
INTERNAL_FAILURE
```

## 13. Typical Mission-side Use

```python
submission = await route_execution_port.submit_route(command)

if submission.outcome is RouteSubmissionOutcome.ACCEPTED:
    status = submission.status
    while status is not None and status.state not in {
        RouteExecutionState.SUCCEEDED,
        RouteExecutionState.FAILED,
        RouteExecutionState.CANCELLED,
    }:
        update = await route_execution_port.wait_for_update(
            RouteExecutionUpdateRequest(
                after_revision=status.revision,
                timeout_s=1.0,
            )
        )
        status = update.status
```

This illustrates Port consumption only; it is not an implementation of Mission
or the external execution system.

## 14. State Ownership

| Data | Owner |
|---|---|
| `MissionState`, task budget, and recovery policy | Mission Engine |
| `RoutePlan` semantics | Produced by Planning; stored as immutable reference by Mission |
| Active command, execution progress, and terminal state | External route-execution system |
| Raw trajectory candidates and selected local trajectory | Policy plugin / Local Trajectory Engine |
| Safety, control, and platform feedback | External execution system / Robot Platform |
| Port object | No domain state; boundary calls only |

Mission may cache the latest Status and revision but cannot rewrite external
state. The external system may report recommendations but does not own Mission's
replanning or final success decision.

## 15. v0.1 Baseline

1. There is no internal Harness `Execution Engine`.
2. `RouteExecutionPort` is an external dependency interface, not a sixth engine.
3. The current Harness publishes local motion proposals through
   `LocalTrajectoryStream`, not controls.
4. The external system owns actual route-execution state.
5. The Harness decides whether to replan from structured failure evidence.
6. The Port package provides only future `Protocol`, DTOs, error semantics, and
   specification, with no implementation.
7. The current initial version does not instantiate this Port; enabling it
   requires another interface review.
