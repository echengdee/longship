# Trajectory Policy Port Interface Specification v0.1

> Status: initial draft
> Scope: boundary between the Harness route-trajectory runtime and visual local-policy plugins

## 1. Role

`VisualGoalTrajectoryPolicy` generates candidate trajectories for one selected,
directed route segment. It is neither a top-level Navigation Harness engine nor
the mission-facing `RouteExecutionPort`. The `LocalTrajectoryEngine` depends on
this SPI to inject a concrete policy plugin; Mission, Map, Localization, and
Planning do not depend on a concrete policy implementation.

```text
Harness LocalTrajectoryEngine
    + active Segment
    + latest observation context
    + Map-owned TARGET resource
             |
             v
VisualGoalTrajectoryPolicy
             |
             v
raw TrajectoryCandidateSet
```

## 2. Request

`VisualGoalTrajectoryRequest` binds all of the following:

- the Map `snapshot_id`;
- `segment_id`, `source_node_id`, and `target_node_id`;
- the `target_anchor_id` resolved by the Map Engine and its immutable goal-image
  resource;
- the request time and maximum permitted observation age;
- the candidate count and reproducible sampling seed;
- the image profile, model artifact ID, and SHA-256 digest.

The policy must reject requests with mismatched snapshots, target resources,
image profiles, or model identities. Asynchronous results produced from stale
observations or goals must not remain usable after the active segment changes.

## 3. Result

`TrajectoryCandidateSet` preserves every route and resource identity from the
request and adds:

- the actual observation and generation times;
- the NoMaD temporal distance;
- every candidate trajectory and every waypoint in each trajectory;
- policy, model, and image-profile identities;
- coordinate-frame and unit labels;
- the sampling seed that was actually used.

In v0.1, waypoints are raw two-dimensional values in
`nomad.policy_native.robot_frame.v1`. The interface does not claim that they
represent meters on the target robot, velocities, or time-parameterized control
trajectories.

## 4. Explicit Non-responsibilities

- selecting one of multiple candidates;
- selecting an individual waypoint;
- applying LoCoBot or target-robot scale factors;
- collision checking, dynamic obstacle avoidance, or feasibility constraints;
- velocity, angular-velocity, curvature, or acceleration limits;
- watchdogs, safety arbitration, controllers, or platform commands.

Candidate selection belongs to the Local Trajectory Engine. Scale conversion,
collision checking, safety, tracking, and control belong outside the Harness.
A candidate trajectory must not become a robot command before passing through
those layers.

## 5. Current NoMaD Adapter

The NoMaD adapter consumes four consecutive observations and the Map Engine's
visual `TARGET` image, then returns every diffusion sample as `[N, 8, 2]`.
Offline video tools draw the candidates in a fixed-range, robot-frame bird's-eye
view together with the current node, target node, and Map goal image. The policy
adapter neither applies an uncalibrated camera-perspective projection nor
selects a candidate. The initial Local Trajectory Engine uses the explicit
`first_candidate.v1` policy to select candidate `0` and publishes all eight
waypoints through `LocalTrajectoryStream`.
