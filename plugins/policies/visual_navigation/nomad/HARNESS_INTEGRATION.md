# NoMaD Harness Integration Handoff

Status: draft handoff for a new navigation harness, 2026-08-19.

This document describes the assets that exist today, their exact contracts,
the provisional fixed-start node state machine, and the responsibilities that
remain outside the NoMaD policy plugin. It is not a hardware qualification
report.

## 1. Current Local Resources

| Resource | Local path | Purpose |
| --- | --- | --- |
| NoMaD plugin | `/home/qcraft/Workspace/longship/plugins/policies/visual_navigation/nomad` | PyTorch model, image ingress, checkpoint loading, inference, and offline topomap builder |
| Checkpoint | `/home/qcraft/Workspace/visualnav-transformer/deployment/model_weights/nomad.pth` | Released NoMaD state dictionary |
| Source video | `/home/qcraft/Desktop/map-recording.mp4` | 30 Hz map-recording traversal used by the asynchronous video mock |
| Dense replay | `/home/qcraft/Desktop/nomad_20260819_103700_analysis/dense_topomap` | 269 timestamped source images used to build and replay the map |
| Control topomap | `/home/qcraft/Desktop/nomad_20260819_103700_analysis/adaptive_topomap_nomad_d6_12_safe` | Current 40-node directed control-goal map |
| Offline edge replay | `/home/qcraft/Desktop/nomad_20260819_103700_analysis/adaptive_topomap_nomad_d6_12_safe_inference` | Per-edge trajectory overlays, contact sheets, and replay results |

Checkpoint identity:

```text
size:   76,473,631 bytes
sha256: 70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333
```

The checkpoint, dense images, generated topomaps, and replay images are local
artifacts. They are not stored in Git. A deployable harness must receive these
paths or resolved immutable artifact references through configuration.

## 2. What Exists and What Does Not

The plugin currently provides:

- checkpoint-compatible NoMaD model construction;
- strict checkpoint loading;
- tensor-only image canonicalization;
- four-frame chronological observation buffering;
- goal-conditioned distance and diffusion-trajectory inference;
- deterministic DDPM scheduling without `diffusers`;
- adaptive offline topomap construction using the distance head; and
- unit and real-checkpoint compatibility tests.

The first integration now also provides:

- a Longship Map Engine adapter;
- a distance-only visual policy adapter;
- a decoded-frame protocol and fixed-grid NoMaD observation producer;
- a plugin-neutral continuous Localization Runtime in Longship core; and
- dense and FFmpeg recorded-video integration tools.

It does not currently provide:

- a production camera driver or production compressed-image decoder;
- arbitrary-start global localization;
- online topomap tracking;
- trajectory sample selection policy;
- robot waypoint units or command scaling;
- a base, locomotion, or joint controller;
- command arbitration, watchdogs, or safety integration; or
- reverse-route qualification.

NoMaD output is a candidate trajectory. It is never a robot command by itself.

## 3. Model and Tensor Contract

Current validated environment:

```text
Python baseline:     3.11+
Validated locally:   Python 3.11.12
PyTorch baseline:    2.1+
Validated locally:   PyTorch 2.6.0+cu124
efficientnet-pytorch 0.7.1
Pillow for offline tools: 11.2.1
model parameters:    19,049,675
```

Direct policy input:

```text
observations: [4, 3, H, W] or [B, 4, 3, H, W]
goal:         [3, H, W] or [B, 3, H, W]
dtype:        floating point
channel:      RGB, channel first
range:        [0, 1]
frame order:  oldest to newest
```

The image-ingress helper also accepts decoded HWC/CHW, RGB/BGR, and
`uint8`/float tensors. It canonicalizes them to contiguous RGB CHW `float32`
values in `[0, 1]`.

Current preprocessing is exactly:

```text
decoded RGB image
-> direct bilinear resize to 96 x 96
-> ImageNet normalization
```

The current map and all reported replay results use direct resize. They do not
use the proposed 4:3 center crop. A harness must preserve direct resize for
this map. Enabling center crop requires regenerating and replaying the map with
the identical crop applied to online observations and goal images.

Policy output for batch size `B` and sample count `N`:

```text
distance: [B]
actions:  [B, N, 8, 2]
```

`distance` is a learned temporal reachability value. It is not meters or
seconds. `actions` contains raw, unnormalized-to-robot NoMaD-frame trajectories.

The offline visualizations multiply trajectories by `0.05`. That value comes
from the legacy LoCoBot deployment expression `MAX_V / RATE = 0.2 / 4`. It is
only a visualization and reference-deployment convention. A new robot harness
must not silently reuse it without target-specific calibration and safety
review.

## 4. Current Dense Replay

```text
images:       269
filenames:    0000.png through 0268.png
resolution:   1280 x 720
time range:   6.0 s through 104.8 s
duration:     98.8 s
camera view:  forward view with the newer, higher pitch
```

The dense timestamps are not uniformly spaced. Fixed node offsets therefore do
not represent fixed time intervals. The dense manifest is the source of truth
for frame time, source node, sharpness, optical flow, and keyframe provenance.

## 5. Current Directed Control Topomap

The safe adaptive map was built with:

```text
candidate gap:       0.8 to 4.0 s
preferred distance:  6 to 12
hard distance:       3 to 15
target distance:     9
minimum sharpness:   250
context jitter:      source endpoint -1, 0, and +1 where available
preprocessing:       direct resize, no center crop
direction:           forward only
```

Output layout:

```text
adaptive_topomap_nomad_d6_12_safe/
├── images/          # 0000.png through 0039.png, control goal images
├── manifest.json    # selection configuration and node provenance
├── edges.json       # every edge and every scored candidate
└── summary.json     # aggregate statistics
```

Pass the `images/` directory, not the directory containing JSON files, to code
that expects the official numeric-only topomap layout.

Map summary:

```text
dense source frames:            269
selected control goal nodes:     40
directed edges:                  39
node compression ratio:       14.9%
mean edge time:                2.50 s
median edge time:              2.40 s
robust mean distance:         10.736
robust maximum distance:      14.507
non-terminal edges in 3..15:   38 / 38
non-terminal edges in 6..12:   28 / 38
```

Node provenance:

| Topology node | Dense node | Time (s) | Goal image |
| ---: | ---: | ---: | --- |
| 0 | 3 | 7.4 | `0000.png` |
| 1 | 13 | 10.8 | `0001.png` |
| 2 | 16 | 11.8 | `0002.png` |
| 3 | 23 | 15.8 | `0003.png` |
| 4 | 26 | 18.2 | `0004.png` |
| 5 | 36 | 21.8 | `0005.png` |
| 6 | 40 | 23.6 | `0006.png` |
| 7 | 49 | 26.4 | `0007.png` |
| 8 | 59 | 30.0 | `0008.png` |
| 9 | 67 | 32.4 | `0009.png` |
| 10 | 80 | 36.4 | `0010.png` |
| 11 | 87 | 38.4 | `0011.png` |
| 12 | 98 | 42.4 | `0012.png` |
| 13 | 107 | 45.0 | `0013.png` |
| 14 | 119 | 48.6 | `0014.png` |
| 15 | 123 | 49.6 | `0015.png` |
| 16 | 129 | 51.6 | `0016.png` |
| 17 | 133 | 54.2 | `0017.png` |
| 18 | 137 | 55.8 | `0018.png` |
| 19 | 140 | 56.6 | `0019.png` |
| 20 | 143 | 57.6 | `0020.png` |
| 21 | 150 | 60.0 | `0021.png` |
| 22 | 161 | 64.0 | `0022.png` |
| 23 | 169 | 66.4 | `0023.png` |
| 24 | 174 | 68.0 | `0024.png` |
| 25 | 180 | 70.8 | `0025.png` |
| 26 | 183 | 74.0 | `0026.png` |
| 27 | 188 | 75.8 | `0027.png` |
| 28 | 193 | 77.8 | `0028.png` |
| 29 | 199 | 79.6 | `0029.png` |
| 30 | 207 | 83.2 | `0030.png` |
| 31 | 218 | 86.6 | `0031.png` |
| 32 | 228 | 90.0 | `0032.png` |
| 33 | 231 | 91.8 | `0033.png` |
| 34 | 239 | 94.2 | `0034.png` |
| 35 | 245 | 97.4 | `0035.png` |
| 36 | 257 | 101.2 | `0036.png` |
| 37 | 261 | 102.6 | `0037.png` |
| 38 | 266 | 104.4 | `0038.png` |
| 39 | 268 | 104.8 | `0039.png` |

Topology goal images are sparse control subgoals. They must not be used as the
four consecutive observation frames. Online observations must always come from
the live camera stream.

## 6. Offline Edge Replay Result

Every directed edge was replayed with the original four-frame dense context,
its selected next goal, four diffusion samples, and sample `0` waypoint `2` for
visualization.

```text
edges replayed:                    39
finite distance and waypoint:      39 / 39
selected waypoint x > 0:           39 / 39
selected |waypoint y| <= 0.02 m:   35 / 39
direct distance in 3..15:          36 / 39
direct distance > 15:               2 / 39
terminal distance < 3:              1 / 39
direct mean distance:              11.056
```

Two edges require attention before hardware use:

```text
dense 3 -> 13:   direct distance 15.361
dense 36 -> 40:  direct distance 15.172
```

The final edge, dense `266 -> 268`, is only `0.4 s` and has direct distance
`1.408`. It is a terminal arrival edge. The harness must stop on arrival and
must not execute the trajectory sampled for that edge.

The robust builder uses the median of neighboring contexts. Direct replay and
the robust value have mean absolute difference `1.569` and maximum difference
`6.760`. Before hardware qualification, edge selection should require both the
center context and the worst neighboring context to remain in the hard range.

## 7. Provisional Fixed-Start Navigation State

Arbitrary-start global localization is deferred. The first harness assumes the
robot is placed at topology node `0`, facing in the recorded route direction,
with matching camera pitch and preprocessing.

State meaning:

```text
current_node = last topology node confirmed reached
target_node  = next topology node currently used as the goal image
```

Initial state:

```text
current_node = none
target_node = 0
last_node = 39
close_count = 0
far_count = 0
```

Recommended harness states:

```text
WAIT_CONTEXT
-> VERIFY_START
-> SEARCHING_NEXT
-> TRACKING
-> ADVANCE_TARGET
-> SEARCHING_NEXT
-> GOAL_REACHED

Any active state
-> LOCALIZATION_LOST or FAULT
```

### 7.1 WAIT_CONTEXT

Collect four valid chronological observation frames. Reject stale timestamps,
out-of-order frames, a resolution change within the context, and frames using
the wrong color or crop profile.

### 7.2 VERIFY_START

The robot is operationally declared to start at node `0`, but the harness
should still compare the live context to goal image `0` before allowing motion.

A provisional gate is two consecutive start distances below `3`. This threshold
must be calibrated with repeated real starts and should remain configurable.
Failure to verify node `0` must hold or stop; it must not silently search the
map in the first fixed-start implementation.

### 7.3 TRACKING and node advancement

At every localization tick, evaluate one observation context against a bounded
local map window in a single encoder batch:

```python
indices = [current_node, target_node, target_node + 1]
distances = predict_goal_distances(latest_context, goals[indices])
evidence.append(classify_local_window(indices, distances))

if successor_is_close(evidence) or successor_repeatedly_wins(evidence):
    advance_exactly_one_node()
elif later_candidate_is_repeatedly_close(evidence):
    publish_lost_and_enter_forward_recovery()
```

Provisional values:

```text
close_threshold:                 3
start_close_confirmations:       2 policy ticks
successor_close_confirmations:   1 policy tick
tracking_candidate_count:        3 nodes
evidence_window_size:            3 policy ticks
relative_advantage_minimum:      1 temporal-distance unit
relative_distance_maximum:       5 temporal-distance units
relative_advance_confirmations:  2 votes in the evidence window
lookahead_close_confirmations:   2 votes before forward recovery
```

The absolute-close gate remains useful, but it is no longer the only evidence.
A bounded relative vote tolerates a missed exact keyframe only when the
expected successor is the best local candidate, is itself nearby, and beats
the current node by the configured margin. The evidence window suppresses a
single noisy comparison.

After changing `target_node`, never publish a trajectory conditioned on the old
goal. Hold or stop for that tick, load the new goal, and infer again.

Normal tracking advances exactly one node at a time and never moves backward.
A repeatedly close look-ahead node is treated as evidence that the expected
successor window was missed, not as permission to skip silently.

### 7.4 Weak and lost target handling

The following values are provisional harness guards, not properties guaranteed
by the checkpoint:

```text
best local distance <= 15:      usable local candidate set
15 < best local distance < 18:  weak candidate set; publish DEGRADED
best local distance >= 18:      untrusted local candidate set
three untrusted ticks:          enter LOCALIZATION_LOST
```

`LOCALIZATION_LOST` is a non-motion state: the Local Trajectory Engine publishes
`HOLDING`. Localization continues evaluating an eight-node forward-only window.
Two close confirmations for the same best node restore `TRACKING`; recovery
cannot jump backward and cannot establish an arbitrary global start. A wider or
bidirectional search belongs to a future global-localization engine.

All candidate distances and the exact observation timestamp should be logged.
Distance is still a learned temporal reachability score, so these guards must
be calibrated on independent traversals rather than interpreted as metric
geometry.

### 7.5 Final node

When target node `39` satisfies the close gate, transition directly to
`GOAL_REACHED` and publish a stop. Do not increment beyond `39`, wrap to `0`,
or execute the terminal edge's sampled trajectory.

## 8. Proposed Harness-Owned Interfaces

The new harness should own typed equivalents of the following data:

```text
CameraFrame
  image_tensor
  timestamp_s
  sequence_id
  image_profile_id

TopologyNode
  node_id
  goal_image_path
  source_dense_node
  source_timestamp_s

TopologyEdge
  source_node
  target_node
  offline_robust_distance
  offline_time_delta_s
  qualification_status

NavigationState
  phase
  current_node
  target_node
  close_count
  far_count
  last_observation_timestamp_s
  last_inference_timestamp_s

PolicyCandidate
  observation_timestamp_s
  target_node
  predicted_distance
  trajectories
  model_artifact_id
  valid_until_s
```

Suggested component boundaries:

```text
ImageSource
-> ContextSampler
-> TopomapLoader
-> FixedStartNavigator
-> NomadPolicyWorker
-> CandidateGuard
-> target-specific controller adapter
-> command arbiter and independent safety layer
```

Only `NomadPolicyWorker` should import the model package. Node state transitions
belong to the harness. Robot command conversion belongs to a separately
qualified target adapter.

## 9. Suggested Frequencies

The official reference uses a 9 Hz camera, 4 Hz navigation updates, and a 9 Hz
PD controller. The localization profile deliberately evaluates evidence more
frequently than it publishes steady beliefs:

```text
camera capture:       15 to 30 Hz
context insertion:     9 to 10 Hz
distance evidence:     9 Hz
steady belief stream:  4 Hz maximum; transitions immediate
trajectory sampling:  driven by new belief revisions
base/locomotion loop: 50 Hz or higher
safety monitoring:   100 Hz or higher
```

If the camera runs at 30 Hz, inserting every raw frame would make four context
frames span only about `0.1 s`. Sample the context stream near 9 to 10 Hz so
four frames cover roughly `0.3 s`. Always use the newest complete context.

The implemented video mock follows this split: FFmpeg decodes the source video
at 30 Hz, `NomadObservationProducer` samples on a non-drifting 9 Hz time grid,
and `ContinuousLocalizationService` evaluates batched local distance evidence
at 9 Hz. The engine throttles unchanged beliefs to 4 Hz. Each new belief then
drives `LocalizationDrivenLocalTrajectoryService`, so localization keeps
priority over trajectory sampling on the shared serial CUDA executor. One
fanout keeps distance and trajectory observation contexts aligned. No mission
call drives a frame or policy tick.

Recorded-video compositions use source video time for observation and policy
clocks; wall time only controls playback cadence. This makes context selection
independent of event-loop or CUDA load. The full 2026-08-19 CUDA composition
generated a 39-traversal RoutePlan, published 297 active full-trajectory rows
across targets `node-0001` through `node-0039`, emitted five `HOLDING` rows for
degraded localization, coalesced no belief revisions, and ended in
`ROUTE_COMPLETED`.

On the map-recording traversal from `6.0 s`, CUDA replay advanced monotonically
to `node-0039`. It detected the missed `node-0033` window, held motion while
`LOST`, and recovered at `node-0034` before reaching the final-node phase. This
validates timing, lifecycle, local recovery, and state-transition integration
only. Required independent-traversal and hardware qualification remain open.

Policy output may be held only while it is fresh. The harness must define a
short timeout, publish a stop or hold on timeout, and discard late inference
for an older observation or target node.

## 10. Minimal Policy Call

```python
import torch

from nomad_runtime import NomadPolicy


device = torch.device("cuda:0")
policy = NomadPolicy.from_checkpoint(
    checkpoint_path,
    device=device,
    strict=True,
)

output = policy.infer(
    observations,  # [4, 3, H, W], RGB float in [0, 1]
    goal,          # [3, H, W], RGB float in [0, 1]
    num_samples=4,
)

distance = output.distance[0]
trajectories = output.actions[0]  # [4, 8, 2]
```

The harness must associate the result with the observation timestamp and target
node used to create it. If either has changed before completion, discard the
result.

## 11. Required Work Before Hardware Motion

The following items remain blocking for hardware motion, even though model
inference and offline replay work:

1. Split or replace direct-replay edges `dense 3 -> 13` and `36 -> 40`.
2. Tighten offline selection so center and worst neighboring context pass the
   hard edge bounds.
3. Replay using an independent traversal, not only the map-recording video.
4. Calibrate fixed-start verification and close-confirmation thresholds.
5. Qualify the initial explicit `first_candidate.v1` sample-`0` selection on
   independent traversals; it is defined for the Harness but not hardware-safe.
6. Define Jackie-specific trajectory units, waypoint choice, command scaling,
   velocity and yaw limits, and controller frequency.
7. Add observation, inference, candidate, and command watchdogs.
8. Add stop, hold, cancellation, stale-output, and model-failure behavior.
9. Preserve direct-resize preprocessing or regenerate every map artifact after
   changing the image profile.
10. Qualify the forward route. Reverse navigation requires a separately
    recorded and validated directed map.

Until these gates are complete, this package and map should be treated as an
offline-validated navigation-policy candidate, not an actuator-ready system.
