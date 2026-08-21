# Longship NoMaD Policy Plugin

`nomad_runtime` is a small, PyTorch-only package for the released NoMaD visual
navigation checkpoint. It is intentionally independent from ROS, camera
drivers, topomap storage, and robot motion control.

This directory contains the policy inference library, provisional Longship
adapters, and offline integration tools. The plugin manifest's public
contracts and runtime exports remain deliberately empty until the production
image source, candidate guard, and robot boundary are qualified. Model outputs
are candidates and never robot commands.

See [`HARNESS_INTEGRATION.md`](HARNESS_INTEGRATION.md) for the current local
assets, control-goal map, fixed-start node state machine, proposed harness
boundaries, validation evidence, and known hardware blockers.

Current scope:

- build the checkpoint-compatible NoMaD network;
- strictly load the released `nomad.pth` state dictionary;
- convert decoded HWC/CHW, RGB/BGR, uint8/float image tensors to one format;
- maintain a timestamped four-frame observation context;
- normalize an RGB observation context and RGB goal tensor;
- predict temporal goal distance;
- sample goal-conditioned waypoint trajectories with the ten-step DDPM policy;
- adapt a published NoMaD topomap to the Longship Map Engine;
- adapt distance inference to fixed-start Localization;
- adapt raw trajectory inference to the optional executor-side trajectory Port;
- render raw candidates and a diagnostic stitched path over an offline
  recorded-video replay.

Deferred scope:

- production camera drivers and image decoding;
- arbitrary-start global localization;
- waypoint selection, scaling, and control;
- ROS or robot SDK integration;
- safety and command arbitration.

## Offline trajectory visualization

`tools/render_video_trajectory_overlay.py` can show both the four raw NoMaD
candidates and a long-horizon diagnostic path. For each inference it computes
the coordinate-wise median candidate, takes only a fixed short arc-distance
step, and composes that step into the replay start frame. The default is `0.15`
policy-native units and can be changed with `--stitch-step-distance`.

The white line in the robot-frame panel is the median candidate, the red short
line is the increment used for stitching, and the center panel is the complete
stitched path so far. The JSONL sidecar records the representative path, local
increment, and resulting stitched pose for each inference.

This path is deliberately diagnostic. It is open-loop dead reckoning over raw
policy outputs, has no metric scale claim, does not use measured robot motion,
and must not be treated as odometry or a control command.

For comparison with the released deployment demo, pass
`--stitch-mode official_demo --num-candidates 8`. This mode fixes the choice to
sample 0 and waypoint index 2, applies the demo's `MAX_V / RATE` scale, runs the
same bounded waypoint-to-velocity controller, and integrates its unicycle
motion at the inference rate. Defaults match the included robot configuration:
4 Hz, `0.2 m/s`, and `0.4 rad/s`. It remains an open-loop controller mock; the
recorded video does not react to these commands.

## Dependencies

The runtime requires Python 3.11 or newer, PyTorch 2.1 or newer, and
`efficientnet-pytorch==0.7.1`. It does not require torchvision, NumPy, Pillow,
PyYAML, diffusers, diffusion-policy, wandb, or ROS.

Install it from the Longship repository:

```bash
conda activate nomad
python --version  # must report 3.11 or newer
python -m pip install -e \
  ./plugins/policies/visual_navigation/nomad
```

The checkpoint is a generated model artifact and is not stored in this
repository. Pass its path explicitly at runtime.

The recorded-video mock additionally requires `ffmpeg` and `ffprobe` on the
host. They are tool-only executables and are not model-runtime dependencies.

## Tensor API

Inputs are floating-point RGB values in `[0, 1]`:

- observations: `[batch, 4, 3, height, width]` or `[4, 3, height, width]`;
- goal: `[batch, 3, height, width]` or `[3, height, width]`.

The runtime resizes them to `96 x 96` and applies ImageNet normalization.

```python
import torch

from nomad_runtime import NomadPolicy


device = torch.device("cuda:0")
policy = NomadPolicy.from_checkpoint(
    "/path/to/nomad.pth",
    device=device,
    strict=True,
)

observations = torch.rand((1, 4, 3, 720, 1280), device=device)
goal = torch.rand((1, 3, 720, 1280), device=device)
generator = torch.Generator(device=device).manual_seed(0)

output = policy.infer(
    observations,
    goal,
    num_samples=4,
    generator=generator,
)

print(output.distance.shape)  # [1]
print(output.actions.shape)   # [1, 4, 8, 2]
```

`output.actions` contains raw NoMaD robot-frame trajectories. This package does
not choose one sample or waypoint and does not apply the old LoCoBot deployment
factor `MAX_V / RATE`. Those decisions belong to the future robot integration
and safety layer.

Set `goal_mask=True` only for goal-masked exploration. Goal-conditioned
navigation uses the default `goal_mask=False`.

## Image Input

The image ingress starts after camera decoding. It accepts one three-dimensional
PyTorch tensor at a time and deliberately does not depend on OpenCV, NumPy,
Pillow, ROS, or a specific camera SDK.

Supported decoded representations are:

- layout: `CHW` or `HWC`;
- channel order: `RGB` or `BGR`;
- dtype: `torch.uint8` or any floating-point dtype;
- value range: byte `[0, 255]` or unit `[0, 1]`.

`uint8` is interpreted as byte range by default. Floating-point data is
interpreted as unit range by default; set `value_range="byte"` explicitly for
floating-point `[0, 255]` images.

```python
from nomad_runtime import (
    ImageTensorSpec,
    ObservationBuffer,
    canonicalize_image,
)


camera_spec = ImageTensorSpec(
    layout="hwc",
    channel_order="bgr",
)
observation_buffer = ObservationBuffer(context_frames=4)

# `decoded_frame` is a torch.uint8 [H, W, 3] tensor from the camera layer.
observation_buffer.append(
    decoded_frame,
    timestamp_s=camera_timestamp_s,
    spec=camera_spec,
)

if observation_buffer.ready:
    context = observation_buffer.snapshot(
        now_s=current_camera_clock_s,
        max_age_s=0.2,
    )
    observations = context.images  # [4, 3, H, W], oldest to newest

# A goal image goes through the same conversion but does not enter the buffer.
goal = canonicalize_image(decoded_goal, camera_spec)  # [3, H, W]
```

Timestamps must be finite and strictly increasing. The buffer retains its own
copy of each frame so a camera backend may safely reuse capture memory. A
resolution change is rejected until `observation_buffer.clear()` is called,
which prevents a mixed-resolution context after camera restart or reconfigure.

`NomadConfig(center_crop_aspect=4.0 / 3.0)` optionally applies a centered 4:3
crop before the configured `96 x 96` resize. The crop setting is part of the
image profile: observations, Map goal resources, offline edge scoring, and
localization must all use the same value. Existing direct-resize maps must not
be mixed with center-cropped observations.

The caller owns camera acquisition and decoding. In particular, this package
does not decide capture rate, convert compressed JPEG/H.264 bytes, or choose a
camera transport. Those are separate image-source concerns rather than model
inference concerns.

## Adaptive Offline Topomap

The optional offline tool converts a timestamped dense image sequence into a
sequential NoMaD-aware topomap. It uses only the deterministic distance head;
diffusion samples do not influence node selection.

For each selected source node, it searches future images within a bounded time
window. It selects the farthest candidate whose predicted distance is in the
preferred range. If none exists, it chooses the candidate closest to the target
distance inside the hard range. If quality filtering would force an edge
outside the hard range, it relaxes the image-quality threshold before accepting
the unsafe edge. Every relaxation and out-of-range fallback is recorded.

Defaults:

```text
candidate time gap:  0.8 to 4.0 seconds
preferred distance:  6 to 12
hard distance range: 3 to 15
target distance:      9
```

Install the offline image dependency and run:

```bash
python -m pip install -e \
  './plugins/policies/visual_navigation/nomad[offline]'

python -m nomad_runtime.adaptive_topomap \
  --images /path/to/dense_topomap \
  --checkpoint /path/to/nomad.pth \
  --output /path/to/adaptive_topomap \
  --device cuda:0
```

The input directory should contain numerically named PNG/JPEG images and a
`manifest.json` list with `filename` and `time_s` fields. Without a manifest,
pass `--frame-period-s` for fixed-rate input.

The output layout is:

```text
adaptive_topomap/
├── images/          # numeric images only; pass this to official navigation
├── manifest.json    # selected source frames and complete configuration
├── edges.json       # distances, candidates, fallbacks, and directed edges
└── summary.json     # node compression and edge distributions
```

The command refuses to write into a non-empty output directory. Use
`--center-crop-aspect 4:3` only when online robot observations use the identical
crop. Forward and reverse traversal require independently qualified directed
edges; a forward recording is not assumed to be reversible.

## Longship Map Engine Adapter

The repository also contains a harness-side adapter in `longship_adapter`.
It is separate from `nomad_runtime`: it reads an already generated adaptive
topomap and exposes one pinned, read-only Longship `MapEngine`. It does not
decode images or import PyTorch.

The caller must provide the publication identity and model compatibility data
that the current topomap format does not store:

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from longship.navigation.common import TimePoint
from longship.navigation.map_engine.models import MapId, MapVersion
from plugins.policies.visual_navigation.nomad.longship_adapter import (
    NomadTopomapMapConfig,
    create_nomad_topomap_engine,
)


map_engine = create_nomad_topomap_engine(
    NomadTopomapMapConfig(
        root=Path("/path/to/adaptive_topomap"),
        map_id=MapId("recorded-forward-route"),
        version=MapVersion("v0.1"),
        published_at=TimePoint(clock_id="unix", nanoseconds=0),
        model_artifact_id="nomad.pth",
        model_artifact_digest="<checkpoint-sha256>",
    )
)
```

Each selected image becomes an opaque image resource and a visual anchor on a
topology node. Each row in `edges.json` becomes one directed segment. The
adapter records center/minimum/maximum offline distance diagnostics separately
from hardware qualification; all imported segments remain
`hardware-unqualified` in this draft.

## Localization Engine Adapter

The localization path is split into five explicit layers:

```text
DecodedObservationSource
    owns capture or replay decoding and source cadence outside model runtime

longship_adapter.NomadObservationProducer
    validates profile/order and samples decoded frames on a fixed time grid

nomad_runtime.NomadDistanceSession
    owns frame history, four-frame context, preprocessing, and batched inference

longship_adapter.NomadVisualGoalDistancePolicy
    validates Map resource/profile/model identity and translates failures

FixedStartVisualLocalizationEngine
    owns start verification, local evidence, monotonic recovery, and LocationBelief
```

The provisional localization engine is hard-coded to start at the Map adapter's
canonical node zero, `node-0000`. It rejects maps whose directed chain does not
start there; callers cannot supply another initial node.

After start verification, each tick compares one observation context with the
current node, expected successor, and one look-ahead node in a single batch.
The engine advances by at most one node when the successor is absolutely close,
or when it repeatedly wins the local comparison by a bounded margin. Evidence
is retained in a short window rather than treated as an isolated threshold
crossing.

If a later look-ahead node is repeatedly close, the engine does not silently
skip the expected successor. It publishes `LOST`, expands to a bounded forward
window, and requires repeated close agreement before restoring `TRACKING` at a
monotonically later node. This is local route recovery, not arbitrary-start or
global relocalization. Persistently untrusted local candidates also enter the
same nonterminal recovery state.

`NomadDistanceSession` deliberately calls only `encode_condition()` and
`predict_distance()`. It batches all local goal images against one repeated
observation context. It never calls `infer()` or `sample_actions()` and never
creates a motion candidate.

Runtime composition is explicit:

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from longship.navigation.localization_engine.service import (
    ContinuousLocalizationService,
    LocalizationServiceConfig,
    MonotonicTimeSource,
)
from longship.navigation.runtime import LocalizationRuntime
from nomad_runtime import NomadDistanceSession, NomadPolicy

from plugins.policies.visual_navigation.nomad.longship_adapter import (
    LocalFileGoalImageLoader,
    NomadObservationProducer,
    NomadObservationProducerConfig,
    NomadVisualGoalDistancePolicy,
    NomadVisualPolicyConfig,
)


model = NomadPolicy.from_checkpoint(
    checkpoint_path,
    device="cuda:0",
    strict=True,
)
session = NomadDistanceSession(model)
nomad_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="nomad-policy",
)
distance_policy = NomadVisualGoalDistancePolicy(
    session=session,
    goal_image_loader=LocalFileGoalImageLoader(
        allowed_roots=(Path(topomap_root),),
    ),
    inference_executor=nomad_executor,
    config=NomadVisualPolicyConfig(
        policy_id="nomad-distance-v1",
        image_profile_id="nomad.rgb.direct_resize_96x96.imagenet.v1",
        model_artifact_id="nomad.pth",
        model_artifact_digest=checkpoint_sha256,
        observation_clock_id="camera",
    ),
)
observation_producer = NomadObservationProducer(
    source=decoded_observation_source,
    policy=distance_policy,
    config=NomadObservationProducerConfig(
        image_profile_id="nomad.rgb.direct_resize_96x96.imagenet.v1",
        sample_hz=9.0,
    ),
)
```

After constructing `FixedStartVisualLocalizationEngine`, the deployment
Composition Root injects its observation producer and executor resource into
the plugin-neutral Runtime:

```python
localization_clock = MonotonicTimeSource(clock_id="camera")
localization_service = ContinuousLocalizationService(
    engine=localization_engine,
    time_source=localization_clock,
    config=LocalizationServiceConfig(tick_period_s=1.0 / 9.0),
)
localization_runtime = LocalizationRuntime(
    observation_producer=observation_producer,
    localization_service=localization_service,
    shutdown_resources=(nomad_executor_resource,),
)
await localization_runtime.start()
```

`decoded_observation_source` and `nomad_executor_resource` are
deployment-owned implementations. `NomadObservationProducer` is the narrow
NoMaD adapter between that decoded source and the plugin policy ingress. None
of them belongs to the public Localization Engine Facade.

For the D435i ROS 2 deployment, use
`tools.ros2_image_source.Ros2ImageFrameSource` as the decoded source. It
subscribes to `/camera/camera/color/image_raw` with best-effort QoS and depth
one, so it consumes the newest RGB frame without opening `/dev/video*` or
owning the camera. Its policy timestamp uses local `time.monotonic()` and its
separate source timestamp preserves `sensor_msgs/Image.header.stamp` for
sampling and gap detection. The injected `TimeSource` for both the policy and
localization service must use the same monotonic clock domain. The ROS source
accepts `rgb8` and `bgr8` images only; the D435i driver is configured to
publish `rgb8`.

The robot's combined PyTorch and ROS 2 container must source Jazzy before it
starts a live NoMaD process. For the current deployment, use:

```bash
source /opt/ros/jazzy/setup.bash
cd /workspace/longship
export PYTHONPATH="$PWD/src:$PWD/plugins/policies/visual_navigation/nomad:$PWD:$PYTHONPATH"
```

This keeps the ROS 2 Python packages and the container's PyTorch packages on
the same Python 3.12 import path.

The source returns already decoded tensors and timestamps from the same
`camera` clock domain. The producer rejects profile or ordering changes,
clears context after a source gap, and samples with a fixed 9 Hz time grid.
The Localization supervisor independently calls the fixed-start engine's
internal `tick()` at 9 Hz. Steady beliefs are publication-throttled to 4 Hz;
state and status transitions publish immediately. Missions do neither; they
only consume the public `LocationBelief` stream.

On shutdown, `await localization_runtime.stop()` first stops camera
submissions, then stops policy ticks, and finally closes the injected executor
resource. A single worker makes inference ordering and resource use explicit
while keeping model execution off the event loop.

Runtime monitors `NomadObservationProducer.wait_stopped()` independently from
the localization service. A camera-source fault or unexpected completion
therefore faults and cleans up the whole Runtime. The dense and video tools are
finite sources and explicitly select `ALLOW_UNTIL_STOP`, so EOF can be followed
by a bounded final-belief grace period before their Composition Root stops the
Runtime.

`LocalFileGoalImageLoader` is one locator-specific resource adapter for offline
and local deployments. It permits only configured roots and verifies the
resource size and SHA-256 digest before decoding. Object-store or service
locators require separate `GoalImageLoader` implementations.

### Fixed-start offline replay

The integration replay tool runs three asynchronous roles: a timestamped dense
image producer, `ContinuousLocalizationService`, and a `LocationBelief`
collector. It prints only state transitions and can also write one JSONL trace
row per consumed dense frame:

```bash
PYTHONPATH=src:plugins/policies/visual_navigation/nomad:. \
python plugins/policies/visual_navigation/nomad/tools/\
replay_fixed_start_localization.py \
  --checkpoint /path/to/nomad.pth \
  --dense-images /path/to/dense_topomap \
  --topomap /path/to/adaptive_topomap \
  --device cuda:0 \
  --trace-output /tmp/nomad-localization-trace.jsonl
```

The replay invokes only the NoMaD distance head. It does not sample actions or
send motion commands. `LOST` is recoverable, so replay continues until final
arrival, a fault, or source completion. Threshold, evidence-window, local
candidate, recovery-window, tick-period, and frame-timeout command-line
overrides are diagnostic controls; omitting them uses the engine defaults.

### Fixed-start recorded-video mock

The recorded-video composition uses FFmpeg only as an offline decoded-frame
source. It is not a production camera driver and remains under `tools/`. The
source preserves the recorded 30 Hz cadence, `NomadObservationProducer`
samples it at 9 Hz, and `ContinuousLocalizationService` also ticks at 9 Hz:

```bash
PYTHONPATH=src:plugins/policies/visual_navigation/nomad:. \
python plugins/policies/visual_navigation/nomad/tools/\
run_video_mock_localization.py \
  --checkpoint /path/to/nomad.pth \
  --video /path/to/recorded-route.mp4 \
  --topomap /path/to/adaptive_topomap \
  --device cuda:0 \
  --start-time-s 6.0 \
  --trace-output /tmp/nomad-video-localization.jsonl
```

The 2026-08-19 source-video replay reached `node-0039` and finished in
`AT_FINAL_NODE`. It advanced monotonically, detected weak evidence for
`node-0033`, and recovered through the bounded forward window at `node-0034`.
The trace records both producer progress and the exact observation timestamp
used by each published belief. This validates the asynchronous mock against its
map-recording traversal; it is not an independent-traversal or hardware-motion
qualification.

### Recorded-video trajectory diagnostic

`NomadTrajectorySession` shares the same four-frame input contract but also
samples every raw diffusion trajectory. The Longship adapter binds each result
to the immutable Map snapshot, directed segment, source and target nodes,
visual target anchor, goal resource, observation timestamp, model digest, and
sampling seed. The optional contract lives under
`longship.navigation.ports.trajectory_policy`; it is not part of the
mission-facing `RouteExecutionPort`.

The diagnostic renderer consumes a localization JSONL trace and the original
video. It preserves the 30 FPS source video, updates NoMaD inference at 4 Hz,
and holds only the latest route-step-consistent result between updates:

```bash
PYTHONPATH=src:plugins/policies/visual_navigation/nomad:. \
python -m plugins.policies.visual_navigation.nomad.tools.\
render_video_trajectory_overlay \
  --checkpoint /path/to/nomad.pth \
  --video /path/to/recorded-route.mp4 \
  --topomap /path/to/adaptive-topomap \
  --localization-trace /tmp/nomad-video-localization.jsonl \
  --output /tmp/nomad-trajectory-overlay.mp4 \
  --trajectory-jsonl /tmp/nomad-trajectory-overlay.jsonl \
  --device cuda:0 \
  --start-time-s 6.0 \
  --end-time-s 104.8 \
  --inference-hz 4 \
  --num-candidates 4
```

Every output video frame contains localization state. Once the fixed start is
confirmed, it also contains the current Map goal image and a fixed-range
robot-frame bird's-eye plot of all four candidate trajectories. The tool does
not perspective-project policy coordinates into camera pixels because the map
does not publish the required camera calibration. It does not choose a sample,
scale coordinates, or create a command. A target-node transition hides the old
result until fresh inference for the new route step completes; final arrival
shows `ARRIVED / HOLD` and no trajectory.

### RoutePlan-to-trajectory Harness mock

`run_video_mock_route_trajectory.py` is the first complete Harness composition.
It loads one immutable Map snapshot, starts continuous fixed-start localization,
waits for a verified `node-0000`, asks `TopologicalPlanningEngine` for a
`RoutePlan` to the map completion anchor, and then starts
`RouteBoundLocalTrajectoryEngine`. Each new localization belief drives one
non-overlapping trajectory update, so localization retains priority on the
shared inference executor.

One decoded-frame producer fans identical accepted frames into separate NoMaD
distance and trajectory contexts. Both sessions share one CUDA model and a
single-thread inference executor. The public output is
`LocalTrajectoryStream`; each `ACTIVE` row contains candidate `0` with all 8
waypoints plus route, map, belief, target, observation, validity, policy, model,
and sampling identities:

```bash
PYTHONPATH=src:plugins/policies/visual_navigation/nomad:. \
python plugins/policies/visual_navigation/nomad/tools/\
run_video_mock_route_trajectory.py \
  --checkpoint /path/to/nomad.pth \
  --video /path/to/recorded-route.mp4 \
  --topomap /path/to/adaptive-topomap \
  --device cuda:0 \
  --start-time-s 6.0 \
  --route-plan-output /tmp/nomad-route-plan.json \
  --trajectory-output /tmp/nomad-local-trajectories.jsonl
```

The JSONL writer is an example consumer of the same read-only stream intended
for integration outside the Harness. `HOLDING`, `ROUTE_COMPLETED`, `FAULTED`,
or `STOPPED` rows never contain a trajectory and invalidate an older proposal.
No row is a controller or chassis command.

For recorded video, observation timestamps and policy requests use the source
video clock; wall time only paces playback. With batched local localization,
the 2026-08-19 CUDA replay produced the 39-traversal RoutePlan, published 297
active sample-0 trajectories covering targets `node-0001` through
`node-0039`, emitted five explicit `HOLDING` updates for degraded localization,
and ended with `ROUTE_COMPLETED`. Every active trajectory contained all 8
waypoints and a unique trajectory id. This validates the recorded-map mock path
only, not independent traversal or hardware motion.

## Smoke Test

Run a strict checkpoint load and random-tensor inference:

```bash
python -m nomad_runtime.smoke_test \
  --checkpoint /path/to/nomad.pth \
  --device cuda:0 \
  --num-samples 4
```

The command reports the model parameter count, output shapes, predicted random
input distance, and whether every sampled action is finite.

## Tests

Run the dependency-free unit tests from this directory:

```bash
pytest -q
```

To include the real checkpoint compatibility test:

```bash
NOMAD_CHECKPOINT=/path/to/nomad.pth pytest -q
```

The implementation is checkpoint- and numerically compatible with the local
`visualnav-transformer` reference using diffusers 0.11.1. The vision condition,
distance output, noise prediction, and full DDPM chain were verified with zero
maximum absolute error. Released-checkpoint loading and inference are also
validated on Python 3.11.12 with PyTorch 2.6.0.
