# FollowPerson V0: closed loop, simulation, and supervised G1 deployment

> **Status:** Independently authored experimental vertical slice. The public
> simulation is reproducible; the RealSense and Unitree paths are integration
> scaffolding and are not hardware-qualified. Use a gantry, an independent
> physical E-stop, a separate motion-state monitor, and a trained operator.

## What is implemented

FollowPerson is integrated along Longship's existing responsibility boundaries:

```text
RealSense provider or synthetic world
  -> versioned FollowScene (one timestamp and sequence)
  -> navigation.follow_person local provider
       short track lock
       fresh base-local occupancy grid and route
       bounded last-seen prediction and nearby-ID reacquisition
  -> FollowPerson Runtime
       state, lease, TTL, pause/stop, recovery budget, event journal
  -> independent raw-clearance guard
  -> mock target or Unitree G1 high-level target adapter
  -> measured world change -> next FollowScene

Runtime events -> JSONL and read-only dashboard (never a command path)
```

The Runtime coordinates the Skill and owns actuator authority; it does not own
person detection or path-search mathematics. The local planner has no invented
global pose. It rebuilds a small route in the instantaneous robot base frame
from each synchronized scene. Only the target adapter translates the approved,
expiring base command to the vendor SDK.

The independent obstacle guard uses the raw forward-corridor distance rather
than the occupancy route with the selected person removed. It applies a dynamic
stop threshold from current speed, configured deceleration, system latency, and
margin. Missing, stale, out-of-order, uncalibrated, or floor-invalid perception
commands zero immediately.

## Runtime lifecycle and recovery

- `acquiring`: select the eligible, best-centred person and lock its short ID.
- `following`: maintain the configured standoff through a base-local route.
- `holding_for_scene`: immediately command zero during a brief camera or floor
  failure; return to the same session only inside the configured grace window.
- `approaching_last_seen`: update the last relative position from the command
  actually accepted, approach it for a bounded time, and accept the old ID or a
  new ID only inside the position gate.
- `blocked`: command zero while the local route or raw guard is blocked.
- `paused`: keep the session but command zero; resume requires a fresh target.
- `failed`: request the target stop path after a sustained scene failure,
  recovery timeout, persistent block, or rejected motion command. Restart then
  requires an operator and a new process/lease.

Reaching the last-seen vicinity without reacquisition is not success. Runtime
stops and asks for supervised recovery. An SDK zero-velocity acknowledgement is
also not called a verified stop; only correlated evidence from a separately
qualified monitor can establish that state.

## 1. Install and run all local checks

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

No camera, vendor SDK, or network is needed for these tests.

## 2. Run end-to-end system simulation

The primary offline gate begins at a mock wake/final transcript event, sends
the instruction through a bounded deterministic Brain, admits the semantic
`navigation.follow_person` Skill, closes the control/world loop, and finishes
through the partial-input STOP path that bypasses Brain:

```bash
longship-follow system-simulate \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --instruction 'Jackie，跟着我走' \
  --events /tmp/longship-follow-system.jsonl
```

The report must contain all six stages: `input`, `brain`, `skill`, `runtime`,
`safety`, and `target`. This uses deterministic input and Brain providers; it
does not claim acoustic ASR or a production model provider has been evaluated.

### Interactive terminal composition

Longship's terminal is an input provider for the same `RuntimeTextPort` used by
the wake/dictation controller. It is not a second command or actuator path.
Run a persistent synthetic stack:

```bash
longship-follow stack \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --events /tmp/longship-follow-stack.jsonl \
  --dashboard-port 8093
```

Enter `Jackie，跟着我走`, `状态`, `暂停`, `继续`, and `停止`. Ordinary text
goes through `FollowBrainPort`; fixed controls are classified by Runtime and
STOP can overtake an outstanding Brain request. `exit`, terminal EOF, or
target closure also stops an active Skill before process teardown.

With the Codex provider, a temporal instruction such as
`跟我走三秒然后暂停，一秒后继续走` is compiled into a Longship
MissionTaskGraph. The terminal returns immediately after admission. Runtime,
not Codex, advances the `follow(3s) -> pause(1s) -> resume` nodes from the
control-loop clock, so `状态`, the read-only HUD, and protected STOP stay
responsive throughout execution. The current executable graph slice is one
linear FollowPerson lane; parallel nodes, barriers, and graph patches remain
future Runtime work rather than plugin-specific behavior.

Pause keeps the existing locomotion policy, motion lease, and control loop
alive while issuing a zero planar-velocity target on every tick. It does not
freeze joints, switch controllers, reinitialize the policy, or use protected
STOP. This minimizes action-switch latency and transient-stability risk. A
timed pause starts when the zero-velocity node is admitted; it does not wait for
measured base stationarity before starting its timer. An actively balancing G1
may still brake or advance its gait phase during a short pause. Protected STOP
is the distinct path that performs a measured stationary-base dwell.

The default `--brain deterministic` is the reproducible offline provider. To
exercise the optional model-backed semantic provider:

```bash
pip install -e '.[codex]'
longship-follow stack \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --brain codex \
  --codex-model gpt-5.6-terra \
  --codex-reasoning-effort none \
  --codex-timeout-s 60
```

The Codex output schema can only respond or propose
`navigation.follow_person` with bounded `follow`, `pause`, and `resume` steps;
Runtime binds the proposal to its current revision and rejects stale, invalid,
or unknown actions. A scheduled STOP is not allowed. This tests Brain
integration, not model quality or network availability. See the
[interaction stack composition](../architecture/interaction-stack.md).

For a faster control-only loop and read-only top-down dashboard:

```bash
longship-follow simulate \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --events /tmp/longship-follow-simulation.jsonl \
  --real-time \
  --dashboard-port 8093 \
  --keep-dashboard
```

Open `http://127.0.0.1:8093`. The synthetic person's world motion and the
simulated robot motion both feed the next observation, so this is a real
software closed loop rather than a canned command replay. The default scenario
checks detouring, acceleration limiting, temporary occlusion, a changed track
ID, raw-clearance stopping, final standoff, and event generation. A nonzero
exit means an acceptance gate failed.

Before any hardware trial, add site-specific scenarios for narrow corridors,
glass/low-reflectivity obstacles, detector loss, network delay, floor-fit loss,
heartbeat expiry, and the site's exact speed profile. Simulation evidence never
qualifies the physical target by itself.

## 3. Run the MuJoCo physics target

The optional target plugin feeds the same scenario through MuJoCo motion and
collision contacts. Install it separately so the portable Runtime remains free
of simulator dependencies:

```bash
pip install -e '.[mujoco]'
python3 plugins/targets/mujoco_follow_person/runner.py \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --system --viewer --keep-viewer
```

For a live terminal beside the viewer, use the same plugin as a target while
keeping interaction and mission authority in Longship:

```bash
python3 plugins/targets/mujoco_follow_person/runner.py \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --stack --viewer
```

Add `--brain codex` to that command only when the optional provider is
installed and model-backed behavior is part of the test scope.

Blue is the planar robot proxy, green is the scripted person, and red bodies
are physical obstacles. Close the viewer to exit after the acceptance report.
For CI or SSH use, omit `--viewer --keep-viewer`; add `--report` and `--events`
with new paths to retain evidence.

This gate checks the target-independent base loop against physical velocity
response and contact detection. The proxy is intentionally not a G1 model: it
does not validate humanoid balance, joints, locomotion policy, rendered-camera
perception, or sim-to-real behavior. Those require a separately licensed G1
model and policy target behind the same adapter boundary.

### External 29-DOF G1 asset-only gate

The `mujoco_g1_external` seam validates an externally installed 29-DOF Unitree
G1 bundle without copying it. For the locally inspected bundle:

```bash
export UNITREE_MUJOCO_ROOT=/absolute/path/to/unitree_mujoco
export LONGSHIP_G1_PYTHON=/absolute/path/to/python-with-mujoco

python3 plugins/targets/mujoco_g1_external/doctor.py \
  --scene "$UNITREE_MUJOCO_ROOT/unitree_robots/g1/scene_29dof.xml" \
  --license "$UNITREE_MUJOCO_ROOT/LICENSE" \
  --expected-bundle-sha256 9ba04edacbaf9bda13bf847e99e845e9b36b27f7b2141e48ccfc8cae211d1f39 \
  --expected-license-sha256 a5d73fc4aca9074e3e6fe0b1a0ba763cf9514b2249b7390ed20fe8d53630bf25
```

The result for this asset-only plugin is `asset_ready=true` and
`dynamic_follow_ready=false`.
The model can also be viewed, without stepping or actuating it:

```bash
"$LONGSHIP_G1_PYTHON" \
  plugins/targets/mujoco_g1_external/viewer.py \
  --scene "$UNITREE_MUJOCO_ROOT/unitree_robots/g1/scene_29dof.xml"
```

A G1 MJCF supplies robot geometry, joints, inertias, collision, sensors, and
actuators—not balance. The inspected official 29-DOF simulator exposes
low-level `LowCmd`/`LowState`; Longship's current physical adapter uses the
onboard high-level `LocoClient`. Dynamic activation of that specific 29-DOF
DDS target therefore remains blocked until an independently implemented and
qualified low-level provider supplies policy control, fresh state, stop
evidence, fall/contact criteria, and replayable tests. The detailed gate is in
the [external asset plugin guide](../../plugins/targets/mujoco_g1_external/README.md).

### Dynamic external G1 policy target

The separate `mujoco_g1_policy` plugin closes the system loop on the externally
installed Unitree RL Gym G1 model and TorchScript locomotion policy. This is a
real free-base articulated G1 with 12 actuated lower-body joints. It is not a
conversion of the 29-DOF DDS target and does not alter Longship's architecture:
only the target/world provider changes.

Following the governed-policy merge, the plugin uses the shared
`ModelArtifactManifest`, `ArtifactStore`, `PolicyRequest`, `PolicyCandidate`,
and deterministic candidate guard. This integrates artifact identity, model
binding, lease epoch, observation version, action shape/bounds, and 20 ms
freshness into the actual physics loop. It does not make the 12-joint
TorchScript compatible with the separately added 29-joint MJLab or Holosoma
contracts; those need a matching 29-DoF simulator target and qualification.

For the exact locally inspected artifact set, run the full headless acceptance:

```bash
UNITREE_RL_GYM=/absolute/path/to/unitree_rl_gym
HSMUJOCO_PYTHON=/absolute/path/to/python-with-mujoco

PYTHONPATH=src "$HSMUJOCO_PYTHON" \
  plugins/targets/mujoco_g1_policy/runner.py \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --scene "$UNITREE_RL_GYM/resources/robots/g1_description/scene.xml" \
  --scene-bundle-root "$UNITREE_RL_GYM/resources/robots/g1_description" \
  --policy "$UNITREE_RL_GYM/deploy/pre_train/g1/motion.pt" \
  --policy-config "$UNITREE_RL_GYM/deploy/deploy_mujoco/configs/g1.yaml" \
  --license "$UNITREE_RL_GYM/LICENSE" \
  --expected-scene-bundle-sha256 f569b1425fc055ca759699f36f94eba97663db547b79e663bafa50560a0c9349 \
  --expected-policy-sha256 cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d \
  --expected-config-sha256 73044e7d355c61915695c16d6e09eb3efef46eec1e3d708fd3eb9157dfe3bbbb \
  --expected-license-sha256 aef6394ba1597725a68308167324e675f562e6606027404deb1b9da254c2b9c1 \
  --mode system
```

Use `--mode system --viewer --real-time --keep-viewer` to watch the same run.
Use `--mode stack --viewer` for Longship's interactive terminal; enter
`跟着我走`, `状态`, `暂停`, `继续`, or `停止`. The terminal remains an input
provider for `RuntimeTextPort`: semantic requests go through Brain and Skill
admission, while fixed controls and STOP bypass Brain.

For the G1-mounted camera and environment HUD, prefer the locked convenience
launcher:

```bash
cd /absolute/path/to/longship
export UNITREE_RL_GYM_ROOT=/absolute/path/to/unitree_rl_gym
export LONGSHIP_G1_PYTHON=/absolute/path/to/python-with-mujoco

MUJOCO_GL=egl bash plugins/targets/mujoco_g1_policy/run_inspected.sh \
  --mode stack \
  --hud-port 8093 \
  --keep-hud
```

Open `http://127.0.0.1:8093`. The page is read-only and shows the simulated G1
camera, person, obstacles, base-local route/next goal, person/follow-goal world
positions, command, clearance, base pose/height/tilt, contacts, policy steps,
Brain events, Skill identity, and the active task graph/node. The launching
terminal remains the input surface. This command is intentionally headless, so
it does not open the native MuJoCo window.

With a desktop display, open the native MuJoCo Viewer and the HUD together by
removing `MUJOCO_GL=egl` and using the complete command below:

```bash
bash plugins/targets/mujoco_g1_policy/run_inspected.sh \
  --mode stack \
  --viewer \
  --hud-port 8093 \
  --keep-viewer \
  --keep-hud
```

Do not combine `MUJOCO_GL=egl` with `--viewer`. After entering `停止`, close
the retained Viewer and press Enter in the launching terminal when the HUD is
no longer needed.

This policy is loaded from external Unitree RL Gym artifacts; it is not
Holosoma. To use the current Codex login as the experimental semantic Brain,
install the official SDK into the G1 Python environment once and add the
documented flags:

```bash
bash plugins/targets/mujoco_g1_policy/install_codex_brain.sh

bash plugins/targets/mujoco_g1_policy/run_inspected.sh \
  --mode stack \
  --viewer \
  --hud-port 8093 \
  --keep-viewer \
  --keep-hud \
  --events /tmp/longship-g1-events.jsonl \
  --brain codex \
  --codex-model gpt-5.6-terra \
  --codex-reasoning-effort none \
  --codex-timeout-s 60
```

The command above is the desktop Viewer + HUD + Codex form. On a headless
machine, remove `--viewer --keep-viewer` and prefix it with `MUJOCO_GL=egl`.
The event journal is created exclusively, so choose a new `--events` path for
each run. Inspect the timed task and zero-velocity pause afterward with:

```bash
rg 'task_graph.transitioned|"state":"paused"|desired_velocity' \
  /tmp/longship-g1-events.jsonl
```

Longship calls the local Codex SDK/app-server directly; an MCP bridge is not
needed for this direction. Codex may only propose the semantic FollowPerson
Skill and its bounded task draft or respond with text. Runtime compiles and
asynchronously advances that draft. Codex cannot emit motion or Safety actions,
and the physical deployment command does not enable the experimental model
provider.

The runner makes no third-party copy. It hashes the whole external G1 asset
directory and verifies the policy, configuration, and license through the
common artifact store before loading them. Its report
requires all six pipeline stages, following acceptance, no G1 fall, no barrier
contact, guarded policy candidates, and measured stationary-base evidence
after STOP. Perception is still
scripted ground truth, and the 12-joint simulation does not establish 29-DOF
DDS parity or physical-robot readiness. See the
[dynamic target guide](../../plugins/targets/mujoco_g1_policy/README.md) for
the provenance record and limitations.

The merged `unitree_rl_lab` directory describes an external training/export
family but remains reference-only. The MJLab and Holosoma additions implement
runtime policy contracts and synthetic tests, not a vendored or currently
runnable training simulator. Use their plugin manifests as future integration
gates rather than treating them as another launch mode of this target.

The exact coverage and remaining reference/deployment gaps are tracked in the
[FollowPerson parity and readiness matrix](follow-person-parity-readiness.md).

## 4. Prepare and validate the RGB-D provider

Install `pyrealsense2`, OpenCV, and NumPy in one reviewed on-robot Python
environment. Do not use the disabled example transform as a real calibration.
Survey the D435 optical frame relative to `base_link`, write the row-major 4x4
rigid transform in `longship.camera-extrinsic.v0`, and validate at minimum:

1. the transform is right-handed and its axes match forward/left/up;
2. a floor sample maps to approximately zero base height across the useful view;
3. known targets at multiple ranges and lateral offsets agree with tape or a
   surveyed fixture;
4. the camera remains rigid under commanded starts, stops, and turns; and
5. the expected operator stays fully visible in the planned corridor.

Only then set `confirmed` to `true` in the site-owned calibration artifact.
Start the single camera owner:

```bash
python3 plugins/perception/realsense_rgbd_follow/worker.py \
  --calibration /absolute/path/to/reviewed-camera-extrinsic.json \
  --host 127.0.0.1 \
  --port 8780
```

Inspect `http://127.0.0.1:8780/` while the robot is stationary. Verify stable
boxes/IDs, plausible person coordinates, floor health, camera age, supported
obstacles, and raw clearance. Then run the non-actuating preflight:

```bash
longship-follow probe --perception-url http://127.0.0.1:8780
```

The probe must report healthy calibration, detector, floor, and raw-obstacle
telemetry. A preview browser can be slow or disconnected without queueing work
in the control loop. The page is unauthenticated, so keep it on loopback or use
an SSH tunnel on untrusted networks.

Create a site-owned qualification record from
`scenarios/follow_person/qualification.g1.example.json`. Bind it to the exact
profile file digest, target identity, and `calibration_id` returned by the
provider; attach immutable simulation, camera, gantry, stop, and site-clearance
evidence; set an expiry and maximum session duration; and have the accountable
reviewer approve it. The checked-in example is deliberately unapproved and
cannot arm hardware.

## 5. Supervised gantry deployment

Complete the robot's normal doctor procedure, confirm the independent physical
E-stop, start a separately qualified robot-state/motion monitor, clear the test
area, fit the gantry, and keep a second operator at the E-stop. The software
heartbeat below is an additional deadman; it is not a safety-rated device.

Terminal A — keep pressing Enter inside the configured timeout:

```bash
longship-follow heartbeat --file /tmp/longship-follow-heartbeat.json
```

Terminal B — after the perception probe and all site checks pass:

```bash
longship-follow deploy \
  --profile scenarios/follow_person/profile.v0.json \
  --qualification /absolute/path/to/reviewed-follow-qualification.json \
  --perception-url http://127.0.0.1:8780 \
  --interface <robot-network-interface> \
  --target-id <robot-identity> \
  --boot-id <current-robot-boot-identity> \
  --heartbeat-file /tmp/longship-follow-heartbeat.json \
  --events /tmp/longship-follow-g1.jsonl \
  --dashboard-host 127.0.0.1 \
  --dashboard-port 8093 \
  --maximum-runtime-s 120 \
  --hardware-enable-token SUPERVISED-GANTRY-ONLY \
  --doctor-passed \
  --physical-estop-verified \
  --camera-calibration-verified
```

The command refuses hardware unless every explicit gate is present, the manual
heartbeat is fresh, the camera preflight passes, and the separately installed
official Unitree SDK can initialize. Velocity remains bounded by the selected
profile; each command expires; the process-local whole-body lease is refreshed
without changing owner identity; and missed heartbeat, stale perception,
planner block, TTL failure, SDK rejection, Ctrl-C, or maximum runtime all enter
the stop path.

The Runtime dashboard at `http://127.0.0.1:8093` is read-only. It displays the
locked person, supported obstacle cells, planned local path, raw clearance,
command, state, and recent events. It cannot pause, resume, arm, or stop the
robot, and its failure cannot delay the control loop.

Event journals use exclusive creation so an earlier trial cannot be silently
overwritten. Choose a new `--events` path for every run and retain it with the
qualification evidence.

After stopping, do not approach the robot merely because the SDK accepted zero
velocity. The CLI deliberately exits with `STOP UNVERIFIED` (normally exit code
6) until a separately qualified monitor supplies measured stationary evidence.
Keep the physical E-stop engaged and follow the target's reviewed stand-down
procedure.

## Current limitations

- HOG plus IoU tracking is a baseline, not identity recognition. Do not use it
  in crowds, across long occlusion, or around visually similar people.
- The provider currently aligns depth to the RGB field of view. A target-
  qualified wider-depth obstacle provider may replace it through the same scene
  contract without changing Runtime.
- Floor validity is a calibrated geometric support check, not a learned terrain
  classifier. Stairs, ramps, mirrors, glass, sunlight, and low-reflectivity
  surfaces require separate evaluation.
- Local A* cannot open doors, build a global map, or guarantee escape from a
  concave environment.
- The high-level Unitree client and host process are not real-time or
  safety-rated. The physical E-stop and independent state monitor remain
  mandatory.
