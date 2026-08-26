# FollowPerson parity and readiness matrix

> **Purpose:** State exactly what Longship independently reproduces from the
> behavioral reference, what the system simulations prove, and what remains
> before an existing G1 deployment can be called equivalent. This is an
> engineering readiness record, not legal or safety certification.

## Qualification levels

| Level | Meaning | Current status |
| --- | --- | --- |
| L0 — contract/unit | Typed contracts, strict configuration, planner, Runtime, Safety, and target fakes | PASS |
| L1 — control closed loop | Synthetic person/robot world feeds every accepted command into the next observation | PASS |
| L2 — system closed loop | Input event or live terminal → Brain proposal → Runtime admission → FollowPerson Skill → Safety → target → observation → protected STOP | PASS with deterministic providers |
| L3 — MuJoCo system loop | The L2 chain drives a visible planar physics target and rejects physical obstacle contact | PASS |
| L3-G1 — articulated G1 loop | L2 drives a manifest-verified, candidate-guarded, free-base 12-joint Unitree RL Gym G1 policy; checks fall, contact, tracking, and measured base stop | PASS for the documented local external artifacts |
| L4 — provider integration | Real detector/depth and the intended G1 locomotion stack pass recorded and bench evaluation | NOT COMPLETE |
| L5 — protected hardware | Exact camera, model, profile, robot, state monitor, gantry, E-stop, and evidence are qualified together | NOT COMPLETE |

Passing one level does not imply the next. L3-G1 exercises one external G1
asset/policy pair, but does not qualify RGB-D perception, the separate 29-DOF
DDS stack, networking, or physical stopping behavior.

The interactive `stack` composition exercises L2/L3 continuously and may use
the optional Codex semantic provider. A model-backed run is evidence for that
specific provider and configuration; its existence does not change the
default deterministic qualification status.

The documented local live-provider smoke uses the official Codex SDK,
`gpt-5.6-terra`, reasoning effort `none`, and the current Codex login. It has
accepted a Chinese follow request into the only allowed
`navigation.follow_person` Skill and completed an articulated G1 interactive
run through reserved STOP with verified stationary-base evidence. This is
connectivity and contract evidence, not model-quality, latency, fault-matrix,
or physical-deployment qualification.

## What the system simulation exercises

```text
Mock wake/final transcript event
  -> WakeDictationController
  -> FollowMissionRuntime
  -> bounded deterministic Brain proposal
  -> revision and allowed-Skill validation
  -> navigation.follow_person Skill admission
  -> FollowPerson Runtime
  -> local planner + governor + independent obstacle guard
  -> synthetic or MuJoCo target
  -> next atomic FollowScene
  -> unawakened partial STOP bypasses Brain
  -> correlated simulation stop evidence
```

The deterministic Brain is an offline test provider. It receives only a
bounded context and may propose `navigation.follow_person`; it cannot emit
velocity, trajectory, SDK, shell, or Safety-override actions. A real Codex or
other model provider must satisfy the same `FollowBrainPort` and pass separate
timeout, invalid-output, stale-decision, and prompt-injection evaluation.

The input boundary begins at versioned wake/ASR events. It validates session
fencing, wake stripping, final transcript routing, and the safety-only partial
path. It does not validate microphone acoustics, KWS, VAD, or ASR model quality.

Run fast L2 acceptance:

```bash
PYTHONPATH=src python3 -m longship.cli.follow_person system-simulate \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --instruction 'Jackie，跟着我走'
```

Run visible L3 acceptance:

```bash
PYTHONPATH=src python3 plugins/targets/mujoco_follow_person/runner.py \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --instruction 'Jackie，跟着我走' \
  --system --viewer --keep-viewer
```

Run the same L3 target with a live Longship interaction terminal:

```bash
PYTHONPATH=src python3 plugins/targets/mujoco_follow_person/runner.py \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --stack --viewer
```

Both reports must show one Brain request, one admitted Skill call, all six
pipeline stages (`input`, `brain`, `skill`, `runtime`, `safety`, `target`), no
unsafe forward command, and a verified simulation stop. MuJoCo additionally
requires zero robot-obstacle contact steps.

Run the external articulated L3-G1 acceptance or its interactive terminal with
the immutable-artifact commands in the
[G1 policy target guide](../../plugins/targets/mujoco_g1_policy/README.md).
That target uses the same scenario and Longship components above the target
boundary. Its additional gates require no fall, no physical barrier contact,
and stationary-base evidence after zero velocity.

## Behavioral and provider parity

| Area | Behavioral target | Longship implementation | Status / evidence needed |
| --- | --- | --- | --- |
| Atomic observation | Person, depth obstacles, health, timestamps, and sequence belong to one frame | `longship.follow-scene.v1` | Implemented and tested |
| Target lock | Short-lived IDs, bounded loss prediction, nearby-ID reacquisition | FollowPerson Runtime | Implemented in L1–L3 |
| Local navigation | Fresh base-frame footprint-aware A* without invented global pose | LocalFollowPlanner | Implemented in L1–L3 |
| Motion bounds | Distance/heading control plus acceleration, jerk, speed, yaw, and TTL limits | MotionGovernor and command contract | Implemented in L1–L3 |
| Independent clearance veto | Raw forward corridor remains separate from person-excluded planning map | ForwardObstacleGuard | Implemented in L1–L3 |
| Perception diagnostics | Single camera owner, atomic preview/scene sequence, read-only browser | RealSense provider plus G1 simulation camera/environment HUD | Implemented and simulation-HUD checked; real camera remains unqualified |
| Person detector | TensorRT YOLO preferred, ONNX debug, HOG fallback | HOG only | Gap: add independently implemented, artifact-locked detector providers |
| Depth field of view | Native wide depth for floor/obstacles, RGB-aligned depth for person range | RGB-aligned depth for all paths | Gap: native-depth obstacle/floor provider |
| Depth stability | Spatial support, multi-frame confirmation, emergency immediate stop, stable floor tracking | Spatial cell support and immediate raw minimum | Partial: temporal confirmation and floor stabilization missing |
| Brain admission | Brain proposes semantic Skill; Runtime validates current revision and resources | Deterministic Follow Brain plus optional schema-constrained Codex provider | Deterministic L2/L3 pass; Codex provider is implemented but not production-qualified; general resource scheduler remains incomplete |
| Interaction controls | Follow request, pause, resume, status, and reserved STOP | Live terminal and transcript event paths share `RuntimeTextPort` | Terminal path tested; acoustic KWS/VAD/ASR and speech output are not in L2/L3 |
| Simulation target | Commands affect measured world state and physical contacts | Synthetic world, planar MuJoCo proxy, and external Unitree RL Gym G1 target | L1–L3-G1 pass for their documented providers |
| G1 29-DOF simulation asset | External MJCF is immutable, licensed, complete, and loadable | External bundle/license hashes, XML structure, 29 motors, and mesh checks | Asset gate passes; dynamic 29-DOF DDS follow remains blocked |
| G1 12-joint dynamic simulation | External policy consumes bounded velocity and drives free-base G1 dynamics | Manifest/store verification, per-step model/lease/version/TTL/action guard, policy/PD target, fall/contact/base-stop evidence | L3-G1 passes for the documented local artifact hashes; checkpoint license remains `NOASSERTION`, perception is ground truth, and this is not 29-DOF parity |
| New 29-DoF policy/training seams | MJLab, Holosoma, and RL Lab stay provider-neutral and artifact governed | 98→29 MJLab and 100→29 Holosoma runtime contracts have synthetic tests; RL Lab is reference-only | No compatible 29-DoF dynamic target or runnable Longship training simulator yet; do not substitute these into L3-G1 |
| G1 target | Intended locomotion policy accepts bounded velocity and returns robot evidence | Direct official high-level `LocoClient.SetVelocity` adapter | Interface tested with fakes only; not equivalent to the reference Holosoma policy path |
| Robot state safety | Roll, pitch, joint velocity/temperature, battery, heartbeat, and freshness | Operator heartbeat only | Gap: independent qualified state monitor and central veto |
| Stop evidence | Zero command plus separately measured stationary evidence | Adapter supports correlated evidence; deployment CLI has no monitor source | Gap: integrate and qualify the monitor; current result remains `STOP_UNVERIFIED` |
| Operator workflow | Doctor, E-stop, calibration, model/profile identity, launcher, evidence retention | Interactive simulation stack, asset doctor, explicit hardware gates and runbook | Partial: no qualified G1 low-level stack launcher or gantry evidence |

## Conditions for claiming non-hardware completeness

“All non-hardware paths pass” may be stated only for an exact provider set and
must include:

1. recorded RGB-D replay through the selected detector and depth provider;
2. a real Brain-provider evaluation in addition to the deterministic Brain;
3. input-session, stale-decision, unauthorized-Skill, cancellation, and STOP
   overtaking tests;
4. L3 scenario coverage for clear path, detour, blocked path, occlusion,
   changed track ID, camera outage, floor invalidity, delayed loop, and stop;
5. artifact digests for detector, calibration, profile, model, and simulator;
6. no skipped acceptance gate and retained JSONL/report evidence.

The current repository proves the deterministic providers through L3 and the
documented external 12-joint provider through L3-G1. It does not yet satisfy
items 1, 2, 4 (full fault matrix), or 5 for a production stack.

## Conditions for claiming deployment equivalence

Do not call the implementation a seamless replacement for the reference G1
deployment until all of the following are complete:

- choose and independently implement the intended target boundary: the
  existing Holosoma policy/PD/LowCmd stack or a separately qualified onboard
  high-level locomotion service;
- restore equivalent TensorRT/ONNX and native-depth provider capability;
- integrate fresh robot-state, posture, temperature, battery, and measured-stop
  evidence without allowing that monitor to share the command path;
- provide a target-specific doctor/start/stand-down launcher with immutable
  dependency and configuration identities;
- pass camera bench, network-loss, hanging-RPC, gantry, obstacle, E-stop, and
  measured-stop trials on the exact robot;
- issue a reviewed, unexpired qualification bound to those exact artifacts.

Until then, the checked-in disabled calibration and qualification examples are
intentional blockers rather than configuration templates that may be casually
enabled.
