# External Unitree G1 policy MuJoCo target

This experimental target runs Longship's existing FollowPerson composition on
a real free-base G1 MuJoCo model and an external TorchScript locomotion policy.
It replaces only the target/world provider. Input routing, Brain admission,
the semantic Skill, Runtime state, local planning, the independent clearance
guard, command TTLs, and protected STOP remain Longship-owned components.

No third-party source, XML, mesh, configuration, license, or model byte is
copied into this repository. The operator supplies those files
from an external installation. The runner verifies the complete G1 asset
directory and each policy/configuration/license file against an explicit
SHA-256 lock before importing the policy or compiling the scene.

## Integration with Longship's governed policy layer

This target now uses the repository's common model and policy boundaries:

1. `model-artifacts.experimental.json` records the external policy,
   configuration, and license as immutable references. Its draft,
   `reference_only`, and `NOASSERTION` state deliberately blocks automatic
   prefetch.
2. `ArtifactStore.verify` checks each operator-supplied regular file by size
   and SHA-256 through a no-follow descriptor. The loader reopens and verifies
   the same inode and digest before parsing YAML or loading TorchScript from an
   in-memory snapshot.
3. Every 50 Hz inference result is represented as an untrusted
   `PolicyCandidate`, bound to the manifest-selected model, current simulation
   lease epoch, observation version, 20 ms horizon, 12-value action space, and
   `g1_lower_body_motion` resource scope.
4. The shared deterministic guard rejects stale identity, wrong dimensions,
   scope escalation, non-finite values, and raw actions outside the explicit
   simulation fault-containment bound before PD targets are updated.

The multi-file MJCF/mesh directory remains under the documented deterministic
bundle hash because `ModelArtifactManifest` currently models regular-file
artifacts, not a mutable directory tree.

The newly merged provider seams are related but cannot be swapped into this
MuJoCo world without a matching model and target adapter:

| Provider seam | Contract | Current executable state |
| --- | --- | --- |
| this external Unitree RL Gym target | 47 observations → 12 lower-body actions | closed-loop G1 MuJoCo PASS for the locked local artifact set |
| official Unitree RL MJLab velocity-v0 | 98 observations → 29 actions | core backend and synthetic tests; artifact/license and simulator target blocked |
| Holosoma G1 loco | 100 observations → 29 actions | core backend and synthetic tests; checkpoint/license and simulator target blocked |
| Unitree RL Lab training family | training/export reference | reference-only; no training stack or checkpoint is vendored or activated |

This preserves Longship's provider/target separation while allowing a later
29-DoF simulator to reuse the same FollowPerson Runtime above the target seam.

For the locally inspected layout, the convenience launcher keeps the same
explicit locks while reducing the command to:

```bash
cd /absolute/path/to/longship
export UNITREE_RL_GYM_ROOT=/absolute/path/to/unitree_rl_gym
export LONGSHIP_G1_PYTHON=/absolute/path/to/python-with-mujoco

bash plugins/targets/mujoco_g1_policy/run_inspected.sh --mode system
```

Use `--mode system --viewer --real-time --keep-viewer` for visible automated
acceptance, or `--mode stack --viewer` for the interactive Longship terminal.

The locomotion in this plugin is **not Holosoma**. It loads the external
Unitree `unitree_rl_gym` G1 example policy named in `THIRD_PARTY.md`; Longship
only supplies bounded planar velocity targets. It is also not the separate
29-DOF `unitree_mujoco` DDS simulation. Do not infer a stability ranking from
the provider name: compare retained fall, maximum-tilt, contact, tracking, and
measured-stop evidence under the same scenarios.

## Camera HUD and interactive input

Set the external paths once in the terminal used to launch the simulation:

```bash
cd /absolute/path/to/longship
export UNITREE_RL_GYM_ROOT=/absolute/path/to/unitree_rl_gym
export LONGSHIP_G1_PYTHON=/absolute/path/to/python-with-mujoco
```

Choose a launch mode from this matrix:

| Mode | MuJoCo window | Browser HUD | Terminal commands | Rendering |
| --- | --- | --- | --- | --- |
| automated fast acceptance | no | no | no | default |
| automated visible acceptance | yes | no | no | desktop display |
| interactive HUD for SSH/headless use | no | yes | yes | `MUJOCO_GL=egl` |
| interactive Viewer + HUD | yes | yes | yes | desktop display; no `egl` |
| interactive Viewer + HUD + Codex | yes | yes | yes | desktop display; no `egl` |

Fast headless acceptance runs faster than wall-clock time, prints the report,
and exits:

```bash
bash plugins/targets/mujoco_g1_policy/run_inspected.sh --mode system
```

Watch the automated acceptance in the native MuJoCo window:

```bash
bash plugins/targets/mujoco_g1_policy/run_inspected.sh \
  --mode system \
  --viewer \
  --real-time \
  --keep-viewer
```

Run interactively with both the native MuJoCo window and browser HUD:

```bash
bash plugins/targets/mujoco_g1_policy/run_inspected.sh \
  --mode stack \
  --viewer \
  --hud-port 8093 \
  --keep-viewer \
  --keep-hud
```

Do **not** set `MUJOCO_GL=egl` for that combined desktop command. Open
`http://127.0.0.1:8093`, then enter `跟着我走`, `状态`, `暂停`, `继续`, or
`停止` in the launching terminal. After STOP, close the MuJoCo window and
press Enter in the terminal when the retained HUD is no longer needed.

For SSH or a machine without a desktop display, run the G1 dynamics headlessly
with its pelvis-mounted simulation camera and the same browser HUD:

```bash
MUJOCO_GL=egl bash plugins/targets/mujoco_g1_policy/run_inspected.sh \
  --mode stack \
  --hud-port 8093 \
  --keep-hud
```

This headless form intentionally has no native MuJoCo window. Accessing the
HUD from another machine additionally requires an operator-managed tunnel or
an explicitly reviewed bind address; the default remains localhost-only.

The terminal is the current Longship text-input provider; the browser is
deliberately read-only and cannot arm, pause, stop, or publish velocity.

The HUD shows the rendered G1 camera, base-local person/obstacles/path/next
goal, person and follow-goal world coordinates, command, clearance, G1 pose,
base height, tilt, contacts, policy/physics steps, Brain/Skill events, and the
active Brain provider. It also shows the active MissionTaskGraph and current
semantic node. It publishes an initial frame before any command and
refreshes while the Runtime is active. The camera is a virtual pinhole attached to the
externally loaded G1 pelvis. Control perception remains scripted ground truth,
so this proves rendering and observability, not RGB detection from those
pixels. Physical deployment instead proxies the independent RealSense
worker's loopback JPEG preview into the same read-only HUD.

## Use the current Codex login as Brain

MCP is not required for Longship to call the current local Codex environment.
The existing `FollowBrainPort` uses the official Python SDK/app-server directly
and reuses `codex login` credentials. Install that optional SDK into the same
Python ABI as MuJoCo once:

```bash
codex login status
bash plugins/targets/mujoco_g1_policy/install_codex_brain.sh
```

Then run:

```bash
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

That command opens Viewer and HUD together. For a headless Codex run, remove
`--viewer --keep-viewer` and prefix the command with `MUJOCO_GL=egl`.

For GPT-5.6, `none` is the no-reasoning setting. Model startup may still take
tens of seconds. Codex can only return a schema-constrained text response or
propose `navigation.follow_person` with bounded `follow`, `pause`, and `resume`
steps. Runtime validates that proposal against its current revision, compiles
it into a single-lane MissionTaskGraph, and advances node deadlines from the
control clock. Codex never sleeps or controls the timer.

For example, enter this in the launching terminal:

```text
跟我走三秒然后暂停，一秒后继续走
```

Admission returns immediately as:

```text
Mission task graph started: follow 3s -> pause 1s -> resume.
```

During those four seconds, `状态` remains responsive and reports the current
graph node. The HUD changes from `navigation.follow_person.start` to `pause`
and then `resume`. A standalone `暂停` or `继续` supersedes and cancels the
remaining graph. STOP always bypasses both Codex and normal graph scheduling,
cancels the graph, and requests the protected target stop path. Motion bounds,
planning, Safety, and the target remain deterministic Longship paths. The
model-backed provider is an experimental simulation option and is not enabled
by the physical deployment command.

`pause` deliberately keeps the same locomotion policy, motion lease, and
control loop active while sending a zero planar-velocity target on every
Runtime tick. It does not freeze joints, unload/reload the policy, switch to a
second controller, or invoke protected STOP. This avoids controller-switch and
policy-reinitialization latency and reduces the associated transient-stability
risk. The requested duration begins when Runtime admits the zero-velocity node;
it is not a measured stationary-base dwell. The G1 may therefore brake, balance,
or continue its policy gait phase during a short pause even though the commanded
velocity is zero. Protected STOP remains the separately measured stop path.

The `--events` path above is created exclusively and is never overwritten; use
a new path for each run. To inspect task-node transitions, pause state, and G1
velocity targets afterward:

```bash
rg 'task_graph.transitioned|"state":"paused"|desired_velocity' \
  /tmp/longship-g1-events.jsonl
```

The read-only HUD retains only bounded/latest state. Use the JSONL journal when
per-event evidence is required.

## Latest local evidence

On the reviewed external artifact hashes below, the final automated system run
passed with 401 target-command steps, 11,525 physics steps, maximum tilt
0.084 rad, zero barrier contacts, 0.227 m final distance error, and a verified
STOP. The three-second STOP observation recorded 0.046 m braking displacement,
0.042 m displacement in the stationary measurement window, and 0.039 m/s
final base speed.

The post-merge governed-policy regression produced 1,152 guarded policy steps;
the maximum absolute raw action was 1.968 against the explicit simulation
fault-containment bound of 10.0. The report records the artifact manifest ID
and digest together with the previous scene/policy/config/license digests.

A separate live `gpt-5.6-terra`/`none` stack run accepted `跟着我走` into one
`navigation.follow_person` Skill call, drove 72 interactive control/target
steps, accepted the reserved `停止`, and ended `passed=true` with no fall or
contact and verified stationary-base evidence. The HUD was also rendered in a
browser against the actual G1 run and served camera frame 401 together with
robot/person/follow-goal telemetry. These are local simulation results for the
exact hashes, not portable hardware or model-quality claims.

After the MissionTaskGraph integration, another live run submitted
`跟我走三秒然后暂停，一秒然后继续走` to the same Codex profile. Runtime
admitted `follow 3s -> pause 1s -> resume`; the HUD snapshot recorded the first
two nodes as succeeded and the resume node as running, while Runtime history
recorded `paused -> acquiring -> following`. A later reserved `停止` produced
`passed=true` after 354 interactive target-command steps, 10,350 physics
steps, no fall, zero barrier contacts, and verified STOP. The result validates
asynchronous graph execution for this simulator and artifact lock only.

## Locally inspected provider

The inspected `unitree_rl_gym` provider uses a free-base G1 with 12 actuated
lower-body joints. It is a real articulated humanoid dynamics simulation, but
it is not the separate 29-DOF Unitree MuJoCo DDS model. See
[THIRD_PARTY.md](THIRD_PARTY.md) for provenance and exact digests.

Run the full deterministic system acceptance without a window:

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

Add `--viewer --real-time --keep-viewer` to watch that acceptance in real
time. For the Longship-native interactive terminal, replace the last line
with `--mode stack --viewer`; then enter `跟着我走`, `状态`, `暂停`, `继续`,
and `停止`. Add `--brain codex` only when that provider is intentionally under
test. Fixed controls and STOP do not pass through Codex.

The system report passes only when all six stages (`input`, `brain`, `skill`,
`runtime`, `safety`, and `target`) are observed, the G1 remains upright, no
physical barrier contact occurs, following acceptance passes, and a
three-second zero-velocity dwell establishes stationary base evidence. The
first half is a braking/settling interval; stationary displacement and final
speed are evaluated over the second half. Balancing joint
motion is retained as diagnostic evidence because an actively balancing
biped need not have stationary joints.

## Scope and non-claims

- Person tracks and raw obstacle observations are scenario ground truth; RGB-D
  rendering, detection, tracking, and calibration are not evaluated here.
- The controller covers 12 lower-body joints, not the 29-DOF DDS surface.
- Simulation STOP evidence is not hardware STOP evidence.
- This plugin never enables the physical Unitree adapter.
- The exact physical robot still requires camera, network, robot-state,
  measured-stop, gantry, and E-stop qualification described in the main
  FollowPerson deployment guide.
