# RL Platform Architecture

> **Status:** Executable training skeleton. Longship owns reusable model
> components and run orchestration; upstream integrations still own their
> simulator-specific trainer loops and native production recipes.

## Responsibilities

The RL platform follows one lifecycle:

```text
experiment -> train -> export -> sim2sim -> deploy
```

There is no independent `algorithms/` package and no separate safety-validation
stage in this lifecycle. PPO, SAC, or another optimizer is selected by the
experiment's `training.trainer.type`; its implementation is supplied by the
selected training backend.

```text
src/longship/rl/             # training repository
├── compatibility/     # one version/source/robot-contract lock
├── models/
│   ├── policies/       # top-level forward graph
│   ├── encoders/       # observation and modality encoding
│   ├── backbones/      # reusable feature processing
│   ├── decoders/       # actor, value, Q, and motion outputs
│   └── distributions/  # action distributions
├── training/
│   └── backends/       # HoloSoma, InstinctLab, SONIC, MimicLite adapters
└── data/               # data contracts and loaders

modules/longship-sim2real/   # independent Git submodule
├── src/longship_runtime/
│   ├── runtime/        # shared policies, DDS adapters, profiles and teleop
│   ├── sim2sim/        # MuJoCo process, scenes and launch orchestration
│   └── deploy/         # physical sensors, targets and process orchestration
└── scripts/            # Sim2Sim and real deployment entry points
```

Training implementations live under `src/longship/rl`. Execution code is an
independently installable `longship-sim2real` submodule and never imports the
training package. The repository root does not create a second experiment hierarchy:

```text
outputs/                # ignored generated runs and resolved configs
environments/           # Longship-owned simulator runtime profiles
third_party/            # pinned upstream source/model assets; not copied into submodule
```

Experiments stay with the selected RL integration. A backend adapter translates
that experiment into Longship's validated configuration boundary rather than
copying every upstream recipe into a root-level `experiments/` directory.

Policies do not belong to either execution target. Both Sim2Sim and physical
deployment launch the same adapter from `runtime/adapters/`, with the same
model pipeline from `runtime/policies/` and the same control profile from
`runtime/profiles/`. Sim2Sim supplies MuJoCo DDS producers/consumers; deploy
supplies physical sensors and robot targets. Neither target copies inference.

## Configuration and construction

An experiment describes a fixed model shape:

```text
observation -> encoder -> backbone -> actor/value/Q decoder
```

Every configured component has a `type`. The typed registry resolves that name
to a Python implementation, and `build_model()` recursively constructs the
declared slots. The top-level policy owns the forward graph; YAML is not a
general-purpose DAG language.

The experiment file also selects the upstream training backend and trainer.
For example, `HoloSomaBackend` with `trainer.type: PPO` means that Longship
adapts the recipe to HoloSoma's PPO runner. It does not reimplement PPO in the
model package.

The first reusable model set is implemented in `models/`:

- proprioceptive-history and depth-image encoders;
- MLP and dense mixture-of-experts backbones;
- Gaussian actor and scalar value decoders;
- actor-critic and perceptive actor-critic policy graphs.

These modules can be assembled and instantiated directly from YAML with
`build_model()`. The bundled `hiking_g1_parkour.yaml` example documents the
released 8x18x32 depth input, 128-D visual latent, 768-D proprioceptive input,
and 29-D action output. Its upstream InstinctLab task remains authoritative for
the complete production trainer graph until model-parameter translation is
implemented by that backend.

Registrations are separated by kind:

- `policy`, `encoder`, `backbone`, and `decoder` for models;
- `training_backend` for upstream training frameworks;
- future `sim2sim_runner`, `exporter`, and `deploy_target` adapters.

This prevents a same-named component from crossing architectural boundaries.

## Reproducibility

`ExperimentRunner` validates the recipe, creates a new output directory, and
writes `resolved.yaml` before handing control to the training backend. The
runner refuses to reuse an existing output directory, preventing accidental
checkpoint and configuration overwrite. Every built-in backend first creates
a shell-free `TrainingPlan` containing the exact argument vector, working
directory, environment overrides, output path, and checkpoint patterns. A real
run records this as `command.json`, including completion or failure state.

The registered training adapters are `HoloSomaBackend`, `SonicBackend`,
`InstinctLabBackend`, and `MimicLiteBackend`. They map Longship's seed and
output directory into the actual upstream trainers without invoking a shell.
MimicLite additionally maps the platform motion dataset, MJLab terrain, policy
module, checkpoint source, parallel environment count, and iteration budget to
Hydra while keeping its upstream PPO implementation authoritative.

Runtime profiles are owned by Longship rather than by upstream repositories.
They are dependency boundaries inside one training platform: the shared
IsaacLab runtime remains pinned to Torch 2.5.1, while
`environments/rl/mjlab` carries MJLab's newer Torch requirement. Both are
selected and launched through the same experiment and `longship-rl-train`
contract.

Inspect a job without starting Isaac Lab or allocating a GPU:

```bash
longship-rl-train --root "$PWD" plan \
  src/longship/rl/experiments/hiking_g1_parkour.yaml \
  --output outputs/hiking-plan
```

Start it by replacing `plan` with `run`. `run` creates the output directory, so
the supplied path must not already exist.

The bundled `mimiclite_g1_71cm_climb_turn_sit.yaml` recipe follows the same
contract for the 611-frame climb-turn-sit reference and `box71` training scene.

## Unified runtime and Sim2Sim preflight

`environment.yml` defines the single `longship-rl` Conda target. The bundled
`compatibility/longship_rl_v1.yaml` is the platform lock for Python, CUDA,
PyTorch, Isaac Sim, Isaac Lab, MuJoCo, ONNX Runtime, upstream source snapshots,
and the shared G1 29-DOF control contract. The lock is authoritative even when
an upstream repository carries a looser or older dependency range.

Create the environment and inspect all registered Sim2Sim integrations with:

```bash
conda env create -f environment.yml
conda activate longship-rl
pip install -e .
pip install -e ./modules/longship-sim2real
longship-rl-sim2sim preflight all --root "$PWD"
```

The preflight loads real ONNX bytes on CPU, verifies required simulator assets,
and rejects Git LFS pointer files. A backend is reported as `BLOCKED` until its
published model and asset bytes are present; source code alone is not reported
as a runnable Sim2Sim integration.

Current upstream snapshot status:

- HoloSoma: the locomotion ONNX and MuJoCo G1 asset are present and loadable.
- SONIC: the release encoder, decoder, target-velocity planner and Unitree SDK
  archive are registered. Its heavyweight ONNX files stay ignored by Git. It
  uses Longship's Python ONNX Runtime and the same MuJoCo G1 scene as the other
  policy integrations; the upstream C++ deploy remains a parity reference.
- Hiking in the Wild: the released parkour and stand depth encoders/actors are
  registered under the InstinctLab integration; heavyweight ONNX files stay
  ignored by Git.
- Perceptive Humanoid Parkour (PHP): the released 29-DoF student policy, depth
  backbone, official obstacle scene, ONNX metadata, and browser-controller
  observation contract are registered. The public snapshot does not contain
  the expert/student training implementation.

The user-facing one-click entry is `modules/longship-sim2real/scripts/sim2sim/run_hiking.sh`;
`run_instinctlab.sh` remains as a compatibility alias for the integration name.

Artifact readiness and launch readiness are deliberately separate. All four
integrations now share one Longship-owned MuJoCo simulator process and Unitree
SDK2 DDS on domain 0, bound to loopback `lo`:

```text
MuJoCo --rt/lowstate---------------> policy --rt/lowcmd--> MuJoCo
        --rt/secondary_imu (SONIC)-->
```

The simulator executable, physics loop, state publisher, command subscriber,
and hardware joint order are shared by HoloSoma, SONIC, Hiking, and PHP.
SONIC's planner, encoder and decoder use the shared Python ONNX Runtime engine.
`provider: auto` selects CUDA when available and otherwise uses CPU, so CUDA and
TensorRT are not launch requirements.
Model-owned control values are not embedded in the simulator. Each backend has
a versioned profile under `modules/longship-sim2real/src/longship_runtime/runtime/profiles/` that identifies
its robot MJCF/foot-contact preset, initialization pose, PD source,
initialization duration, control rates, gantry settings, and policy artifact.
The simulator executable and transport remain shared; physical parameters are
injected by the backend profile. HoloSoma owns explicit initialization
values and reads runtime gains from ONNX metadata; SONIC's Python pipeline owns
the gains, default pose, action scale and joint mapping ported from its model;
Hiking resolves pose, action scale, and gains from each checkpoint's
`params/env.yaml`; PHP reads the corresponding values and policy joint order
from ONNX metadata. Every adapter sends the resolved
`q/dq/tau/kp/kd` in `rt/lowcmd`.

Before the first complete finite `LowCmd`, MuJoCo freezes the reset state while
continuing to publish `LowState`; it does not apply a shared fallback pose or
PD controller. Hiking and PHP enable the optional depth sensor and publish it on
`rt/camera/depth`. Only the policy-side adapter and profile change. HoloSoma is run through
a Longship Unitree SDK2 adapter because its native FAR C++ DDS binding is not
wire-compatible with the platform's Python SDK participant. The policy and
simulator remain separate processes; direct Python calls are not a supported
Sim2Sim transport.

Interactive validation uses a third, policy-independent ZMQ keyboard process:

```text
keyboard --ZMQ--> policy adapter --rt/lowcmd--> Longship MuJoCo
                                     ^                |
                                     +--rt/lowstate---+
```

Press `i` to interpolate the robot to the policy's initial pose over the duration
declared by its profile, then `]` to enable inference. HoloSoma accepts
`W/S/A/D` linear commands and `Q/E` yaw.
PHP uses the released 15-way discrete command bank: `W` moves forward, `Q/E`
select diagonal left/right, `A/D` select left/right, `S` stops, and `Y` toggles
the low/high-speed bank. Its policy runs at 50 Hz with the previous action and a
seven-step-delayed 32-D depth latent; it does not stack proprioceptive frames.
SONIC does not consume HoloSoma-style velocity commands. `]` seeds a measured
one-frame tracking reference and enables the Python ONNX policy without starting
planner playback. The first SONIC motion key starts its target-velocity planner;
subsequent plans use rolling motion context and the native eight-frame splice.
`W/S` select forward/backward movement, `A/D` select pure lateral movement,
`Q/E` clear movement and change the discrete facing target, and `1/2` select the
slow-walk/walk modes in the default standing set. `N/P` cycles SONIC's four
native mode sets and `1-8` selects within the active set:

- standing: slow walk, walk, run, forward jump, stealth walk, injured walk;
- squat/crawl: squat, two-leg kneel, kneel, crawl, elbow crawl;
- boxing: idle boxing, walk boxing, punches and hooks;
- styled walk: ledge, object-carrying, stealth-2, happy-dance, zombie, gun and
  scare walks.

`9/0` adjusts planner speed and `-/=` adjusts squat/crawl height. The terminal
and MuJoCo window remain separate: terminal `9` changes SONIC speed, while
MuJoCo-window `9` toggles the gantry. HoloSoma's stand/walk toggle and continuous
velocity/yaw semantics are therefore not exposed for SONIC.
Hiking uses the same shared Python ONNX Runtime engine for both released
agents. `1` selects its stand agent, `2` selects its depth-aware parkour agent,
and `N/P` cycles between them. The default is stand, matching the upstream
handoff. Only parkour accepts `W/Q/E` for forward and yaw; it explicitly rejects
`S/A/D`, and stand rejects every motion key. Adapters never synthesize an
unsupported command dimension. Each Hiking agent keeps its own history, pose,
action scale, and checkpoint-resolved PD gains while DDS, MuJoCo, depth transport,
ZMQ teleop, and the ONNX provider policy remain shared.
Its proprioception uses `rt/secondary_imu` from the torso, matching the released
deployment stack; `rt/lowstate`'s pelvis IMU is not substituted.
Both checkpoints receive the source-aligned DDS depth stream during Sim2Sim,
and the last action is carried across the stand-to-parkour handoff.
The Hiking adapter also owns its source-scene waist-axis sign transform; PD
gains are reordered without applying those position/velocity signs.
Hiking's profile selects its source-aligned low-catch/horizontal/upright
spotter during cold-start and stand handoff; HoloSoma and SONIC retain the
visible spring-rope gantry. Both modes are implementations of the same shared
simulator, selected only through backend configuration.
The Hiking profile also injects its source-aligned crouched reset pose before
the shared simulator calibrates the feet to the terrain.
During cold start/READY it applies the upstream `2x` initialization gain scale,
then atomically changes to each selected checkpoint's native gains on enable.
Headless Hiking regressions can use `--sim-duration` on both simulator and
controller so a slow high-fidelity contact model is evaluated by physical
simulation time rather than wall-clock time.
`--auto-parkour-at-sim` is an opt-in headless regression hook for reproducing
the upstream timed stand-to-parkour handoff; interactive launches never set it.
Hiking policy inference is scheduled from the LowState simulation timestamp,
not wall time, so its 50 Hz observation/action history remains correct even
when high-fidelity contacts run below real time.
SONIC follows the same simulation-clock rule. Its model-owned planner mask uses
the released 6--11-token range for ordinary locomotion and the full 6--16-token
range only for walk-boxing and elbow-crawling. The mask changes the generated
gait distribution and is therefore not a generic runtime tuning parameter.
When an enabled Hiking controller switches from stand to parkour, it publishes
a simulator-only DDS handoff event so the Hiking spotter is released in that
same transition instead of contaminating the parkour history with restrained
forward commands. Viewer `9` remains available for manual gantry control.
Repeated `i` presses during interpolation do not restart its timer. An enable
request received during interpolation is queued and automatically activates
the policy when initialization completes.
HoloSoma keeps the stronger initialization PD gains through READY, then changes
to policy actions and policy gains in the same control cycle. DDS callbacks
copy state and command messages into immutable snapshots before the control
thread reads them, preventing mixed-frame handoff commands. Its zero-command
phase matches upstream by holding both feet at phase pi. MuJoCo physics remains
at 500 Hz while the passive viewer synchronizes at 60 Hz, with absolute-time
pacing keeping the simulation real-time factor at approximately one.

Check the host and print backend launch commands with:

```bash
longship-rl-sim2sim dds-check --interface lo --domain-id 0
python -m longship.rl.sim2sim.dds_probe --interface lo --domain-id 0
longship-rl-sim2sim commands holosoma --python "$CONDA_PREFIX/bin/python"
longship-rl-sim2sim commands sonic --python "$CONDA_PREFIX/bin/python"
longship-rl-sim2sim commands instinctlab --python "$CONDA_PREFIX/bin/python"
longship-rl-sim2sim commands php --python "$CONDA_PREFIX/bin/python"
```

Each `commands` invocation prints three terminals: simulator, controller, and
ZMQ keyboard. Start them in that order, press `i`, wait for initialization to
finish, then press `]` and issue only the keys listed by the keyboard process.

For the normal interactive path, the repository also provides one-command
launchers that start and clean up all three processes automatically:

```bash
./modules/longship-sim2real/scripts/sim2sim/run_holosoma.sh
./modules/longship-sim2real/scripts/sim2sim/run_sonic.sh
./modules/longship-sim2real/scripts/sim2sim/run_instinctlab.sh
./modules/longship-sim2real/scripts/sim2sim/run_php.sh
```

They prefer the active compatible Conda environment, then discover the
`longship-rl` or `env_isaaclab511` environment. Set `LONGSHIP_PYTHON` to select
an explicit interpreter. Per-process logs are retained under
`outputs/sim2sim/<backend>/<timestamp>/`; `Ctrl-C` stops the complete stack.
The one-command launchers open a MuJoCo Viewer by default. Closing that window
stops the simulator; direct invocations of `longship.rl.sim2sim.simulator`
remain headless unless `--viewer` is supplied.

The interactive viewer also exposes the original Sim2Sim gantry controls:
`7/8` shorten/lengthen the elastic band by 0.1 m, `9` toggles its physical
spring-damper force, and `Backspace` resets robot pose, velocity, cached DDS
command, gantry anchor, and simulation time. One-command launchers start with
the gantry enabled; direct simulator runs can opt in with `--gantry`.
The reset pose is vertically calibrated from both foot contact points, while
the initial gantry length comes from the selected backend profile. Both feet
therefore start at ground height without manual adjustment while the rope remains
enabled during initialization. The viewer draws the active rope as a blue capsule and
its world-frame anchor as a red sphere; both disappear when `9` disables the
gantry. The gantry retains HoloSoma's original `7/8/9` behavior.
`Y` toggles a camera whose look-at point follows the torso as the robot moves.
Tracking updates only the look-at point: MuJoCo mouse orbit, elevation, and
zoom remain under interactive control while tracking is enabled.

For SONIC, follow the upstream validation order: keep the gantry enabled during
initialization and policy takeover, then press `9` in the MuJoCo window after it
has settled. Headless regression runs can use `--gantry-release-after <seconds>`;
this option is for deterministic testing and does not change interactive input.

The first command checks OS permissions. The second sends real Unitree HG
`LowState` and `LowCmd` samples through CycloneDDS and fails on timeout. The
preflight prints required host capabilities but cannot grant them inside a
network-disabled sandbox.
