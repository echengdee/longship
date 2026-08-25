# Plugins

Longship is designed to use plugins so the core can remain independent of specific models,
simulators, sensors, and robots.

Planned plugin kinds include Brain and dialogue adapters, ASR and TTS adapters,
knowledge sources, perception providers, Skills, policy adapters, target
adapters, and evaluators. Runtime selects providers per role, so several
qualified plugins may be active concurrently.

Codex is the first public reference Brain provider and remains opt-in in the
mock scenario. It consumes text plus Longship-owned context and emits only
untrusted high-level proposals. Microphone capture, VAD/ASR, reserved command
recognition, TTS, Runtime state, and Safety remain separate roles.

Every plugin will provide a machine-readable manifest similar to:

```yaml
plugin_id: longship.target.mock
plugin_version: 0.1.0
api_version: 1.0.0
kind: target

contracts:
  world_state: 1.x
  command: 1.x

supported_targets:
  - mock

maturity: draft
```

The manifest format is still a proposal. It will be versioned before plugins
are accepted as compatible.

## Core Capability Wrappers

Some Longship-maintained, provider-independent capabilities are implemented in
`src/longship` and exposed through a thin plugin wrapper. The Navigation
Harness follows this pattern: its Mission, Map, Localization, Planning, and
Local Trajectory engines live in `src/longship/navigation`, while
`skills/navigation_harness` declares the `navigate_to` Skill and its Runtime
boundary. The core RoutePlan-driven runtime may publish policy-native local
trajectories through `LocalTrajectoryStream`; target-, transport-, safety-,
control-, and vendor-specific execution remain separate adapters.

## Heavy Models, Frameworks, and Robot SDKs

Plugins remain thin even when their upstream dependency is large. Keep
Longship-owned adapters, manifests, modality and target configuration, license
notices, mock tests, and evaluation summaries in Git. Keep model weights,
policy checkpoints, datasets, compiled engines, media, and runtime images in
external registries by immutable digest.

Recommended role separation:

- `brains/<provider>` — event-driven high-level planning APIs;
- `dialogue/<provider>` — open-ended conversation;
- `skills/<skill-plugin>` — optional semantic capability implementations;
- `navigation/<provider>` — mapping, localization, planning, and controller
  integrations behind the navigation Skill contract;
- `speech/asr/<provider>` and `speech/tts/<provider>` — independent audio roles;
- `perception/<provider>` — versioned detection, tracking, or scene interpretation;
- `policies/groot` or `policies/unifolm` — bounded VLA policy adapters;
- `locomotion/holosoma` — training/export integration and checkpoint runtime;
  and
- `targets/unitree` — exact robot and SDK state/command translation.

The first executable examples follow these boundaries:

- [`brains/codex_local`](brains/codex_local/) is the experimental, opt-in,
  non-actuating reference high-level Brain provider;
- [`speech/voice_inputs/jackie_sherpa_onnx`](speech/voice_inputs/jackie_sherpa_onnx/)
  reserves a local Jackie KWS/VAD/ASR composition without bundling or activating
  model artifacts;
- [`targets/unitree_g1`](targets/unitree_g1/) wraps a small, bounded portion of
  the official high-level SDK behind a default-off hardware gate;
- [`locomotion/unitree_rl_lab`](locomotion/unitree_rl_lab/) reserves an external
  policy seam without copying or activating model weights;
- [`locomotion/unitree_rl_mjlab_g1_velocity`](locomotion/unitree_rl_mjlab_g1_velocity/)
  implements a synthetic-testable 98-to-29 policy contract for the pinned
  official G1 velocity-v0 ONNX, while license and target gates keep the real
  artifact inactive;
- [`policies/groot`](policies/groot/) records a pinned, reference-only GR00T
  VLA boundary whose incomplete timing, gated dependency, and licensing block
  an executable adapter; and
- [`locomotion/holosoma`](locomotion/holosoma/) records a pinned framework and
  candidate-checkpoint audit boundary without importing the training stack.

A deployment may package capture, Jackie and reserved-STOP KWS, VAD, and ASR as
one `voice_input` composition when those stages must share one microphone
stream. The STOP spotter remains always on and safety-only; ordinary dictation
ASR remains wake-gated. The composition still emits the provider-neutral
`VoiceInputEvent` contract; it does not own Runtime, Codex, command
authorization, TTS, or actuators. This avoids several plugins racing to open
the same audio device while keeping the individual stages replaceable inside
the composition.

For robot capabilities, distinguish a semantic Skill from its provider. For
example, `navigation.navigate_to` is a Skill; Nav2 is a navigation provider;
and Unitree message/frame translation is a target adapter. See
[Skills, Runtime, Navigation, and Target Boundaries](../docs/architecture/skills-and-runtime.md)
and the [`plugins/skills` guidance](skills/README.md).

A plugin references, but does not copy, its upstream repository. Production
preflight resolves and verifies a reviewed artifact lock, checks license and
resource requirements, warms the runtime, and confirms target qualification.
Live missions never trigger a first-time model download.

The current shared implementation lives in `longship.policies` and
`longship.artifacts`. Policy backends receive an immutable request and may only
return a candidate bound to the same call, model binding, observation version,
lease ID and epoch, deadline, horizon, action space, and resource scope. A live
lease callback is checked before and after inference; the future arbiter and
target must repeat that fence before any side effect. The guard rejects stale,
escalated, malformed, expired-frame, or out-of-bound candidates. The artifact
store strictly validates manifest structure, uses immutable SHA-256 identities,
and requires a separate trusted approval plus deployment-supplied mission-state
and byte-transport boundaries. Longship intentionally ships no general network
downloader in this experimental layer.

`exclusive_with` is currently reviewable plugin metadata, not a claim that a
general Runtime plugin loader already enforces it. All new whole-body policy
plugins remain blocked until that loader, a live lease authority, and a
qualified target composition consume the same ownership declarations.

A `ModelSessionLock` binds exact provider and artifact revisions to roles.
The Model Session Manager warms and shadows a candidate before a role-specific
safe-point handoff. Shadow sessions own no actuator lease. Concurrent
action-producing plugins require disjoint Runtime leases and an approved
composition profile; conflicting outputs are rejected, never blended.

Public CI uses synthetic observations and mock providers. GPU, simulator, and
hardware evaluation run in separately authorized environments and publish
small signed reports and artifact references.

See
[Model, Framework, and Artifact Integration](../docs/architecture/model-and-artifact-integration.md)
and the draft
[ModelArtifactManifest](../schemas/proposals/model-artifact-manifest.v1.schema.json)
and
[ModelSessionLock](../schemas/proposals/model-session-lock.v1.schema.json).
