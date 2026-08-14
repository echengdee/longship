# Plugins

Longship is designed to use plugins so the core can remain independent of specific models,
simulators, sensors, and robots.

Planned plugin kinds include Brain and dialogue adapters, ASR and TTS adapters,
knowledge sources, perception providers, Skills, policy adapters, target
adapters, and evaluators. Runtime selects providers per role, so several
qualified plugins may be active concurrently.

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

## Heavy Models, Frameworks, and Robot SDKs

Plugins remain thin even when their upstream dependency is large. Keep
Longship-owned adapters, manifests, modality and target configuration, license
notices, mock tests, and evaluation summaries in Git. Keep model weights,
policy checkpoints, datasets, compiled engines, media, and runtime images in
external registries by immutable digest.

Recommended role separation:

- `brains/<provider>` — event-driven high-level planning APIs;
- `dialogue/<provider>` — open-ended conversation;
- `speech/asr/<provider>` and `speech/tts/<provider>` — independent audio roles;
- `perception/<provider>` — versioned detection, tracking, or scene interpretation;
- `policies/groot` or `policies/unifolm` — bounded VLA policy adapters;
- `locomotion/holosoma` — training/export integration and checkpoint runtime;
  and
- `targets/unitree` — exact robot and SDK state/command translation.

A plugin references, but does not copy, its upstream repository. Production
preflight resolves and verifies a reviewed artifact lock, checks license and
resource requirements, warms the runtime, and confirms target qualification.
Live missions never trigger a first-time model download.

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
