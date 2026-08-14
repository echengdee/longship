# Draft Contract Proposals

> These files are discussion drafts. They are not released compatibility
> guarantees and must not be treated as production-ready robot interfaces.

The proposal set supports
[System Architecture v2](../../docs/architecture/system-architecture-v2.md)
and
[Model, Framework, and Artifact Integration](../../docs/architecture/model-and-artifact-integration.md):

- `operator-intent.v1.schema.json` — semantic task or dialogue intent that may
  use a model; deterministic controls are excluded.
- `runtime-control-command.v1.schema.json` — authenticated pause, resume,
  cancel, status, speed, confirmation, and teleoperation-mode operations that
  explicitly bypass Brain providers.
- `safety-stop-request.v1.schema.json` — an idempotent protective-stop request
  that bypasses Brain, compiler, scheduler, and safe-point waiting. It is not a
  physical emergency-stop protocol.
- `control-command-result.v1.schema.json` — immutable acknowledgement,
  commanded, measured-effect, safe-state, timeout, and failure transitions.
- `mission-task-graph.v1.schema.json` — a versioned parallel DAG with resource
  claims, admission groups, event edges, barriers, preemption, and hierarchical
  cancellation.
- `mission-task-graph-patch.v1.schema.json` — typed, expiring pending-only graph
  mutations bound to graph, Runtime state, and active-Skill-set versions.
- `execution-snapshot.v1.schema.json` — authoritative graph state, active
  Skill set, foreground call, safe points, resource leases, versions, and
  current Safety state.
- `brain-request.v1.schema.json` — event-triggered current state, available
  Skills, previous decision, recent events, and relevant historical episodes.
- `brain-decision.v1.schema.json` — an expiring compare-and-swap high-level
  proposal bound to the whole active-Skill-set version.
- `model-artifact-manifest.v1.schema.json` — role-conditional external model
  artifacts, licenses, resources, interfaces, degraded modes, and action-only
  safety requirements.
- `model-session-lock.v1.schema.json` — an immutable audit snapshot of
  independently versioned role bindings, deployment locks, hashed handoff
  gates, and role-scoped rollback targets.
- `runtime-event.v1.schema.json` — authoritative task, control, barrier,
  cancellation, model handoff, safety, recovery, and health transitions.
- `telemetry-envelope.v1.schema.json` — transport-neutral identity, timing,
  units, frames, freshness, quality, graph/model correlation, and privacy.
- `experience-episode.v1.schema.json` — a structured execution index that
  references immutable external artifacts instead of embedding media or logs.

## Proposal rules

- All schemas use JSON Schema Draft 2020-12.
- Cross-file references use each target schema's absolute Longship URN; relative
  references are reserved for fragments within the current schema.
- A proposal carries `x-longship-status: draft` and may change incompatibly
  before release.
- Examples must be synthetic and license-compatible.
- High-rate controller and teleoperation transports may use Protobuf or another
  encoding while preserving authority, TTL, ownership, and Safety boundaries.
- Physical emergency-stop wiring and target controllers remain outside these
  application-level schemas.
- Public Brain registries never expose shell, SDK, joint, torque, motor, PWM, or
  raw trajectory tools.
- Cross-schema equality, reference existence, DAG acyclicity, barrier quorum,
  time ordering, lease expiry, capacity, deadlock prevention, cancellation
  epochs, trigger deduplication, stop evidence, and compare-and-swap checks are
  enforced transactionally by Runtime and target qualification.
- A schema-valid `resume` never clears a latched protective or emergency stop.
- Safety-critical implementation requires target-specific qualification beyond
  schema validation.

Review should focus on authority, lifecycle, latency evidence, parallel failure
behavior, resource ownership, model handoff, versioning, privacy, and whether a
component can fail without delaying protective stopping.
