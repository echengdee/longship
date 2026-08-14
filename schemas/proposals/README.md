# Draft Contract Proposals

> These files are discussion drafts. They are not released compatibility
> guarantees and must not be treated as production-ready robot interfaces.

The first proposal set supports the architecture described in
[System Architecture v2](../../docs/architecture/system-architecture-v2.md)
and [Model, Framework, and Artifact Integration](../../docs/architecture/model-and-artifact-integration.md):

- `operator-intent.v1.schema.json` — normalized task-level voice, keyboard, UI,
  API, or WMS intent. It deliberately excludes actuator commands, manual
  teleoperation packets, and emergency-stop authority.
- `execution-snapshot.v1.schema.json` — authoritative active Skill, safe point,
  resource leases, state and plan versions, and current Safety state.
- `brain-request.v1.schema.json` — event-triggered current state, available
  Skills, previous decision, recent events, and relevant historical episodes.
- `brain-decision.v1.schema.json` — an expiring, compare-and-swap high-level
  proposal with a limited action set and pending-only plan patches.
- `model-artifact-manifest.v1.schema.json` — immutable external model and policy
  artifacts, licenses, resources, interfaces, deployment, and safety envelope.
- `runtime-event.v1.schema.json` — authoritative lifecycle, task-switch,
  recovery, safety, and health transitions, including deterministic
  announcement metadata.
- `telemetry-envelope.v1.schema.json` — transport-neutral identity, timing,
  units, frames, freshness, quality, correlation, and privacy metadata for
  typed telemetry payloads.
- `experience-episode.v1.schema.json` — a structured execution index that
  references immutable external artifacts instead of embedding media or logs.

## Proposal rules

- All schemas use JSON Schema Draft 2020-12.
- A proposal carries `x-longship-status: draft` and may change incompatibly.
- Payload-specific schemas must be versioned independently.
- Examples must be synthetic and license-compatible.
- High-rate transports may use Protobuf or another encoding while preserving
  the semantic contract.
- Safety-critical implementation requires target-specific qualification beyond
  schema validation.
- Cross-schema equality, time ordering, lease expiry, trigger deduplication,
  and compare-and-swap checks are enforced transactionally by Runtime.

Review should focus on authority, lifecycle, failure behavior, versioning,
privacy, and whether another implementation can reproduce the same semantics.
