# Contracts and Schemas

Longship follows a contracts-first approach. This directory will contain versioned,
implementation-neutral schemas for:

- knowledge artifacts,
- missions and sessions,
- skills and commands,
- world-state snapshots,
- experience episodes,
- evaluation results, and
- artifact and plugin manifests.

Low-frequency contracts are expected to use JSON Schema with human-readable
YAML or JSON examples. High-frequency runtime interfaces may use Protobuf or
another explicitly versioned transport without changing the semantic contract.

Schemas are not yet released. Proposals should begin as public design
discussions and include compatibility tests before becoming stable.

## Draft Proposals

The first eight public discussion drafts cover operator intent, stable brain
decisions, execution state, runtime events, telemetry, structured experience,
and external model artifacts. See
[`schemas/proposals/README.md`](proposals/README.md). These files may change
incompatibly until a stable contract release is declared.
