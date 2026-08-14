# Schemas

This directory will contain versioned Longship contracts.

The current public discussion set is in [`proposals/`](proposals/) and covers
semantic intent, deterministic runtime control, protective stop, control
results, parallel mission graphs, execution/Brain continuity, role-scoped model
sessions, runtime events, telemetry, model artifacts, and experience.

These are draft JSON Schema 2020-12 documents, not stable releases or
hardware-qualification claims. Application schemas do not replace physical
emergency-stop circuits, target-specific controllers, Runtime transaction
checks, or Safety qualification.

Before a contract is promoted to a stable namespace it needs:

1. positive and negative synthetic examples;
2. parser and validator conformance tests;
3. explicit authority, ownership, timing, and failure semantics;
4. Runtime checks for cross-document and transactional invariants;
5. security, privacy, license, and safety review;
6. simulation or mock-target evaluation; and
7. target-specific qualification where physical actuation is involved.

Large media, logs, datasets, model weights, policy checkpoints, and compiled
engines are referenced as immutable external artifacts rather than committed
here.
