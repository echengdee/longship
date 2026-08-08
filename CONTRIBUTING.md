# Contributing to Longship

Longship is at the foundation stage. Early contributions should make the public
contracts, safety boundaries, and evaluation story clearer before adding broad
implementation scope.

## Before You Contribute

Only submit material that you have the right to publish. Do not contribute
employer or client code, private model weights, datasets, logs, robot
credentials, internal documents, or other confidential material.

Contributions must be:

- independently authored for Longship, or
- derived from a clearly identified public source with a compatible license and
  proper attribution.

If ownership is uncertain, do not upload the material. Open a design issue and
describe the idea at a high level instead.

## Workflow

1. Open an issue before beginning a large change.
2. Keep the proposal narrow and state which contract or public interface it
   affects.
3. Add tests or reproducible evidence appropriate to the change.
4. Document assumptions, limitations, failure behavior, and target scope.
5. Keep large artifacts outside Git and add a manifest when they are required.
6. Submit a focused pull request.

## Contract Changes

Public contracts are the compatibility boundary of the project. A contract
change should include:

- the reason for the change,
- backward-compatibility impact,
- schema and example updates,
- versioning decision, and
- migration notes when applicable.

## Skill and Target Changes

A skill, policy, or target pull request should answer:

- Which contracts does this change use or modify?
- What new risk or failure mode does it introduce?
- Which targets and evaluation gates have been passed?
- How can execution be cancelled or stopped safely?
- How can another contributor reproduce the evidence?

Do not claim real-hardware support without target-specific qualification
evidence.

## Artifact Rules

Git should contain source, schemas, manifests, hashes, documentation,
reproducible scripts, and small license-compatible examples. Model weights,
videos, point clouds, robot logs, large maps, and datasets belong in external
artifact storage.

Never commit secrets, access tokens, private URLs, or credentials.

## Licensing

Unless explicitly stated otherwise, contributions intentionally submitted for
inclusion in Longship are provided under the Apache License 2.0.

