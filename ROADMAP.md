# Roadmap

Longship follows evidence, not calendar promises. A phase is complete when its
public outputs are reviewable, reproducible, and useful as foundations for the
next phase.

## Foundation: Current and Planned Outputs

- Public mission and architecture
- Clean-room contribution rules
- Apache-2.0 licensing
- Initial contract proposals
- Plugin-manifest proposal
- Architecture decision process

## Phase 1: Minimal Open Runtime

Current experimental evidence:

- a deterministic voice-tour FSM runs on a mock navigation target,
- reserved stop and fixed operator controls bypass optional Brain providers,
- travel speech and motion overlap under separate conceptual resources,
- arrival evidence gates exhibit narration,
- the reference Codex Brain is opt-in and restricted to high-level proposals,
- an experimental Unitree G1 high-level target seam is hardware-off by default.

These are V0 interfaces used to learn before the corresponding public contracts
are stabilized. They do not yet satisfy the Phase 1 exit criteria or establish
real-target qualification.

Planned outputs:

- versioned contract schemas and examples,
- plugin SDK and compatibility checks,
- mock target,
- local streaming VAD/ASR and TTS plugin boundaries,
- deterministic end-to-end benchmark,
- reference safe-stop skill,
- structured experience episodes, and
- a warehouse scenario manifest with replayable synthetic examples.

Exit criteria:

- another contributor can validate the schemas,
- a mock mission can run without proprietary dependencies,
- failure and cancellation paths are testable, and
- the same episode produces reproducible evaluation results.

## Phase 2: Knowledge and Experience

Planned outputs:

- knowledge ingestion with source provenance,
- deterministic context compilation,
- Longship-owned durable memory, summaries, and relevance retrieval,
- experience recording and failure classification,
- simulation adapters, and
- additional public scenario packs.

## Phase 3: Evaluated Evolution

Planned outputs:

- curriculum and candidate generation,
- multi-model and multi-target benchmarks,
- evidence-based promotion gates, and
- rollback workflows.

## Not in the Initial Release

- automatic deployment to real robots,
- production fleet management,
- unreviewed self-modification,
- proprietary model weights or private datasets, and
- claims of hardware qualification without published criteria.
