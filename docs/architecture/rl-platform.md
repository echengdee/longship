# RL Platform Architecture

> **Status:** Initial executable boundary. The configuration loader and
> registries exist; concrete neural networks and upstream framework adapters
> are intentionally not claimed as implemented.

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
src/longship/rl/
├── models/
│   ├── policies/       # top-level forward graph
│   ├── encoders/       # observation and modality encoding
│   ├── backbones/      # reusable feature processing
│   ├── decoders/       # actor, value, Q, and motion outputs
│   └── distributions/  # action distributions
├── training/
│   └── backends/       # HoloSoma, InstinctLab, SONIC adapters
├── data/               # data contracts and loaders
├── sim2sim/            # secondary-simulator runners and adapters
└── deploy/             # exporters and target adapters
```

Python implementations live under `src/longship/rl`. The repository root does
not create a second experiment hierarchy:

```text
outputs/                # ignored generated runs and resolved configs
third_party/            # pinned upstream source snapshots
```

Experiments stay with the selected RL integration. A backend adapter translates
that experiment into Longship's validated configuration boundary rather than
copying every upstream recipe into a root-level `experiments/` directory.

Data, Sim2Sim, and deployment implementations each have exactly one package
under `src/longship/rl`. Their run parameters will be referenced by an
experiment or supplied to the corresponding command; they do not create a
second top-level directory tree.

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

Registrations are separated by kind:

- `policy`, `encoder`, `backbone`, and `decoder` for models;
- `training_backend` for upstream training frameworks;
- future `sim2sim_runner`, `exporter`, and `deploy_target` adapters.

This prevents a same-named component from crossing architectural boundaries.

## Reproducibility

`ExperimentRunner` validates the recipe, creates a new output directory, and
writes `resolved.yaml` before handing control to the training backend. The
runner refuses to reuse an existing output directory, preventing accidental
checkpoint and configuration overwrite.

The source trees currently stored under `third_party/` are upstream references.
Concrete adapters must be independently reviewed and registered before an
experiment becomes executable.
