# Unitree RL Lab locomotion provider (reserved integration seam)

This directory records the intended boundary for an external Unitree G1
velocity policy. It does **not** download, copy, or activate a checkpoint.

The official Unitree RL Lab repository exposes G1 velocity-policy training and
deployment workflows. A concrete Longship provider still needs an exact public
checkpoint, observation/action contract, normalization statistics, weight
license, immutable URI, SHA-256, simulator evidence, and target qualification.
Until those facts exist, activation is blocked and redistribution is
`reference_only`. The adjacent experimental manifest makes that blocked state
machine-readable; its null artifact is intentional.

The reviewed upstream `Unitree-G1-29dof-Velocity` configuration uses a
29-action joint policy, including arm and wrist joints. A provider for that
artifact must therefore own `whole_body_motion`, not only `legs` and `waist`.
It is mutually exclusive with the onboard high-level locomotion service and
must never be switched during motion. A separately trained lower-body policy
would require its own manifest and qualification evidence. Navigation remains
upstream and emits fresh, bounded velocity setpoints through a
target-independent contract.

Planned gate:

```text
exact artifact + license review
  -> sim-to-sim
  -> replay and freshness/lease tests
  -> protected sim-to-real
  -> supervised target qualification
```

Upstream reference:

- <https://github.com/unitreerobotics/unitree_rl_lab>
- reviewed revision `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`
- repository code license at review time: Apache-2.0

The repository code license is not assumed to be the license of any external
checkpoint. Model bytes remain outside Git.
