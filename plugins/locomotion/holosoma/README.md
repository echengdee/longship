# Holosoma G1 locomotion contract seam

This plugin records where the pinned Amazon FAR Holosoma framework can join
Longship's ecosystem loop: offline training, retargeting, export, evaluation,
and—only after separate review—a bounded inference provider. The upstream
framework is not copied into this monorepo and is not imported by Longship's
production runtime.

Holosoma's current G1 locomotion examples expose 100 observation values and a
29-value joint-residual action at 50 Hz. The contract applies the exported
`0.25` base-angular-velocity and `0.05` joint-velocity scales before inference;
target-side position conversion is separately defined as
`q_target = q_default + 0.25 * raw_action`. The action spans both legs, waist,
and arms, so it claims `whole_body_motion`; it cannot share the robot with the
high-level Unitree locomotion target or an upper-body FSM/PD controller unless
a separately qualified composition changes those ownership boundaries.

The upstream repository contains candidate FastSAC and PPO ONNX files, but they
have no separate weight-license statement, complete model card, or Longship
target qualification. The default FastSAC identity is recorded in a **draft**
artifact manifest so authorized local bytes can be verified; that manifest is
not approved for automatic download or activation. No target commands are
implemented.

A future activatable plugin should run in simulation first, produce the shared
`PolicyCandidate` type, declare explicit raw-action bounds, bind every
observation and action field to the exact named 29-joint order, and pass
freshness, lease-epoch, fallback, and promotion gates before any protected
hardware trial.
