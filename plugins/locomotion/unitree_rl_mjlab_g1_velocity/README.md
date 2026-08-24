# Unitree RL MJLab G1 velocity-v0 seam

This experimental plugin records a clean Longship boundary for Unitree's
official G1 29-DoF velocity policy at the pinned upstream revision in
`plugin.experimental.json`. No upstream source, checkpoint, SDK, simulator, or
generated evaluation output is copied into this repository.

The implemented adapter accepts the official raw 98-value observation order
and returns one untrusted 29-value raw action candidate. It does **not** convert
that vector to joint targets, open DDS, import the Unitree SDK, or command a
robot. `GuardedPolicyProvider` checks request identity, lease scope, freshness,
shape, horizon, and finite values before a target adapter may see a candidate.

The exact upstream files are reference-locked in
`model-artifacts.experimental.json`. Automated prefetch and activation remain
blocked because the policy file has no separately reviewed weight license.
An operator may verify already-authorized local bytes with `ArtifactStore`, but
the manifest's draft state must not be bypassed to download them.

## Contract snapshot

- input: `obs`, float32 `[1, 98]`, unnormalized raw observation;
- output: `actions`, float32 `[1, 29]`, linear raw action;
- period and maximum candidate horizon: 20 ms (50 Hz);
- resource lease: `whole_body_motion`;
- lease fencing: exact lease ID and monotonically increasing lease epoch,
  revalidated after inference;
- observation order: angular velocity (3), projected gravity (3), velocity
  command (3), gait phase (2), joint position delta (29), joint velocity (29),
  previous raw action (29); and
- velocity profile: x `[-0.5, 1.0]` m/s, y `[-0.5, 0.5]` m/s, yaw
  `[-1.0, 1.0]` rad/s.

The pinned `deploy.yaml`, not ONNX metadata, is the future runtime authority
for offset, scale, gains, and the exact named 29-joint order. A separate
simulator-only low-level
target and qualification evidence are required before activation. Real
hardware support is deliberately outside this change.
