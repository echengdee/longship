# Unitree G1 high-level target (experimental)

This thin adapter calls the official SDK2 Python `LocoClient` high-level
service. It does not contain Unitree source, firmware, robot assets, or model
weights. The onboard locomotion service remains responsible for walking; this
plugin only submits bounded base-frame velocity setpoints.

Safety defaults are intentionally restrictive:

- real hardware is disabled until explicitly enabled;
- velocity limits default to zero and must be set by a target-qualified profile;
- a monotonic, expiring **process-local Longship ownership token** is required;
  the reviewed SDK client does not enable Unitree's service-side lease feature;
- a long-lived Runtime may refresh only the same active lease ID, epoch, and
  actuator scope with a newer issue time and another bounded TTL; refresh
  cannot revive an expired or stop-latched lease;
- the onboard service is asked to use at most a 250 ms command duration, and an
  absolute-deadline watchdog requests zero velocity and latches on missed refresh;
- the configured SDK timeout is a best-effort hint, not a wall-clock bound; the
  shim cannot preempt a synchronous SDK call and is not a real-time or
  safety-rated stop channel;
- non-finite, stale, wrong-frame, wrong-lease, and over-limit commands fail closed;
- stop atomically revokes the active epoch and latches out further commands;
- `Move(..., continuous_move=True)`, `Damp()`, and `ZeroTorque()` are never called;
- an SDK success code records only an accepted RPC, not verified execution; and
- the locomotion lease is retained until a qualified monitor supplies a
  post-transport dwell window correlated to the stop generation, target, boot,
  monotonic clock, and lease, with bounded base, yaw, roll/pitch, and joint
  velocities;
- a physical E-stop and fresh target-state safety monitor remain mandatory.

The onboard high-level locomotion service is conservatively assigned the
`whole_body_motion` resource because firmware/policy behavior may include arm
swing or posture. It must not run concurrently with an upper-body controller
until a target-specific qualification proves disjoint actuator ownership.

A blocking SDK call cannot be safely preempted through the same client. STOP
latches the local command epoch immediately, but zero velocity may still wait
behind an in-flight call for longer than the timeout hint. Latch reset takes the
same RPC barrier and rejects evidence sampled before transport quiescence.
Independent hardware safety is therefore a requirement, not a fallback
feature.

The public test suite uses an injected fake client. A supervised deployment may
install the reviewed official SDK separately and call `connect_unitree_g1`.
No public CI job sends commands to a robot.

See [THIRD_PARTY.md](THIRD_PARTY.md) for the reviewed upstream reference.
