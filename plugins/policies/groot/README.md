# NVIDIA Isaac GR00T N1.7 reference seam

This directory reserves a provider boundary for the pinned NVIDIA Isaac GR00T
source and model family. It contains no upstream source, weights, processor
files, containers, credentials, or executable provider.

GR00T is treated as a whole-body vision-language-action policy, not as a
locomotion controller or target adapter. Any future integration must emit an
untrusted `PolicyCandidate` through Longship's shared policy boundary. It must
not call a Unitree target directly or run concurrently with another owner of
`whole_body_motion`.

The public `REAL_G1` profile describes two ego-image frames, a 49-value state,
and language input. It emits a 40-step sequence with 53 values per step for
end-effector, hand, arm, waist, base-height, and navigation components. The
public material at the pinned revisions does not completely fix coordinate
frames, units, joint order, navigation semantics, or the per-step period.
Longship therefore cannot assign safe frame offsets or translate this output
to a target command yet.

Activation is blocked for additional reasons:

- the GR00T checkpoint snapshot contains conflicting license descriptions;
- its required Cosmos backbone is gated and needs credentials outside Git;
- a complete reviewed artifact hash set is not available anonymously;
- the upstream package requires an isolated Python 3.12 environment; and
- no Longship target qualification or safe fallback evidence exists.

If those gates are resolved, run the provider as a pinned, offline sidecar on
a loopback or Unix-domain transport. Do not expose the upstream unauthenticated
ZeroMQ control service on a robot network. SONIC or any task-specific fine-tune
must be registered as a separate provider and artifact lock.
