# External Unitree G1 MuJoCo asset seam

This plugin validates and displays an externally installed Unitree G1 MJCF.
It never copies robot XML, meshes, controller source, or model weights into
Longship. The external bundle is bound by a deterministic SHA-256 over every
regular file plus the separately reviewed license digest.

Asset validation and dynamic FollowPerson qualification are deliberately
different gates. A valid 29-DOF MJCF proves that MuJoCo can load the robot's
geometry, joints, inertias, collisions, sensors, and actuators. It does not
provide a balance controller. The official Unitree MuJoCo simulator exposes
the low-level `LowCmd`/`LowState` surface, whereas Longship's current physical
G1 target uses the onboard high-level `LocoClient` service. Those interfaces
are not interchangeable.

For the locally inspected external bundle, run:

```bash
export UNITREE_MUJOCO_ROOT=/absolute/path/to/unitree_mujoco
export LONGSHIP_G1_PYTHON=/absolute/path/to/python-with-mujoco

python3 plugins/targets/mujoco_g1_external/doctor.py \
  --scene "$UNITREE_MUJOCO_ROOT/unitree_robots/g1/scene_29dof.xml" \
  --license "$UNITREE_MUJOCO_ROOT/LICENSE" \
  --expected-bundle-sha256 9ba04edacbaf9bda13bf847e99e845e9b36b27f7b2141e48ccfc8cae211d1f39 \
  --expected-license-sha256 a5d73fc4aca9074e3e6fe0b1a0ba763cf9514b2249b7390ed20fe8d53630bf25
```

The digest is intentionally an environment-specific lock, not a vendored
asset. Recompute and review it when the external bundle changes. To inspect
the real G1 model without actuating it:

```bash
"$LONGSHIP_G1_PYTHON" \
  plugins/targets/mujoco_g1_external/viewer.py \
  --scene "$UNITREE_MUJOCO_ROOT/unitree_robots/g1/scene_29dof.xml"
```

Do not call this a dynamic FollowPerson pass. Activation remains blocked until
an independently implemented Longship target provider owns all of the
following: an immutable and licensed locomotion policy, its exact observation
and action contract, `LowCmd`/`LowState` transport, command freshness and stop
semantics, robot-state feedback, fall/contact criteria, and replayable
qualification evidence. That provider replaces only the target/locomotion
plugin; the terminal, Brain, Skill, Runtime, Safety, and FollowPerson planner
stay unchanged.

See [THIRD_PARTY.md](THIRD_PARTY.md) for the exact locally inspected provenance
and license record. It is an engineering inventory, not legal advice.
