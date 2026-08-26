# FollowPerson closed-loop scenario

This synthetic scenario exercises the same provider-neutral FollowPerson core
used by the gated Unitree adapter. A person moves in a small world, one
obstacle forces a local detour, the detector temporarily loses the person, and
the reappearing observation intentionally receives a new short-lived track ID.

Run the deterministic acceptance gate:

```bash
longship-follow simulate \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --events /tmp/longship-follow-simulation.jsonl
```

Run the system-level gate from an instruction event through Brain and Skill
admission to the same closed-loop world:

```bash
longship-follow system-simulate \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --instruction 'Jackie，跟着我走'
```

To watch the same run in real time:

```bash
longship-follow simulate \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --events /tmp/longship-follow-simulation.jsonl \
  --real-time \
  --dashboard-port 8093 \
  --keep-dashboard
```

Open `http://127.0.0.1:8093`. The page is read-only and is never part of the
control loop. The scenario passes only when it spends the declared number of
steps following, emits no forward command inside the nominal stop distance,
converges within the declared final distance error, and avoids a failed Runtime
state.

All people, tracks, paths, obstacles, and events in this scenario are synthetic.

The same scenario can drive the optional MuJoCo target. This adds physical
velocity response and collision contacts while preserving the exact Runtime,
Skill, Safety, and observation contracts:

```bash
python3 plugins/targets/mujoco_follow_person/runner.py \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --system --viewer --keep-viewer
```

See `plugins/targets/mujoco_follow_person/README.md` for dependency and scope
details.

To run this scenario against external articulated G1 dynamics instead of the
planar target, follow the immutable-artifact command in
`plugins/targets/mujoco_g1_policy/README.md`. The scenario and every Longship
component above the target boundary remain unchanged.
