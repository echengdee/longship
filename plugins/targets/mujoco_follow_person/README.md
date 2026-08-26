# MuJoCo FollowPerson target

This optional target plugin runs the same Longship FollowPerson Runtime, local
planner, Safety guard, command expiry, and scenario contract against a visible
MuJoCo world. The green body is the synthetic person, blue body is a planar
robot proxy, and red cylinders are physical obstacles.

It is deliberately a compact base-motion physics proxy. It validates command
timing, world-frame motion, turning, collision avoidance, target loss, and
reacquisition. It does **not** claim Unitree G1 balance, joint-control, policy,
camera rendering, or sim-to-real qualification. A future G1 locomotion target
can replace this plugin without changing the FollowPerson Runtime.

Install the optional dependency into the active environment:

```bash
python3 -m pip install -e '.[mujoco]'
```

Run the headless physics acceptance:

```bash
python3 plugins/targets/mujoco_follow_person/runner.py \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --system
```

Run the viewer at wall-clock speed and keep the final frame open:

```bash
python3 plugins/targets/mujoco_follow_person/runner.py \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --system --viewer --keep-viewer
```

Open the Longship-native persistent interaction terminal beside the viewer:

```bash
python3 plugins/targets/mujoco_follow_person/runner.py \
  --profile scenarios/follow_person/profile.v0.json \
  --scenario scenarios/follow_person/closed_loop.v0.json \
  --stack --viewer
```

Type `跟着我走`, `状态`, `暂停`, `继续`, or `停止`. This terminal injects
text through `RuntimeTextPort`; it never commands the target directly. Add
`--brain codex` only after installing the optional Codex provider when a
model-backed semantic decision is explicitly in scope. STOP and all fixed
controls continue to bypass the Brain.

The process exits nonzero if the Runtime fails, a forward command violates raw
clearance, the robot touches an obstacle, or the final standoff gate fails.
With `--system`, acceptance also requires instruction input, a Brain proposal,
one admitted semantic Skill call, Runtime evidence, target commands, and the
safety-only operator STOP path. Omit `--system` only for focused control tests.
