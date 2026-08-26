# Longship interaction stack composition

`stack` is a process-composition entry point, not a new control architecture.
Longship keeps the same small input and capability boundaries whether text
comes from a terminal, local ASR, a test fixture, or a future UI:

```text
terminal / WakeDictationController
  -> RuntimeTextPort.handle_text
  -> reserved local controls: STOP, pause, resume, status
  -> otherwise FollowBrainPort
  -> version-bound TaskDraft: navigation.follow_person + bounded steps
  -> Runtime validates and compiles a MissionTaskGraph
  -> graph-node admission, deadlines, and base_motion ownership
  -> FollowPerson provider + independent Safety
  -> expiring target-independent velocity command
  -> selected target adapter
  -> measured target/world state -> next FollowScene
```

The terminal does not publish velocity, call a vendor SDK, or own the control
loop. It only injects text. A reserved STOP runs on a separate local task and
can overtake a slow Brain turn. Partial ASR is safety-only and cannot call a
Brain. The Runtime rejects late Brain output when its mission revision has
changed.

The current FollowPerson composition exposes two Brain providers:

- `deterministic` is offline and reproducible. It recognizes only a bounded
  FollowPerson intent, produces one open-ended `follow` step, and is the
  default test provider.
- `codex` is an optional experimental semantic provider. Its output schema can
  only respond or propose `navigation.follow_person` with `follow`, `pause`,
  and `resume` steps; it cannot emit motion, target, shell, or Safety actions.
  Fixed controls still bypass it.

## Executable MissionTaskGraph slice

FollowPerson now executes the first deliberately small slice of Longship's
`TaskDraft -> MissionTaskGraph -> Runtime` design. A model turn does not sleep
and does not own a timer. For example, the input:

```text
跟我走三秒然后暂停，一秒后继续走
```

may produce this untrusted semantic draft:

```text
follow(duration=3s) -> pause(duration=1s) -> resume(duration=open)
```

Runtime binds the proposal to the current Runtime revision, validates its
shape and transition order, and compiles three nodes plus two `after_success`
edges. The control loop admits the first node immediately and advances later
nodes from its monotonic clock. The original terminal command therefore
returns as soon as the graph is admitted. While it runs, new terminal input,
HUD updates, status, and the control loop remain live.

This first executable slice is intentionally a single admission lane:

- one through seven nodes using only `follow`, `pause`, and `resume`;
- the first node is `follow`, every non-final node has a 0.1–60 second
  duration, and the final node is open-ended;
- transitions are `follow -> pause`, `pause -> resume`, or
  `resume -> pause`;
- `base_motion` and `person_tracker` remain owned by the same Skill call for
  the graph's lifetime; and
- malformed drafts, stale revisions, dispatch failures, or unavailable
  operations fail closed.

This is not yet the general parallel DAG, barrier, admission-group, safe-point,
or pending-only graph-patch scheduler described in
`system-architecture-v2.md`. The contract is a truthful executable subset so
the FollowPerson vertical slice does not invent a competing architecture.

Standalone `暂停` or `继续` is an explicit Runtime override: it cancels the
remaining graph and directly changes the active Skill. `停止` is stronger. It
bypasses Brain and task-node scheduling, increments the graph cancellation
epoch, requests the protected target stop path, and cannot be encoded as a
future model-generated node. `状态` reports the graph ID/state, current
operation, and remaining node time without changing execution.

A FollowPerson pause is a zero-velocity hold inside the existing Skill call.
Runtime retains the motion lease and selected locomotion policy and continues
the normal control cadence. It does not switch controllers, freeze joints, or
tear down and recreate policy state; this keeps the short pause path free of
controller-switch latency and reduces transient-stability risk. Node duration
is measured from zero-command admission, not from measured base stationarity.
Protected STOP is intentionally separate and may require a measured settling
and stationary-evidence phase.

The Codex provider is connected through Longship's existing `FollowBrainPort`
and the official local Python SDK/app-server. It reuses the current Codex login
and keeps an ephemeral read-only thread; an MCP hop would add no capability to
this in-process direction. MCP remains a reasonable future adapter when an
external agent host needs to discover and invoke Longship, but it must still
terminate at the same semantic input/Skill boundary rather than the target.
The documented GPT-5.6 profile uses `gpt-5.6-terra` with reasoning effort
`none`; the fixed STOP path remains available while a model turn is pending.

Likewise, the target is replaceable below Safety:

| Composition | Target meaning | What it proves |
| --- | --- | --- |
| `longship-follow stack` | deterministic synthetic world | interactive input through Brain, Skill, Runtime, Safety, command, and feedback |
| MuJoCo plugin `--stack` | Longship-authored planar physics proxy | the same interactive chain plus physics response and contact observation |
| external G1 asset seam | externally installed official-style 29-DOF MJCF | asset identity and loadability only; no locomotion claim |
| external G1 policy target | externally installed Unitree RL Gym 12-joint G1 policy | the same chain with articulated free-base dynamics, fall/contact/base-stop evidence, and optional read-only camera HUD |
| future 29-DOF low-level provider | licensed policy + `LowCmd/LowState` + state evidence | 29-DOF dynamic simulation after separate qualification |
| Unitree high-level target | onboard `LocoClient` walking service | supervised physical path after target qualification |

This keeps simulator and robot dependencies out of core. Adding the G1
low-level provider must replace the target/locomotion plugin; it must not change
the terminal, Brain, semantic Skill, Runtime authority, or independent Safety
layers. A launcher may supervise several provider processes, but it does not
become another mission or actuator authority.

The browser HUD is an observability sink, not an interaction authority. The
launching terminal is the current text-input provider, while the HUD displays
camera frames, environment geometry, target/follow-goal positions, Runtime,
Brain/Skill events, active task graph/node, and target dynamics. A future voice
or browser text-input
provider should call `RuntimeTextPort`; it must not turn the HUD into a direct
motion endpoint.

One stack session owns one FollowPerson lease. `stop`, `exit`, terminal EOF,
target closure, Runtime failure, or supervisor shutdown enters the stop path
and ends the process. Restarting requires a fresh process and lease; a stopped
session is never silently rearmed.
