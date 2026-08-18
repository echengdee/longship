# Skill Plugins

Skill plugins provide replaceable implementations of bounded semantic
capabilities. They do not turn arbitrary functions, scripts, models, or vendor
APIs into executable robot authority.

See [Skills, Runtime, Navigation, and Target Boundaries](../../docs/architecture/skills-and-runtime.md)
for the complete responsibility model.

## What belongs here

Use `plugins/skills/<skill-plugin>/` when a semantic Skill implementation is
optional, independently versioned, or has dependencies that should remain
outside Longship core. Suitable examples include a specialized grasp Skill, a
tour-guide composition, or a replaceable object-search implementation.

Provider integrations that are not themselves mission-level capabilities
belong under their provider kind instead:

```text
plugins/navigation/nav2/       # navigation stack provider
plugins/navigation/<provider>/ # alternative navigation stack
plugins/speech/asr/<provider>/  # speech recognition
plugins/speech/tts/<provider>/  # speech synthesis
plugins/targets/<target>/       # robot or simulator translation
```

In particular, `navigation.navigate_to` is a semantic Skill. Mapping,
localization, planning, obstacle handling, and controller execution are the
navigation provider/subsystem behind that Skill. Nav2 belongs in a navigation
provider plugin, not in the portable mission contract.

## Required shape

A Skill plugin should contain only the small, reviewable integration needed by
Longship. A recommended package is:

```text
plugins/skills/<skill-plugin>/
  README.md
  plugin.json                 # version, kind, contracts, dependencies, maturity
  src/                        # implementation or adapter
  tests/                      # contract and cancellation tests using mocks
  examples/                   # non-hardware example calls
  qualification/             # compact reports or immutable report references
  THIRD_PARTY.md              # upstream source and license notices when needed
```

Do not copy the canonical Skill schema into the plugin. Reference the
compatible contract version from the manifest. Keep large models, datasets,
maps, bags, media, and runtime images in external artifact storage and record
their immutable digests.

## A Skill is more than a method

Registering a Python method is insufficient for physical execution. Each
exposed Skill must declare:

- a stable Skill ID, contract version, and implementation version;
- typed arguments and typed results;
- preconditions and postconditions;
- shared and exclusive resource claims;
- timeout and failure behavior;
- bounded cancellation and named safe points;
- progress, success, failure, and safe-point evidence;
- risk, authorization, and confirmation requirements; and
- provider, artifact, target, and Safety-profile qualification scope.

Runtime binds those declarations to one invocation, target, graph revision,
resource lease set, cancellation epoch, and deadline. The plugin cannot grant
itself a lease, mutate canonical Runtime state, declare its own output trusted,
or bypass Command Arbitration and Safety.

## Runtime relationship

Runtime owns the mission DAG, admission, dependency barriers, resources,
parallel execution, timeouts, cancellation, switching, versions, authoritative
state, and recovery. A Skill plugin owns its capability algorithm and reports
typed progress and evidence.

This separation allows compatible work to overlap:

```text
interaction.say + navigation.navigate_to
```

`say` normally claims the speaker while `navigate_to` claims base motion, so
Runtime may schedule both. It also prevents incompatible work from being
blended:

```text
navigation.navigate_to + navigation.follow_person
```

Both normally claim `base_motion` exclusively, so Runtime must serialize,
reject, or switch them at a declared safe point.

Plugins must not start hidden child Skills. Multi-Skill behavior belongs in a
visible mission subgraph or compiled composite so Runtime can account for every
resource, timeout, and cancellation path.

## Navigation and target rules

A navigation Skill should pass approved semantic destinations, map revisions,
deadlines, and authority epochs to the selected navigation provider. It should
return correlated progress and arrival evidence. It must not expose arbitrary
velocity, pose, ROS topic, shell, or vendor SDK control to a Brain.

The navigation provider may perform mapping, localization, planning, and
controller work. Its action-producing output remains bounded by Runtime leases
and flows through arbitration and Safety. Only the Target adapter translates
approved commands into target-specific frames, messages, SDK calls, or
actuation. A provider plugin must not open a parallel hardware command path.

Before a Skill plugin can run on a real target, its exact contract,
implementation, provider configuration, artifacts, target adapter, and Safety
profile require target-specific qualification evidence. Mock or simulation
success is useful evidence, not automatic hardware approval.
