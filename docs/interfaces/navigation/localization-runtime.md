# Localization Runtime Bootstrap v0.1

> Status: initial implementation baseline
> Scope: internal Navigation Harness system lifecycle
> Not part of: Mission API, public Localization Engine facade, or policy plugin

## 1. Purpose

`LocalizationRuntime` combines one observation producer, one continuous
localization tick service, and zero or more runtime resources into a single
system-level localization lifecycle:

```text
LocalizationObservationProducer
    → plugin-owned observation ingress

ContinuousLocalizationService
    → Localization Engine internal tick()
    → LocationBelief stream

Mission / Planning / diagnostics
    → get_belief() / wait_for_update()
```

The runtime knows nothing about images, tensors, NoMaD, camera protocols, or map
file formats. Concrete plugins and devices are created only in the composition
root and injected through protocols.

## 2. Injected Interfaces

### 2.1 LocalizationObservationProducer

```python
class LocalizationObservationProducer(Protocol):
    def get_status(self) -> LocalizationObservationProducerStatus: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def wait_stopped(...) -> LocalizationObservationProducerStatus: ...
```

The producer encapsulates a concrete input path such as camera capture, video
replay, or offline dense-frame replay. After `start()` returns, localization
ticks may start. After `stop()` returns, the producer must never submit another
observation. `get_status()` is a non-blocking snapshot. `wait_stopped()` allows
the runtime to supervise the source continuously instead of discovering a dead
input indirectly after the localization context becomes stale.

The generic producer lifecycle is:

```text
CREATED → STARTING → RUNNING → STOPPING → STOPPED
                         ├──→ COMPLETED
                         └──→ FAULTED
```

`COMPLETED` means a finite input ended naturally, `STOPPED` means explicit
shutdown, and `FAULTED` means the input path failed. These outcomes are
distinct.

The producer submits observations to the policy ingress injected at
construction time, not through the public `LocalizationEngine` facade.

### 2.2 LocalizationTickService

```python
class LocalizationTickService(Protocol):
    def get_status(self) -> LocalizationServiceStatus: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def wait_stopped(...) -> LocalizationServiceStatus: ...
```

The current implementation is `ContinuousLocalizationService`. Keeping this
protocol separate lets the runtime test lifecycle and fault handling without
depending on a concrete localization algorithm or policy.

### 2.3 LocalizationRuntimeResource

```python
class LocalizationRuntimeResource(Protocol):
    async def close(self) -> None: ...
```

Resources close in their declared construction order. A NoMaD deployment
normally injects an asynchronous close adapter for the policy executor here.
Future deployments may also inject device handles or read-only map-resource
handles.

## 3. Lifecycle

```text
CREATED
→ STARTING
→ RUNNING
→ STOPPING
→ STOPPED

STARTING / RUNNING / STOPPING
→ FAULTED
```

A stopped runtime instance cannot restart. Switching maps or models requires a
new engine, belief stream, tick service, and runtime so an old mission cannot
consume belief revisions from a new map accidentally.

Startup order is fixed:

```text
observation_producer.start()
→ localization_service.start()
→ monitor observation producer + localization service
```

Shutdown order is fixed:

```text
observation_producer.stop()
→ localization_service.stop()
→ shutdown_resources[n].close()
```

Stopping new observations first freezes the policy context. Waiting for the
current tick next prevents the executor from closing while inference is still
running. Every shutdown action has an independent timeout, and one failure does
not skip subsequent cleanup.

## 4. Fault Semantics

- when the producer enters `FAULTED`, the runtime executes the complete shutdown
  chain and enters `FAULTED`;
- an unsolicited producer transition to `STOPPED` is a runtime fault;
- natural producer `COMPLETED` is also considered abnormal for a continuous
  source by default;
- only a finite replay configured explicitly with `ALLOW_UNTIL_STOP` may leave
  the runtime temporarily `RUNNING` after EOF while the composition layer
  finishes the final belief and calls `stop()`;
- when the localization service enters `FAULTED`, the runtime executes the
  complete shutdown chain and enters `FAULTED`;
- an unsolicited service stop is a runtime fault;
- exceptions from producer or service `wait_stopped()` trigger the same unified
  fault cleanup;
- if any startup phase fails, already-created producers, services, and resources
  are still isolated in reverse order;
- all shutdown errors are aggregated in
  `LocalizationRuntimeStatus.last_error`;
- the runtime never restarts itself in place; an outer System Supervisor decides
  whether to create a new instance;
- Mission reads `LocationBelief` but does not start, stop, or recover the
  runtime.

`LocalizationRuntimeStatus` contains the latest producer and service states so
diagnostics can distinguish input loss, localization-inference failure, and
resource-shutdown failure.

## 5. NoMaD Composition Boundary

```text
Deployment Composition Root
├── NomadTopomap Map Engine adapter
├── NomadDistanceSession
├── NomadVisualGoalDistancePolicy
├── FixedStartVisualLocalizationEngine
├── Camera / DenseReplay observation producer
├── ContinuousLocalizationService
├── policy executor resource adapter
└── LocalizationRuntime
```

Dependency direction remains:

```text
NoMaD plugin → Longship contracts
Composition Root → Longship Runtime + NoMaD plugin
Longship core -X→ NoMaD plugin
```

`LocalizationRuntime` does not sample trajectories, generate control
candidates, or interact with a Route Executor.

## 6. NoMaD Decoded-observation Adapter

The initial NoMaD integration uses two narrow interfaces to separate device or
file decoding from policy ingress:

```python
class DecodedObservationSource(Protocol):
    async def start(self) -> None: ...
    async def read(self) -> DecodedObservationFrame | None: ...
    async def stop(self) -> None: ...

class NomadObservationSink(Protocol):
    def submit_observation(...) -> None: ...
    def clear_observations() -> None: ...
```

A `DecodedObservationSource` implementation owns camera or video decoding and
the original input cadence. `NomadObservationProducer` validates
`image_profile_id`, timestamps, sequence numbers, and tensor representation,
samples on a fixed time grid, and writes the NoMaD four-frame context. When an
input-time discontinuity exceeds the configured threshold, it first clears the
context to avoid combining images from before and after a restart into one
localization window.

The three loops do not serve as clocks for one another:

```text
decoded source:       device/recording cadence, for example 30 Hz
NoMaD context sample: fixed 9 Hz time grid
Localization tick:    scheduled 9 Hz, with overlapping inference forbidden
steady belief output: throttled to 4 Hz; transitions publish immediately
```

By default, NoMaD frame history retains more than four frames. A tick selects
only the latest complete four-frame context satisfying
`observation_time <= requested_at`. This prevents frames arriving after a
request from being attached incorrectly to that older request when observation
production and inference run concurrently.

The FFmpeg implementation is only a recorded-video mock. It lives under NoMaD
`tools/`, never enters the core runtime, and does not imply that a production
camera interface has been selected.
