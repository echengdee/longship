# Voice tour V0

This scenario is the first independently authored runnable vertical slice in
Longship. It uses console text as the ASR boundary, console text as TTS, and a
deterministic mock navigation target. It is not a real-robot qualification.

## Run the mock tour

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
longship-tour scenarios/voice_tour/tour.zh-CN.json
```

Try `start`, `pause`, `resume`, `next`, `repeat`, `status`, `cancel`, and
`stop`. The `zh-CN` scenario also accepts configured localized aliases. A
microphone ASR plugin should
send partial and final transcripts into the same interaction boundary; partial
transcripts are accepted only by the reserved stop grammar.

The mock tour waits for `next` or its configured localized alias after each
narration. Travel announcements use
the speaker resource while navigation uses the base resource, so they may run
at the same time. Exhibit narration waits for explicit mock arrival evidence.

## Reference Codex Brain (opt-in)

The deterministic tour does not need Codex. To route unrecognised language to
the experimental Codex Brain provider:

```bash
pip install -e '.[codex]'
longship-tour scenarios/voice_tour/tour.zh-CN.json --brain codex
```

To request a specific model that is available to the current account:

```bash
longship-tour scenarios/voice_tour/tour.zh-CN.json \
  --brain codex \
  --codex-model gpt-5.6-terra
```

"Local Codex" here means the SDK and app-server run locally; it does not mean
offline or on-device model inference. User text and the small runtime snapshot
may be sent to the Codex service configured for that account, so do not place
private telemetry, maps, company content, or credentials in dialogue prompts.
The official SDK reuses existing local Codex authentication. The provider uses
an empty temporary current working directory, read-only sandbox, and
`ApprovalMode.deny_all`, then returns one strictly validated high-level action.
Timed-out turns are interrupted; if the SDK stream does not reach a terminal
event, the provider replaces the entire app-server session rather than
cancelling its blocking notification waiter.
Those settings are not process isolation: use a dedicated Linux account or
container with no private code mounts, robot credentials, global MCP servers,
or actuator access. Runtime revision checks discard late decisions. No accepted
action carries pose, actuator, SDK, shell, or safety authority; the `message`
field is untrusted speech-only text and is never executed.

ChatGPT Voice in the desktop app can coordinate Codex, but the Codex Python SDK
used here accepts text rather than microphone audio. A robot therefore still
needs independent low-latency VAD/ASR and TTS plugins. The selected model's
context window is model- and account-dependent; it is not durable memory and
does not change the bounded runtime snapshot sent on each turn.

## Moving beyond the mock

`VoiceTourRuntime` depends only on the semantic `NavigationPort`:

```text
approved waypoint ID
  -> NavigationPort (mock now; Nav2 or another provider later)
  -> bounded velocity setpoints
  -> locomotion provider
  -> target adapter
  -> independent safety
```

Do not add raw building poses, private maps, network credentials, or real route
timings to this public scenario. A deployment resolves public waypoint IDs
against its own reviewed map artifact.
