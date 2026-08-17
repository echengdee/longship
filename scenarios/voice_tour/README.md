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

Try `开始导览`, `暂停`, `恢复`, `下一站`, `重复`, `状态`, `取消`, and
`停止`. English aliases are also available. A microphone ASR plugin should
send partial and final transcripts into the same interaction boundary; partial
transcripts are accepted only by the reserved stop grammar.

The mock tour waits for `下一站` after each narration. Travel announcements use
the speaker resource while navigation uses the base resource, so they may run
at the same time. Exhibit narration waits for explicit mock arrival evidence.

## Optional local Codex dialogue

The deterministic tour does not need an LLM. To route unrecognised language to
the experimental local Codex provider:

```bash
pip install -e '.[codex]'
longship-tour scenarios/voice_tour/tour.zh-CN.json --brain codex
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
field is untrusted speech-only text and is never executed. Codex is currently a
coding-focused agent, so this dialogue use is
explicitly experimental rather than a replacement for a low-latency ASR/TTS
stack.

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
