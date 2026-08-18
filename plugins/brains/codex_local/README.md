# Local Codex brain (experimental)

This is Longship's reference high-level Brain when AI is enabled. The optional
adapter uses the official `openai-codex` Python SDK and existing local Codex
authentication. It keeps a read-only, ephemeral thread alive for short-term
conversation, but Longship remains the authority for runtime state and memory.

"Local" describes the SDK/app-server process, not offline model inference.
User text and the runtime snapshot may be sent to the Codex service configured
for the account. Prompts must therefore exclude private telemetry, maps,
company content, and credentials.

| Component | Longship responsibility |
| --- | --- |
| ChatGPT desktop Voice / GPT-Live | Optional desktop operator interface; not the robot audio API |
| Codex Python SDK and app-server | Text-based, high-level Brain thread |
| Selected Codex model | Determines available capability and context window |
| Longship VAD/ASR and TTS plugins | Robot microphone, reserved speech controls, and speaker output |
| Longship memory and Runtime | Canonical history, world state, Skills, task state, and authority |

The adapter sets `ApprovalMode.deny_all`, serializes turns, and explicitly
interrupts a timed-out turn. It does not cancel the SDK's blocking notification
wait: it waits for terminal completion and closes/rebuilds the entire Codex
app-server session if the turn does not quiesce. This prevents a stranded SDK
worker from blocking later console input or process exit. `Sandbox.read_only`
and a temporary current working directory are **not** process isolation: Codex
may still inherit account-level configuration or readable mounts.

The output schema permits only `respond`, `clarify`, `start_tour`,
`continue_tour`, and `status`. Exact-field validation and a runtime revision
check happen after every turn. No accepted action has actuator, SDK, shell, or
safety authority. The arbitrary `message` string is treated as untrusted
speech-only text and is never executed.

Reserved stop and deterministic operator controls bypass this provider. A
deployment must run it in a dedicated container or low-privilege Linux account,
with no private repository mounts, robot credentials, device nodes, DDS/ROS
control domain, target sockets, global MCP configuration, or actuator-capable
tools. This is especially important when the robot also contains employer code.

Codex is an agent runtime over a selected model. Context capacity, latency, and
availability are properties of that model and account, not guarantees made by
this plugin. A large context window does not replace Longship-owned summaries,
retrieval, current Skill descriptors, or canonical world state.

At the 2026-08-18 documentation review, OpenAI listed
[`gpt-5.6-terra`](https://developers.openai.com/api/docs/models/gpt-5.6-terra)
with a 1,050,000-token context window and no audio modality, while
[`gpt-5.3-codex`](https://developers.openai.com/api/docs/models/gpt-5.3-codex)
had a 400,000-token context window and no audio modality. This is why the
manifest records context as `selected_model` rather than a Codex-wide number.

ChatGPT Voice can coordinate Codex tasks in the ChatGPT desktop app, but that
product feature is powered by GPT-Live and is not an audio transport exposed by
the Codex Python SDK. For robot use, keep microphone capture, VAD, ASR, reserved
command grammar, TTS, and interruption handling as separate low-latency
providers. See the official [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk),
[ChatGPT Voice](https://learn.chatgpt.com/docs/features/voice), and
[model catalog](https://developers.openai.com/api/docs/models) documentation;
recheck model capabilities when preparing a deployment lock.
