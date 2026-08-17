# Local Codex brain (experimental)

This optional adapter uses the official `openai-codex` Python SDK and existing
local Codex authentication. It keeps a read-only, ephemeral thread alive for
short-term conversation, but Longship remains the authority for runtime state
and memory.

"Local" describes the SDK/app-server process, not offline model inference.
User text and the runtime snapshot may be sent to the Codex service configured
for the account. Prompts must therefore exclude private telemetry, maps,
company content, and credentials.

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

Codex SDK documentation currently describes Codex as a coding-focused agent.
For low-latency production conversation, keep ASR, command grammar, and TTS as
separate providers and evaluate a dialogue-specialized model behind the same
non-actuating boundary.
