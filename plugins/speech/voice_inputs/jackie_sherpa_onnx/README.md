# Jackie local voice input (reserved integration seam)

This directory records the intended on-device voice-input plugin for Jackie.
It does not download, copy, or activate microphones or model artifacts.

The plugin owns one microphone stream and composes two local branches:

```text
local PCM capture
  +-> always-on reserved-STOP keyword spotting -> safety-only event
  +-> Jackie keyword spotting: Hey Jackie / Jackie / 你好杰基
        -> VAD-delimited utterance
        -> local ASR transcript
        -> typed VoiceInputEvent stream
```

The first reference candidate is sherpa-onnx because its public interfaces
cover customized Chinese/English keyword spotting, VAD, ASR, Linux, and common
edge architectures. A deployment may select another implementation behind the
same `VoiceInputPort`.

## Privacy boundary

- The always-on paths perform only Jackie and reserved-STOP keyword spotting
  on the robot; ordinary dictation ASR opens after Jackie.
- A short pre-roll buffer lives only in memory and is overwritten continuously.
- Raw ambient audio is not persisted or sent to a Brain provider.
- Only a post-wake final transcript may enter the semantic interaction route.
- Provider, model, keyword, threshold, microphone, and evaluation revisions are
  deployment artifacts rather than hidden defaults.

## Control boundary

Wake detection opens a dictation session; it grants no task, filesystem,
network, Skill, or actuator authority. Final transcripts enter Longship's local
interaction router. Fixed controls remain deterministic, and open-ended text
may reach a non-actuating Brain provider only after that routing step.

Reserved stop phrases do not require a wake word. A dedicated local STOP
spotter, or a partial transcript after wake, may request Longship's
protective-stop path from any voice-session state, including while a Brain or
TTS is busy. The STOP spotter must emit only the safety-path hypothesis and
cannot open dictation. This software path does not replace a physical emergency
stop or target-side safety monitor.

## Artifact policy

Keep these outside Git and resolve them by immutable digest before startup:

- keyword-spotting, VAD, and ASR weights;
- tokens, lexicons, keyword files, and normalization data;
- microphone-array, echo-cancellation, and noise-suppression profiles; and
- target-specific latency, false-accept, false-reject, and word-error evidence.

Review framework and model licenses independently. An open-source runtime
license does not automatically grant redistribution rights for every weight.
Live missions must never trigger a first-time model download.

Activation remains blocked until one deployment profile supplies those locks,
passes real-microphone evaluation, and demonstrates that robot speech cannot
self-trigger the wake path.

Upstream references:

- <https://github.com/k2-fsa/sherpa-onnx>
- <https://k2-fsa.github.io/sherpa/onnx/kws/index.html>
- <https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html>
