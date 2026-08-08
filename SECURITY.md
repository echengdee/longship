# Security and Physical Safety

Robotics software can affect physical machines and people. Treat every command,
adapter, model, and configuration change as potentially safety relevant.

## Reporting a Vulnerability

Do not publish exploit details, credentials, or unsafe reproduction steps in a
public issue.

Use GitHub Private Vulnerability Reporting when it is available. If it is not
available, open a minimal public issue asking maintainers to establish a private
channel; do not include sensitive details in that issue.

## Safety Expectations

- High-level AI must not directly control joints, torques, actuators, or safety
  overrides.
- Commands must be bounded, attributable, cancellable, and time limited.
- A safety layer must be able to veto or stop execution independently.
- Simulation success does not qualify a capability for real hardware.
- Real-robot tests require target-specific limits, supervision, and an
  accessible emergency stop.
- Candidate lessons or generated code must not promote themselves into trusted
  or production behavior.

## Current Support Status

Longship is at the foundation stage. No released component is currently
production-ready or safety-certified.

