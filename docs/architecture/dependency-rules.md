# Dependency Rules

These rules keep Longship portable, reviewable, and safe as the project grows.

1. Contracts contain protocols and schemas, not business logic.
2. Core packages do not depend on a specific robot, simulator, model provider,
   middleware vendor, or training framework.
3. Plugins may depend on the core; the core must not depend on individual
   plugins.
4. The runtime does not depend directly on training frameworks.
5. High-level brain adapters may select skills but may not issue low-level
   actuator commands or disable safety.
6. Only target adapters translate approved commands into target-specific
   actions.
7. Candidate lessons from experience are untrusted until reviewed and promoted
   through an evaluation gate.
8. Evolution tools cannot modify or deploy to a production runtime directly.
9. The safety layer must remain effective without a foundation model or network
   connection.
10. A change to a model, map, knowledge artifact, configuration, or binary
    requires a new artifact manifest and content hash.
11. A skill must not run on a real target unless it has passed that target's
    declared qualification gate.

Exceptions require a public architecture decision record that explains the
tradeoff and migration plan.

