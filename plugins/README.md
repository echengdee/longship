# Plugins

Longship is designed to use plugins so the core can remain independent of specific models,
simulators, sensors, and robots.

Planned plugin kinds include brain adapters, knowledge sources, skills, policy
adapters, target adapters, and evaluators.

Every plugin will provide a machine-readable manifest similar to:

```yaml
plugin_id: longship.target.mock
plugin_version: 0.1.0
api_version: 1.0.0
kind: target

contracts:
  world_state: 1.x
  command: 1.x

supported_targets:
  - mock

maturity: draft
```

The manifest format is still a proposal. It will be versioned before plugins
are accepted as compatible.
