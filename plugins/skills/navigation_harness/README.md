# Navigation Harness Skill Plugin

This is the thin plugin boundary that exposes the core navigation harness as
the `navigate_to` Skill. The Mission, Map, Localization, Planning, and Local
Trajectory engines live in `src/longship/navigation`; they are not separate
Longship plugins.

The eventual adapter must:

- translate a Longship Skill call into one navigation mission;
- preserve mission, task, Skill-call, lease, state, and cancellation identity;
- publish bounded progress and a terminal result;
- propagate Runtime pause and cancellation to the Navigation Mission Engine;
- activate route-bound Local Trajectory Engine instances through the reviewed
  runtime composition boundary;
- use `RouteExecutionPort` only after that future external lifecycle is
  implemented and reviewed; and
- never bypass command arbitration, Safety, or the target adapter.

The plugin has no runnable adapter in v0.1. Its Python entry point remains a
Protocol until the generic Longship plugin SDK is stabilized.
