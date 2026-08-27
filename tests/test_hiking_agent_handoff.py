from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np

from longship.rl.sim2sim.adapters.instinctlab_dds import _handoff_agent
from longship.rl.sim2sim.control import PolicyControl
from longship.rl.sim2sim.hiking_pipeline import HikingOnnxPolicy


def test_handoff_clears_velocity_and_resets_only_incoming_agent() -> None:
    outgoing_action = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
    outgoing = SimpleNamespace(last_action=outgoing_action)
    received: list[np.ndarray] = []
    incoming = SimpleNamespace(reset_history=lambda action: received.append(action.copy()))
    policies = {"parkour": outgoing, "stand": incoming}
    control = PolicyControl(np.zeros(29))
    control.lin_x = control._target_lin_x = 0.4
    control.yaw = control._target_yaw = -0.5

    _handoff_agent(policies, control, "parkour", "stand")

    assert (control.lin_x, control.lin_y, control.yaw) == (0.0, 0.0, 0.0)
    assert (control._target_lin_x, control._target_lin_y, control._target_yaw) == (
        0.0,
        0.0,
        0.0,
    )
    assert len(received) == 1
    np.testing.assert_array_equal(received[0], outgoing_action)


def test_policy_history_reset_preserves_handoff_action() -> None:
    policy = HikingOnnxPolicy.__new__(HikingOnnxPolicy)
    policy.histories = [deque([np.ones(3)], maxlen=8) for _ in range(6)]
    policy.depth_history = deque([np.ones((18, 32))], maxlen=37)
    policy.last_action = np.zeros(29, dtype=np.float32)
    outgoing_action = np.arange(29, dtype=np.float32)

    policy.reset_history(outgoing_action)

    assert all(not history for history in policy.histories)
    assert not policy.depth_history
    np.testing.assert_array_equal(policy.last_action, outgoing_action)
    assert policy.last_action is not outgoing_action


def test_same_agent_selection_is_a_noop() -> None:
    policy = SimpleNamespace(last_action=np.ones(29), reset_history=lambda _: None)
    control = PolicyControl(np.zeros(29))
    control.lin_x = 0.3

    _handoff_agent({"stand": policy}, control, "stand", "stand")

    assert control.lin_x == 0.3
