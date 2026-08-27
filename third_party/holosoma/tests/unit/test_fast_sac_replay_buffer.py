import torch
from tensordict import TensorDict

from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer, set_optimizer_learning_rate


def _transition(n_env: int = 2, n_obs: int = 3, n_act: int = 2, n_critic_obs: int = 4) -> TensorDict:
    return TensorDict(
        {
            "observations": torch.zeros(n_env, n_obs),
            "critic_observations": torch.zeros(n_env, n_critic_obs),
            "actions": torch.zeros(n_env, n_act),
            "next": {
                "observations": torch.ones(n_env, n_obs),
                "critic_observations": torch.ones(n_env, n_critic_obs),
                "rewards": torch.zeros(n_env),
                "dones": torch.zeros(n_env, dtype=torch.long),
                "truncations": torch.zeros(n_env, dtype=torch.long),
            },
        },
        batch_size=(n_env,),
    )


def test_replay_warmup_uses_stored_transitions_after_resume() -> None:
    replay = SimpleReplayBuffer(
        n_env=2,
        buffer_size=16,
        n_obs=3,
        n_act=2,
        n_critic_obs=4,
        device="cpu",
    )

    # A restored training global step is deliberately irrelevant: a new buffer
    # must collect its own warmup transitions before any gradient update.
    assert replay.num_stored == 0
    assert not replay.ready_for_sampling(learning_starts=10)

    for _ in range(10):
        replay.extend(_transition())
    assert replay.num_stored == 10
    assert not replay.ready_for_sampling(learning_starts=10)

    replay.extend(_transition())
    assert replay.num_stored == 11
    assert replay.ready_for_sampling(learning_starts=10)


def test_replay_num_stored_caps_at_capacity() -> None:
    replay = SimpleReplayBuffer(
        n_env=2,
        buffer_size=4,
        n_obs=3,
        n_act=2,
        n_critic_obs=4,
        device="cpu",
    )

    for _ in range(7):
        replay.extend(_transition())

    assert replay.ptr == 7
    assert replay.num_stored == 4
    assert replay.ready_for_sampling(learning_starts=3)


def test_runtime_learning_rate_is_restored_after_optimizer_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    checkpoint_state = optimizer.state_dict()

    resumed = torch.optim.AdamW([parameter], lr=1.5e-4)
    resumed.load_state_dict(checkpoint_state)
    assert resumed.param_groups[0]["lr"] == 3e-4

    set_optimizer_learning_rate(resumed, 1.5e-4)
    assert resumed.param_groups[0]["lr"] == 1.5e-4
