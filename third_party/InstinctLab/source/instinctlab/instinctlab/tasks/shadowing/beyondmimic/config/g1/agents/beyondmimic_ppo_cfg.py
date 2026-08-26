import os

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class BeyondMimicPolicyCfg(InstinctRlActorCriticCfg):
    """BeyondMimic policy configuration - simple MLP following their approach."""

    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


@configclass
class BeyondMimicAlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    """BeyondMimic PPO algorithm configuration."""

    class_name = "PPO"

    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class BeyondMimicNormalizersCfg:
    """BeyondMimic normalizers configuration."""

    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class G1BeyondMimicPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    """G1 BeyondMimic PPO runner configuration."""

    policy: BeyondMimicPolicyCfg = BeyondMimicPolicyCfg()
    algorithm: BeyondMimicAlgorithmCfg = BeyondMimicAlgorithmCfg()
    normalizers: BeyondMimicNormalizersCfg = BeyondMimicNormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 1000
    log_interval = 10
    experiment_name = "g1_beyondmimic"

    load_run = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = self.load_run is not None

        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )
