import os
from copy import copy
from torch.backends.cuda import enable_flash_sdp, enable_mem_efficient_sdp

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlEncoderActorCriticRecurrentCfg,
    InstinctRlMlpCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
    InstinctRlTransformerHeadCfg,
)


@configclass
class MotionRefEncoderMlpCfg(InstinctRlMlpCfg):
    hidden_sizes = [
        256,
        128,
    ]
    nonlinearity = "CELU"
    takeout_input_components = True
    output_size = 64

    component_names = [
        # "time_to_target_ref",
        # "time_from_ref_update",
        # "pose_ref",
        # "position_ref",
        # "rotation_ref",
        # "position_ref_mask",
        # "rotation_ref_mask",
        # "joint_pos_ref",
        # "joint_vel_ref",
        # "joint_pos_err_ref",
        # "joint_pos_ref_mask",
        # "link_pos_ref",
        # "link_pos_err_ref",
        # "link_pos_ref_mask",
        # "link_rot_ref",
        # "link_rot_err_ref",
        # "link_rot_ref_mask",
    ]


@configclass
class MotionRefEncoderTransformerCfg(InstinctRlTransformerHeadCfg):
    activation = "gelu"
    nonlinearity = "CELU"
    output_size = 64
    num_heads = 2
    num_layers = 1
    d_model = 64
    dim_feedforward = 64
    output_selection = "maxpool"
    component_names = [
        # "time_to_target_ref",
        # "time_from_ref_update",
        # "pose_ref",
        # "position_ref",
        # "rotation_ref",
        # "position_ref_mask",
        # "rotation_ref_mask",
        # "joint_pos_ref",
        # "joint_vel_ref",
        # "joint_pos_err_ref",
        # "joint_pos_ref_mask",
        # "link_pos_ref",
        # "link_pos_err_ref",
        # "link_pos_ref_mask",
        # "link_rot_ref",
        # "link_rot_err_ref",
        # "link_rot_ref_mask",
    ]


@configclass
class EncoderConfigs:
    motion_ref = MotionRefEncoderMlpCfg()
    # motion_ref = MotionRefEncoderTransformerCfg()


@configclass
class PolicyCfgMixin:
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


@configclass
class EncoderPolicyCfg(PolicyCfgMixin, InstinctRlEncoderActorCriticCfg):
    encoder_configs: EncoderConfigs = EncoderConfigs()
    critic_encoder_configs: EncoderConfigs = EncoderConfigs()


@configclass
class MlpPolicyCfg(PolicyCfgMixin, InstinctRlActorCriticCfg):
    pass


@configclass
class MoEPolicyCfg(PolicyCfgMixin, InstinctRlActorCriticCfg):
    # Approximately 2.5M params
    class_name = "MoEActorCritic"
    actor_hidden_dims = [256, 256, 128]
    critic_hidden_dims = [256, 256, 128]
    num_moe_experts = 8
    moe_gate_hidden_dims = [128, 64]  # Naive MLP for gating: input -> 128 -> 64 -> num_experts


@configclass
class EquivalentMlpPolicyCfg(PolicyCfgMixin, InstinctRlActorCriticCfg):
    # Approximately 2.5M params
    actor_hidden_dims = [1024, 1024, 640]
    critic_hidden_dims = [1024, 1024, 640]


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
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
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class G1ShadowingPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    # policy: MlpPolicyCfg = MlpPolicyCfg()
    policy: MoEPolicyCfg = MoEPolicyCfg()
    # policy: EquivalentMlpPolicyCfg = EquivalentMlpPolicyCfg()
    # ckpt_manipulator = "reinitialize_actor_critic_backbone"
    # ckpt_manipulator = "newStd"
    # ckpt_manipulator = "fit_smaller_mlp_input"
    # ckpt_manipulator_kwargs = {
    #     "weight_removal_slices": {
    #         "actor.0.weight": (585, 585 + 30), # remove 30 dimensions from the input
    #         "critic.0.weight": (15 + 585, 15 + 585 + 30), # remove 30 dimensions from the input
    #     },
    # }
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    log_interval = 10
    experiment_name = "g1_shadowing"

    load_run = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = self.load_run is not None

        self.run_name = "".join(
            [
                "_MLPPolicy" if isinstance(self.policy, MlpPolicyCfg) else "",
                (
                    "_MlpEncoder"
                    if isinstance(self.policy, EncoderPolicyCfg)
                    and isinstance(self.policy.encoder_configs.motion_ref, MotionRefEncoderMlpCfg)
                    else ""
                ),
                "_MoEPolicy" if isinstance(self.policy, MoEPolicyCfg) else "",
                "_EquivalentMlpPolicy" if isinstance(self.policy, EquivalentMlpPolicyCfg) else "",
                (
                    # "_newStd" if self.ckpt_manipulator == "newStd" else "",
                    # ("_obsNorm" if not isinstance(self.normalizers, dict) else ""),
                    # f"_entropyCoeff{self.algorithm.entropy_coef:.0e}" if self.algorithm.entropy_coef else "",
                    f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}"
                    if "CUDA_VISIBLE_DEVICES" in os.environ
                    else ""
                ),
            ]
        )


@configclass
class G1MultiRewardShadowingPPORunnerCfg(G1ShadowingPPORunnerCfg):
    def __post_init__(self):
        return_ = super().__post_init__()
        self.algorithm.advantage_mixing_weights = [0.7, 0.3]
        self.run_name += "_Adv622"
        return return_
