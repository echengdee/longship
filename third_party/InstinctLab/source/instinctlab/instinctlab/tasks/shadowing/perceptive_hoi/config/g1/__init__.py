import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.shadowing.perceptive_hoi.config.g1"

gym.register(
    id="Instinct-Perceptive-HOI-Shadowing-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_cfg:G1PerceptiveHoiShadowingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PerceptiveHoiShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveHoiShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-HOI-Shadowing-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_cfg:G1PerceptiveHoiShadowingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PerceptiveHoiShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveHoiShadowingPPORunnerCfg",
    },
)
