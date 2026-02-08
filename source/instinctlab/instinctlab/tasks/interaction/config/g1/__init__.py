import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.interaction.config.g1"

gym.register(
    id="Instinct-Interaction-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_interaction_shadowing_cfg:G1ShadowingInteractionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1InteractionShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1InteractionShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Interaction-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_interaction_shadowing_cfg:G1ShadowingInteractionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1InteractionShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1InteractionShadowingPPORunnerCfg",
    },
)
