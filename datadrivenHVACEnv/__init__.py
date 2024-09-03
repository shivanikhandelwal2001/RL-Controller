from gymnasium.envs.registration import register

register(
    id="CustomEnv-v0",
    entry_point="datadrivenHVACEnv.envs.custom_env:HVACEnv",
)
