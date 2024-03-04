from gym.envs.registration import register

register(id='CustomEnv-v0',
    entry_point='custom_envs.envs.custom_env:HVACEnv'
)