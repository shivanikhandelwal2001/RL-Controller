from gymnasium.envs.registration import register
from hvac_data import get_hvac_data

df = get_hvac_data()

register(
    id="CustomEnv-v0",
    entry_point="custom_envs.envs.custom_env:HVACEnv",
    kwargs={"df": df},
)
