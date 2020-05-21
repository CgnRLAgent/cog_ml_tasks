from gym.envs.registration import register

register(id="12_AX-v0", entry_point="gym_cog_ml_tasks.envs:AX_12_ENV")
register(id="12_AX_S-v0", entry_point="gym_cog_ml_tasks.envs:AX_S_12_ENV")
register(id="AX_CPT-v0", entry_point="gym_cog_ml_tasks.envs:AX_CPT_ENV")
register(id="12_AX_CPT-v0", entry_point="gym_cog_ml_tasks.envs:AX_12_CPT_ENV")

register(id="seq_prediction-v0", entry_point="gym_cog_ml_tasks.envs:seq_prediction_ENV")

register(id="Simple_Copy-v0", entry_point="gym_cog_ml_tasks.envs:Simple_Copy_ENV")
register(id="Simple_Copy_Repeat-v0", entry_point="gym_cog_ml_tasks.envs:Simple_Copy_Repeat_ENV")