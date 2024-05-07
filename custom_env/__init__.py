from gymnasium import register

register(
    id="SmartTetris-v0",
    entry_point='custom_env.envs:TetrisEnv'
)