from gymnasium.envs.registration import register

register(
    id='SmartTetris-v0',
    entry_point='tetris_game.envs:TetrisEnv',
    max_episode_steps=10
)