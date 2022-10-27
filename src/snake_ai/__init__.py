from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake_ai.envs.snake_2d_env:Snake2dEnv',
    max_episode_steps=500,
)