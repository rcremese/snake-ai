from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake_ai.envs.snake_classic_env:SnakeClassicEnv',
    max_episode_steps=500,
)