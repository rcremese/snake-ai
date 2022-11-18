##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-18 4:58:15 pm
 # @copyright https://mit-license.org/
 #
from stable_baselines3.dqn import DQN
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from snake_ai.envs import SnakeGoalEnv
from snake_ai.wrappers import RelativePositionWrapper, DistanceWrapper

train = False
env = SnakeGoalEnv(render_mode="human", width=20, height=20, nb_obstacles=20)
env = RelativePositionWrapper(env)
env = DistanceWrapper(env)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "final" # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = 500

# Initialize the model
model = DQN(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=1,
)
if train:
    # Train the model
    model.learn(50_000)
    model.save("her_dqn_snake_env")
else:
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    model = DQN.load("her_dqn_snake_env", env=env)

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render(mode="human")
        if done:
            obs = env.reset()