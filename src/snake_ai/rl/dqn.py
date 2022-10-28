import gym

import torch
from snake_ai.wrappers.distance_wrapper import DistanceWrapper, Snake2dEnv
from snake_ai.wrappers.binary_wrapper import BinaryWrapper
from snake_ai.wrappers.relative_position_wrapper import RelativePositionWrapper

# class DQN:
#     def __init__(self, model : torch.nn.Module, lr : float = 1e-3, gamma : float = 0.9, buffer_size : int = 10_000) -> None:
#         self.lr = lr
#         self.gamma = gamma
#         self.model = model
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         self.loss = torch.nn.MSELoss()

class QTrainer:
    def __init__(self, model : torch.nn.Module, lr : float, gamma : float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred : torch.Tensor = self.model(state)

        target = pred.clone()
        for idx, d in enumerate(done):
            Q_new = reward[idx] + (1 - d) * self.gamma * torch.max(self.model(next_state[idx]))
            # if not done[idx]:
            #     Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        output : torch.Tensor = self.loss(target, pred)
        output.backward()

        self.optimizer.step()

if __name__ == '__main__':
    from stable_baselines3.dqn.dqn import DQN
    train = False
    if train:
        rmode = None
    else:
        rmode='human'
    env = Snake2dEnv(render_mode=rmode, width=20, height=20, nb_obstacles=40)

    env = RelativePositionWrapper(env)
    env = DistanceWrapper(env)
    if train:
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100_000)
        model.save("dqn_relative_dist")
    else:
        model = DQN.load("dqn_relative_dist")
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render(rmode)
