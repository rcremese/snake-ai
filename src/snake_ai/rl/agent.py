##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-16 12:18:03 pm
 # @copyright https://mit-license.org/
 #
import torch
import torch.nn.functional as F
import collections
import random
from pathlib import Path

from snake_ai.envs import Snake2dEnv
from snake_ai.wrappers.binary_wrapper import BinaryWrapper
from snake_ai.rl.dqn import QTrainer
from snake_ai.models.mlp import MLP
from snake_ai.wrappers.distance_wrapper import DistanceWrapper
from snake_ai.wrappers.relative_position_wrapper import RelativePositionWrapper

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
EXPLORATION_THREASHOLD = 100
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = collections.deque(maxlen=MAX_MEMORY) # popleft()
        self.model = MLP(input_size=8, output_size=3, hidden_sizes=[256])
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # TODO : Implement exponential decay
        self.epsilon = EXPLORATION_THREASHOLD - self.n_games
        if random.randint(0, 2 * EXPLORATION_THREASHOLD) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # TODO :  change the way fonctions are computed
            move = torch.argmax(prediction).item()

        return move


def train():
    record = 0
    agent = Agent()
    env = Snake2dEnv(render_mode='human', width=20, height=20)
    # env = SnakeBinary(env)
    env = RelativePositionWrapper(env)
    env = DistanceWrapper(env)
    env.metadata['render_fps'] = 50
    # Si un model_0 existe, le lire
    # model_path = Path.cwd().joinpath('model','model_0.pth').resolve()
    # if model_path.exists():
    #     agent.model.load(model_path.as_posix())
    # get old state
    state_old = env.reset()

    while True:
        # get move
        final_move = agent.get_action(state_old)
        # final_move = F.one_hot(torch.tensor(move), num_classes=3)

        # perform move and get new state
        state_new, reward, done, _ = env.step(final_move)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        state_old = state_new

        if done:
            # train long memory, plot result
            state_old = env.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if env.score > record:
                record = env.score
            print('Game', agent.n_games, 'Score', env.score, 'Record:', record)


if __name__ == '__main__':
    train()