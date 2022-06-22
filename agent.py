##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-16 12:18:03 pm
 # @copyright https://mit-license.org/
 #
import torch
import random
from collections import deque
from snake_game_ai import SnakeGameAI, PIXEL_SIZE
from utils import Direction
from model import Linear_QNet, QTrainer
from pathlib import Path

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
EXPLORATION_THREASHOLD = 100
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        # head = game.snake.head
        # point_l = Point(head.x - 20, head.y)
        # point_r = Point(head.x + 20, head.y)
        # point_u = Point(head.x, head.y - 20)
        # point_d = Point(head.x, head.y + 20)
        point_l = game.snake.head.move(-PIXEL_SIZE, 0)
        point_r = game.snake.head.move(PIXEL_SIZE, 0)
        point_u = game.snake.head.move(0, -PIXEL_SIZE)
        point_d = game.snake.head.move(0, PIXEL_SIZE)

        dir_l = game.snake.direction == Direction.LEFT
        dir_r = game.snake.direction == Direction.RIGHT
        dir_u = game.snake.direction == Direction.UP
        dir_d = game.snake.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.snake.head.x,  # food left
            game.food.x > game.snake.head.x,  # food right
            game.food.y < game.snake.head.y,  # food up
            game.food.y > game.snake.head.y  # food down
            ]
        # print('obstacles', np.array(state[:3], dtype=int))
        # print('direction', np.array(state[3:7], dtype=int))
        # print('food', np.array(state[7:], dtype=int))
        return np.array(state, dtype=int)

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
        final_move = [0,0,0]
        if random.randint(0, 2 * EXPLORATION_THREASHOLD) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # TODO :  change the way fonctions are computed
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    # Si un model_0 existe, le lire
    model_path = Path.cwd().joinpath('model','model_0.pth').resolve()
    if model_path.exists():
        agent.model.load(model_path.as_posix())
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game._reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)


if __name__ == '__main__':
    train()