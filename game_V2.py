##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-16 11:46:54 am
 # @copyright https://mit-license.org/
 #
from re import I
import pygame
import random
from utils import Direction
from collections import namedtuple
import numpy as np
from snake import Snake

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0,255,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

PIXEL_SIZE = 20
BODY_PIXEL_SIZE = 12
OBSTACLE_SIZE_RANGE = (2, 4)
SPEED = 50
PERCENTAGE = 0.05
TIME_LIMIT = 100

class SnakeGameAI:

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.snake = Snake(self.width/2, self.height/2, pixel_size=PIXEL_SIZE)
        self.score = 0
        self._populate_grid_with_obstacles(percent=PERCENTAGE)
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        # Define the central coordinates of the food to be placed
        x = random.randrange(0, self.width, PIXEL_SIZE)
        y = random.randrange(0, self.height, PIXEL_SIZE)
        #self.food = Point(x, y)
        self.food = pygame.Rect(x, y, PIXEL_SIZE, PIXEL_SIZE)
        # Check to not place food inside the snake
        if self.food.collidelist(self.snake.body) != -1:
            self._place_food()
        # Check to place the food at least 1 pixel away from a wall 
        if self.food.inflate(2*PIXEL_SIZE, 2*PIXEL_SIZE).collidelist(self.obstacles) != -1:
            self._place_food()

    def _populate_grid_with_obstacles(self, percent : float) -> list[pygame.Rect]:
        if percent < 0 or percent >= 1:
            raise ValueError(f'Enter a valid percentage for obstacle generation. {percent} is out of range [0,1]')
        total_area = self.width * self.height
        area = 0

        self.obstacles = []
        while area < percent * total_area:
            size = random.randint(*OBSTACLE_SIZE_RANGE) * PIXEL_SIZE
            obstacle = self._place_obstacle(size)
            # update area by adding the obstacle area minus the overlapping areas with other obstacles
            area += size**2
            for i in obstacle.collidelistall(self.obstacles):
                # TODO : take care of multi intersection cases
                intersection = obstacle.clip(self.obstacles[i])
                area -= intersection.height * intersection.width
            self.obstacles.append(obstacle)

    def _place_obstacle(self, size : int) -> pygame.Rect:
        x = random.randrange(0, self.width, size)
        y = random.randrange(0, self.height, size)

        obstacle = pygame.Rect(x, y, size, size)
        # check colision with the initial snake bounding box 
        bounding_box_factor = 2 * (self.snake.size -1) * PIXEL_SIZE 
    
        if obstacle.colliderect(self.snake.head.inflate(bounding_box_factor, bounding_box_factor)):
            obstacle = self._place_obstacle(size)
        # check inclusion inside other obstacles
        for i in obstacle.collidelistall(self.obstacles):
            # If the new obstacle is completely contained in an existing obstacle, reprocess
            if self.obstacles[i].contains(obstacle):
                obstacle = self._place_obstacle(size)
        return obstacle


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > TIME_LIMIT*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.snake.head.colliderect(self.food):
            self.score += 1
            reward = 10
            self.snake.grow()
            self._place_food()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_outside(self, rect : pygame.Rect = None):
        if rect is None:
            rect = self.snake.head
        return rect.x < 0 or rect.x + rect.width > self.width or rect.y < 0 or rect.y + rect.height > self.height 

    def is_collision(self, rect : pygame.Rect = None):
        if rect:
            return rect.collidelist(self.obstacles) != -1 or self.is_outside(rect)
        else:
            return self.is_outside() or self.snake.collide_with_itself() or self.snake.collide_with_obstacles(self.obstacles)

    def _update_ui(self):
        self.display.fill(BLACK)

        move = (PIXEL_SIZE - BODY_PIXEL_SIZE) / 2
        # draw the snake
        for pt in self.snake:
            pygame.draw.rect(self.display, WHITE, pt)
            # draw the body of the snake, in order to count the parts
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + move, pt.y + move, BODY_PIXEL_SIZE, BODY_PIXEL_SIZE))
        # Draw food 
        pygame.draw.rect(self.display, GREEN, self.food)
        # plot obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, RED, obstacle)

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]
        # TODO : include possibility to do back turn 
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # for i, direction in enumerate(clock_wise):
        #     # TODO : comprendre pourquoi ça déconne
        #     if direction.value == self.snake.direction.value:
        #         idx = i
        #         break
        #         print('Same dir')
        idx = clock_wise.index(self.snake.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        
        self.snake.move(new_dir)
        # self.direction = new_dir

        # x = self.head.x
        # y = self.head.y
        # if self.direction == Direction.RIGHT:
        #     x += PIXEL_SIZE
        # elif self.direction == Direction.LEFT:
        #     x -= PIXEL_SIZE
        # elif self.direction == Direction.DOWN:
        #     y += PIXEL_SIZE
        # elif self.direction == Direction.UP:
        #     y -= PIXEL_SIZE

        # self.head = Point(x, y)