##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-13 3:15:13 pm
 # @copyright https://mit-license.org/
 # TODO : S'assurer qu'il existera toujours un chemin pour atteindre l'objectif

import pygame
import random
import logging
from typing import Dict, List
from snake import Snake
from utils import Direction, Line, get_opposite_direction
from pathlib import Path

pygame.init()
font_path = Path(__file__).parent.joinpath('graphics', 'arial.ttf').resolve(strict=True)
font = pygame.font.Font(font_path, 25)

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 255, 0)
BLACK = (0,0,0)
GREY = (150, 150, 150)

PIXEL_SIZE = 20
BODY_PIXEL_SIZE = 12
OBSTACLE_SIZE_RANGE = (2, 4)
SPEED = 50
PERCENTAGE = 0.2

class SnakeGame:
    def __init__(self, width=640, height=480, speed = 10, obstacle_rate=0):
        self.width = width
        self.height = height
        self._speed = speed
        self._obstacle_rate = obstacle_rate
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self._reset()

    def _reset(self):
        # init game state
        self.snake = Snake(self.width/2, self.height/2, pixel_size=PIXEL_SIZE)
        # self.collision_lines = self._init_collision_lines()
        self.score = 0
        self._populate_grid_with_obstacles(rate=self._obstacle_rate)
        self._place_food()
        self.collision_lines = {}
        self.frame_iteration = 0

    def _place_food(self):
        # Define the central coordinates of the food to be placed
        x = random.randrange(0, self.width, PIXEL_SIZE)
        y = random.randrange(0, self.height, PIXEL_SIZE)
        self.food = pygame.Rect(x, y, PIXEL_SIZE, PIXEL_SIZE)
        # Check to not place food inside the snake
        if self.food.collidelist(self.snake.body) != -1:
            self._place_food()
        # Check to place the food at least 1 pixel away from a wall
        if self.food.inflate(2*PIXEL_SIZE, 2*PIXEL_SIZE).collidelist(self.obstacles) != -1:
            self._place_food()

    def _populate_grid_with_obstacles(self, rate : float) -> List[pygame.Rect]:
        if rate < 0 or rate >= 1:
            raise ValueError(f'Enter a valid percentage for obstacle generation. {rate} is out of range [0,1]')
        total_area = self.width * self.height
        area = 0

        self.obstacles = []
        while area < rate * total_area:
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
        bounding_box_factor = 2 * (self.snake._size -1) * PIXEL_SIZE

        if obstacle.colliderect(self.snake.head.inflate(bounding_box_factor, bounding_box_factor)):
            obstacle = self._place_obstacle(size)
        # check inclusion inside other obstacles
        for i in obstacle.collidelistall(self.obstacles):
            # If the new obstacle is completely contained in an existing obstacle, reprocess
            if self.obstacles[i].contains(obstacle):
                obstacle = self._place_obstacle(size)
        return obstacle


    def _is_outside(self, rect : pygame.Rect = None):
        if rect is None:
            rect = self.snake.head
        return rect.x < 0 or rect.x + rect.width > self.width or rect.y < 0 or rect.y + rect.height > self.height

    def _is_collision(self):
        return self._is_outside() or self.snake.collide_with_itself() or self.snake.collide_with_obstacles(self.obstacles)

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw snake
        self.snake.draw(self.display, WHITE, BLUE2)
        # Draw collision lines
        for direction, line in self.collision_lines.items():
            if line is not None:
                logging.debug(f'Drawing line {line} for direction {direction}')
                line.draw(self.display, GREY, BLUE1)
        # Draw food
        pygame.draw.rect(self.display, GREEN, self.food)
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, RED, obstacle)
        # Print the score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

    def _init_collision_lines(self) -> Dict[Direction, Line]:
        snake_head = self.snake.head
        self.collision_lines[Direction.UP] = Line(snake_head.center, (snake_head.centerx, 0))
        self.collision_lines[Direction.UP_RIGHT] = Line(snake_head.center, (snake_head.centerx + snake_head.centery, 0))
        self.collision_lines[Direction.RIGHT] = Line(snake_head.center, (self.width, snake_head.centery))
        self.collision_lines[Direction.DOWN_RIGHT] = Line(snake_head.center, (self.width, (self.width - snake_head.centerx) + snake_head.centery))
        self.collision_lines[Direction.DOWN] = Line(snake_head.center, (snake_head.centerx, self.height))
        self.collision_lines[Direction.DOWN_LEFT] = Line(snake_head.center, (snake_head.centerx - (self.height - snake_head.centery), self.height))
        self.collision_lines[Direction.LEFT] = Line(snake_head.center, (0, snake_head.centery))
        self.collision_lines[Direction.UP_LEFT] = Line(snake_head.center, (0, snake_head.centery - snake_head.centerx))

    def _get_collision_box(self, direction : Direction):
        snake_head = self.snake.head
        if direction == Direction.UP:
            collision_box = pygame.Rect(snake_head.left, 0, snake_head.width, snake_head.top)
        elif direction == Direction.UP_RIGHT:
            collision_box = pygame.Rect(snake_head.right, 0, self.width - snake_head.right, snake_head.top)
        elif direction == Direction.RIGHT:
            collision_box = pygame.Rect(snake_head.topright, (self.width - snake_head.right, snake_head.height))
        elif direction == Direction.DOWN_RIGHT:
            collision_box = pygame.Rect(snake_head.bottomright, (self.width - snake_head.right, self.height - snake_head.bottom))
        elif direction == Direction.DOWN:
            collision_box = pygame.Rect(snake_head.bottomleft, (snake_head.width, self.height - snake_head.bottom))
        elif direction == Direction.DOWN_LEFT:
            collision_box = pygame.Rect(0, snake_head.bottom, snake_head.left, self.height - snake_head.bottom)
        elif direction == Direction.LEFT:
            collision_box = pygame.Rect(0, snake_head.top, snake_head.left, snake_head.height)
        elif direction == Direction.UP_LEFT:
            collision_box = pygame.Rect(0, 0, snake_head.left, snake_head.top)
        else:
            raise ValueError(f'Unknown direction {direction}')
        logging.debug(f"collision box for direction {direction} is : {collision_box} ")
        return collision_box

    def _update_collision_line(self, direction : Direction):
        collision_box = self._get_collision_box(direction)
        collision_line = self.collision_lines[direction]
        collision_list = [self.obstacles[index] for index in collision_box.collidelistall(self.obstacles)]
        intersection_point = None
        if collision_list:
            # Check all possible collisions within the collision box
            intersection_point = collision_line.intersect_obstacles(collision_list)
            if intersection_point:
                self.collision_lines[direction] = Line(collision_line.start, intersection_point)
        # Case where no obstacles are found in the bounding box or none of the available obstacle collide with the collision line
        if (not collision_list) or (not intersection_point):
            logging.debug(f'collision list is empty or no intersection point found in bounding box')

            collision = collision_line.intersect(collision_box)
            logging.debug(f'collision between line {collision_line} from head {self.snake.head} and bounding box {collision_box} is : {collision}')
            self.collision_lines[direction] = Line(*collision)

    def _update_collision_lines(self):
        opposite_direction = get_opposite_direction(self.snake.direction)
        # multithreading for this task do not improve performances !
        for direction in Direction:
            # Check all particular cases before calling the update fonction
            if direction == opposite_direction:
                self.collision_lines[direction] = None
            elif (direction == Direction.UP) and (self.snake.head.top <= 0):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midtop)
            elif (direction == Direction.UP_RIGHT) and (self.snake.head.top <= 0 or self.snake.head.right >= self.width):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.topright)
            elif (direction == Direction.RIGHT) and (self.snake.head.right >= self.width):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midright)
            elif (direction == Direction.DOWN_RIGHT) and (self.snake.head.bottom >= self.height or self.snake.head.right >= self.width):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.bottomright)
            elif (direction == Direction.DOWN) and (self.snake.head.bottom >= self.height):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midbottom)
            elif (direction == Direction.DOWN_LEFT) and (self.snake.head.bottom >= self.height or self.snake.head.left <= 0):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.bottomleft)
            elif (direction == Direction.LEFT) and (self.snake.head.left <= 0):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midleft)
            elif (direction == Direction.UP_LEFT) and (self.snake.head.left <= 0 or self.snake.head.top <= 0):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.topleft)
            else:
                self._update_collision_line(direction)


class SnakeGameHuman(SnakeGame):
    def __init__(self, width=640, height=480, speed=10, obstacle_rate=0):
        super().__init__(width, height, speed, obstacle_rate)
        self._direction = Direction.RIGHT

    def play_step(self):
        # 1. collect user input
        self._collect_user_inputs()
        # 2. move
        self.snake.move(self._direction) # update the head
        # compute collision lines
        self._init_collision_lines()
        self._update_collision_lines()
        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. place new food or just move
        if self.snake.head.colliderect(self.food):
            self.score += 1
            self.snake.grow()
            self._place_food()

        # 5. update ui and clock
        self._update_ui()
        pygame.display.flip()
        self.clock.tick(self._speed)
        # 6. return game over and score
        return game_over, self.score

    def _collect_user_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self._direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self._direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self._direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self._direction = Direction.DOWN


def main():
    format = '%(levelname)s:%(module)s.%(funcName)s : %(message)s'
    logging.basicConfig(level=logging.INFO, format=format)
    game = SnakeGameHuman(1000,1000, speed=20, obstacle_rate=0.1)

    # game loop
    while True:
        game_over, score = game.play_step()

        if game_over == True:
            break

    logging.info(f'GAME OVER\nFinal Score = {score}')

    pygame.quit()

if __name__ == '__main__':
    main()