##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-13 3:15:13 pm
 # @copyright https://mit-license.org/
 # TODO : S'assurer qu'il existera toujours un chemin pour atteindre l'objectif

import pygame
import random
from typing import List
from snake import SnakeHuman
from utils import Direction

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 255, 0)
BLACK = (0,0,0)

PIXEL_SIZE = 20
BODY_PIXEL_SIZE = 12
OBSTACLE_SIZE_RANGE = (2, 4)
SPEED = 10
PERCENTAGE = 0.2
class SnakeGame:    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = Direction.RIGHT
        
        # TODO : Initialise snake coordinates randomly
        self.snake = SnakeHuman(self.width/2, self.height/2, pixel_size=PIXEL_SIZE)
        
        self.score = 0

        self._populate_grid_with_obstacles(percent=PERCENTAGE)
        self._place_food()
        
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

    def _populate_grid_with_obstacles(self, percent : float) -> List[pygame.Rect]:
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
        bounding_box_factor = 2 * self.snake._size * PIXEL_SIZE 
    
        if obstacle.colliderect(self.snake.head.inflate(bounding_box_factor, bounding_box_factor)):
            obstacle = self._place_obstacle(size)
        # check inclusion inside other obstacles
        for i in obstacle.collidelistall(self.obstacles):
            # If the new obstacle is completely contained in an existing obstacle, reprocess
            if self.obstacles[i].contains(obstacle):
                obstacle = self._place_obstacle(size)
        return obstacle

    def draw(self, display : pygame.display):
        pass

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. move
        self.snake.move(self.direction) # update the head
        
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
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score
    
    def __is_outside(self, rect : pygame.Rect = None):
        if rect is None:
            rect = self.snake.head
        return rect.x < 0 or rect.x + rect.width > self.width or rect.y < 0 or rect.y + rect.height > self.height 

    def _is_collision(self):
        return self.__is_outside() or self.snake.collide_with_itself() or self.snake.collide_with_obstacles(self.obstacles)
        
    def _update_ui(self):
        self.display.fill(BLACK)
        move = (PIXEL_SIZE - BODY_PIXEL_SIZE) / 2
        for pt in self.snake:
            pygame.draw.rect(self.display, WHITE, pt)
            # draw the body of the snake, in order to count the parts
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + move, pt.y + move, BODY_PIXEL_SIZE, BODY_PIXEL_SIZE))
            
        pygame.draw.rect(self.display, GREEN, self.food)
        # plot obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, RED, obstacle)
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()        

if __name__ == '__main__':
    game = SnakeGame(1000,1000)
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()