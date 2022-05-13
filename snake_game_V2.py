##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-13 3:15:13 pm
 # @copyright https://mit-license.org/
 #
import queue
import pygame
import random
from snake import Snake
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
SPEED = 10

class SnakeGame:    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.snake = Snake(self.w/2, self.h/2, pixel_size=PIXEL_SIZE)
        # self.head = Point(self.w/2, self.h/2)
        # self.snake = [self.head, 
        #               Point(self.head.x-BLOCK_SIZE, self.head.y),
        #               Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        # Define the central coordinates of the food to be placed
        x = random.randint(PIXEL_SIZE / 2, self.w - PIXEL_SIZE / 2)
        y = random.randint(PIXEL_SIZE / 2, self.h - PIXEL_SIZE / 2 )
        #self.food = Point(x, y)
        self.food = pygame.Rect(x - PIXEL_SIZE / 2, y - PIXEL_SIZE / 2, PIXEL_SIZE, PIXEL_SIZE)
        # Check to not place food inside the snake
            
        if self.food.collidelist(self.snake.body):
            self._place_food()
        
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
        # self.snake.insert(0, self.head)
        
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
    
    def _is_collision(self):
        return self.snake.collide_with_borders(self.w, self.h) or self.snake.collide_with_itself()
        
    def _update_ui(self):
        self.display.fill(BLACK)
        move = (PIXEL_SIZE - BODY_PIXEL_SIZE) / 2
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pt)
            # draw the body of the snake, in order to count the parts
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + move, pt.y + move, BODY_PIXEL_SIZE, BODY_PIXEL_SIZE))
            
        pygame.draw.rect(self.display, GREEN, self.food)
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()        

if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()