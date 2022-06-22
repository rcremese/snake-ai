##
# @author Firstname Lastname <firstname.lastname@example.com>
 # @file Description
 # @desc Created on 2022-06-21 5:20:46 pm
 # @copyright APPI SASU
 #
from asyncio.log import logger
import logging
from snake_game import SnakeGame
from utils import Direction
import pygame

SPEED = 20

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
    logging.basicConfig(level=logging.DEBUG, format=format)
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