import pygame
import random
import numpy as np

class FlappyBirdEnv:
    def __init__(self):
        self.SCREEN_WIDTH = 288
        self.SCREEN_HEIGHT = 512
        self.PIPE_GAP = 100
        self.GRAVITY = 1
        self.JUMP_VELOCITY = -10
        self.PIPE_VELOCITY = -4
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.bg = pygame.image.load('assets/bg.png')
        self.birdup = pygame.image.load('assets/birdup.png')
        self.pipedown = pygame.image.load('assets/pipedown.png')
        self.pipeup = pygame.image.load('assets/pipeup.png')

        self.reset()

    def reset(self):
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = [self.generate_pipe()]
        self.score = 0
        self.done = False
        return self.get_state()

    def generate_pipe(self):
        pipe_height = random.randint(50, self.SCREEN_HEIGHT - self.PIPE_GAP - 50)
        return {'x': self.SCREEN_WIDTH, 'y': pipe_height - self.pipeup.get_height(), 'scored': False}

    def get_state(self):
        pipe = self.pipes[0]
        return np.array([self.bird_y, pipe['x'], pipe['y']])

    def step(self, action):
        if action == 1:
            self.bird_velocity = self.JUMP_VELOCITY

        self.bird_velocity += self.GRAVITY
        self.bird_y += self.bird_velocity

        for pipe in self.pipes:
            pipe['x'] += self.PIPE_VELOCITY

        if self.pipes[0]['x'] < -self.pipeup.get_width():
            self.pipes.pop(0)
            self.pipes.append(self.generate_pipe())

        reward = 0
        if self.pipes[0]['x'] < 50 and not self.pipes[0]['scored']:
            self.score += 1
            self.pipes[0]['scored'] = True
            reward = 1

        if self.bird_y >= self.SCREEN_HEIGHT or self.bird_y < 0:
            self.done = True
            reward = -10

        return self.get_state(), reward, self.done

    def render(self):
        self.screen.blit(self.bg, (0, 0))
        self.screen.blit(self.birdup, (50, self.bird_y))
        for pipe in self.pipes:
            self.screen.blit(self.pipeup, (pipe['x'], pipe['y']))
            self.screen.blit(self.pipedown, (pipe['x'], pipe['y'] + self.pipeup.get_height() + self.PIPE_GAP))
        pygame.display.update()
        self.clock.tick(30)
