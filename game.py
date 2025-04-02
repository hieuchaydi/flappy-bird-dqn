import torch
import random
import pygame
import time

# Giả sử bạn đã định nghĩa DQN và đã tải mô hình (flappy_bird_dqn.pth)
from dqn_model import DQN

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_GAP = 100
PIPE_VELOCITY = -4
GRAVITY = 1
JUMP_VELOCITY = -10
input_dim = 3  # Với mô hình của bạn, số lượng đầu vào là 3 (ví dụ: y của chim, y của ống và khoảng cách giữa chim và ống)

# Load assets
bg = pygame.image.load('assets/bg.png')
birdup = pygame.image.load('assets/birdup.png')
birddown = pygame.image.load('assets/birddown.png')
font = pygame.font.Font('assets/font.ttf', 24)
ground = pygame.image.load('assets/ground.png')
pipedown = pygame.image.load('assets/pipedown.png')
pipeup = pygame.image.load('assets/pipeup.png')

# Load model
model = DQN(input_dim, 2)  # Giả sử mô hình của bạn có 2 hành động: "nhảy" và "không nhảy"
model.load_state_dict(torch.load("flappy_bird_dqn.pth"))
model.eval()

# Initialize sound
pygame.mixer.init()
try:
    sound = pygame.mixer.Sound('assets/source.ogg')
except pygame.error as e:
    print(f"Error loading sound: {e}")

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Các biến trong game
bird_y = SCREEN_HEIGHT // 2
bird_velocity = 0
pipes = []
score = 0
game_active = False
mute_sound = False

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Functions (Không thay đổi các hàm này)
def draw_background():
    screen.blit(bg, (0, 0))

def draw_ground():
    screen.blit(ground, (0, SCREEN_HEIGHT - ground.get_height()))

def draw_bird():
    if bird_velocity < 0:
        screen.blit(birdup, (50, bird_y))
    else:
        screen.blit(birddown, (50, bird_y))

def draw_pipes():
    for pipe in pipes:
        screen.blit(pipeup, (pipe['x'], pipe['y']))
        screen.blit(pipedown, (pipe['x'], pipe['y'] + pipeup.get_height() + PIPE_GAP))

def generate_pipe():
    pipe_height = random.randint(50, SCREEN_HEIGHT - ground.get_height() - PIPE_GAP - 50)
    return {'x': SCREEN_WIDTH, 'y': pipe_height - pipeup.get_height(), 'scored': False}

def check_collision():
    bird_rect = pygame.Rect(50, bird_y, birdup.get_width(), birdup.get_height())
    for pipe in pipes:
        upper_pipe_rect = pygame.Rect(pipe['x'], pipe['y'], pipeup.get_width(), pipeup.get_height())
        lower_pipe_rect = pygame.Rect(pipe['x'], pipe['y'] + pipeup.get_height() + PIPE_GAP, pipeup.get_width(), pipedown.get_height())
        if bird_rect.colliderect(upper_pipe_rect) or bird_rect.colliderect(lower_pipe_rect):
            return True
    if bird_y + birdup.get_height() >= SCREEN_HEIGHT - ground.get_height():
        return True
    return False

def draw_score():
    score_text = font.render(f"Score: {score}", True, white)
    screen.blit(score_text, (10, 10))

def gameLoop():
    global bird_y, bird_velocity, pipes, score, game_active, mute_sound
    game_over = False
    game_close = False

    bird_y = SCREEN_HEIGHT // 2
    bird_velocity = 0
    pipes = []
    score = 0
    game_active = True

    while not game_over:
        while game_close:
            draw_background()
            draw_score()
            pygame.display.update()
            time.sleep(2)
            gameLoop()  # Restart the game

        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Nhấn phím "r" để chơi lại
                    gameLoop()
                if event.key == pygame.K_q:  # Nhấn phím "q" để thoát game
                    game_over = True

        # Lấy thông tin hiện tại của game (ví dụ: độ cao của chim và ống)
        state = [bird_y, pipes[0]['y'] if len(pipes) > 0 else 0, pipes[0]['x'] if len(pipes) > 0 else SCREEN_WIDTH]

        # Dự đoán hành động từ mô hình
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = model(state_tensor).max(1)[1].item()  # Chọn hành động có giá trị cao nhất (0: không nhảy, 1: nhảy)

        # Thực hiện hành động: nếu action == 1, chim nhảy
        if action == 1:
            bird_velocity = JUMP_VELOCITY

        # Cập nhật trạng thái chim
        bird_velocity += GRAVITY
        bird_y += bird_velocity

        if bird_y >= SCREEN_HEIGHT or bird_y < 0:
            game_close = True

        # Tạo ống mới
        if len(pipes) == 0 or pipes[-1]['x'] < SCREEN_WIDTH - 150:
            pipes.append(generate_pipe())

        for pipe in pipes:
            pipe['x'] += PIPE_VELOCITY

        pipes = [pipe for pipe in pipes if pipe['x'] > -pipeup.get_width()]

        # Kiểm tra va chạm
        if check_collision():
            game_active = False
            game_close = True

        # Vẽ lại game
        draw_background()
        draw_pipes()
        draw_bird()
        draw_ground()
        draw_score()

        pygame.display.update()
        pygame.time.Clock().tick(30)

    pygame.quit()
    quit()

# Chạy menu game
def main_menu():
    global game_active
    game_active = False
    while True:
        screen.fill(black)
        title_text = font.render("Flappy Bird", True, white)
        screen.blit(title_text, (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4))

        start_button = pygame.Rect(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2, 150, 50)
        pygame.draw.rect(screen, white, start_button)
        start_text = font.render("Start Game", True, black)
        screen.blit(start_text, (SCREEN_WIDTH // 4 + 25, SCREEN_HEIGHT // 2 + 10))

        quit_button = pygame.Rect(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2 + 70, 150, 50)
        pygame.draw.rect(screen, white, quit_button)
        quit_text = font.render("Quit", True, black)
        screen.blit(quit_text, (SCREEN_WIDTH // 4 + 60, SCREEN_HEIGHT // 2 + 80))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos):
                    gameLoop()
                if quit_button.collidepoint(event.pos):
                    pygame.quit()
                    quit()

        pygame.display.update()
        pygame.time.Clock().tick(30)

# Chạy menu game
main_menu()
