import torch
import pygame
from flappy_bird import FlappyBirdEnv  # Sửa lại thành FlappyBirdEnv thay vì FlappyBirdGame

from dqn_model import DQN

# Khởi tạo game và mô hình
env = FlappyBirdEnv()
model = DQN(4, 2)
model.load_state_dict(torch.load("flappy_bird_dqn.pth"))
model.eval()  # Đưa mô hình vào chế độ đánh giá (không huấn luyện)

# Chạy game với AI
state = env.reset()
state = torch.tensor(state, dtype=torch.float32)

running = True
while running:
    env.render()

    with torch.no_grad():
        action = torch.argmax(model(state)).item()

    next_state, _, done = env.step(action)
    state = torch.tensor(next_state, dtype=torch.float32)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if done:
        state = torch.tensor(env.reset(), dtype=torch.float32)

pygame.quit()
