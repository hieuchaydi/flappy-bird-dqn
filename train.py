# train.py
import torch
import numpy as np
import random
from collections import deque
from flappy_bird import FlappyBirdEnv
from dqn_model import DQN

# Khởi tạo môi trường và mô hình
env = FlappyBirdEnv()
model = DQN(input_dim=4, output_dim=2)  # Cập nhật input_dim thành 4
target_model = DQN(input_dim=4, output_dim=2)  # Cập nhật input_dim thành 4
target_model.load_state_dict(model.state_dict())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Tham số huấn luyện
epsilon = 1.0
gamma = 0.99
num_episodes = 1000
max_steps = 1000
replay_buffer = deque(maxlen=10000)
batch_size = 64

def sample_from_replay():
    return random.sample(replay_buffer, batch_size) if len(replay_buffer) >= batch_size else []

# Huấn luyện
for episode in range(num_episodes):
    state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    total_reward = 0
    
    for step in range(max_steps):
        action = random.randint(0, 1) if random.random() < epsilon else torch.argmax(model(state)).item()
        
        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        
        total_reward += reward
        replay_buffer.append((state, action, reward, next_state, done))
        
        state = next_state
        if done:
            break

    epsilon = max(0.1, epsilon * 0.995)
    
    if len(replay_buffer) >= batch_size:
        minibatch = sample_from_replay()
        state_batch = torch.cat([s[0] for s in minibatch])
        action_batch = torch.tensor([s[1] for s in minibatch], dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor([s[2] for s in minibatch], dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.cat([s[3] for s in minibatch])
        done_batch = torch.tensor([s[4] for s in minibatch], dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            target_batch = reward_batch + gamma * torch.max(target_model(next_state_batch), dim=1, keepdim=True)[0] * (1 - done_batch)
        
        output_batch = model(state_batch).gather(1, action_batch)
        
        loss = loss_fn(output_batch, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())
    
    print(f"Episode {episode + 1}: Score {total_reward}, Epsilon {epsilon:.3f}")
    
    if (episode + 1) % 500 == 0:
        torch.save(model.state_dict(), "flappy_bird_dqn.pth")
        print(f"Model saved at episode {episode + 1}")

torch.save(model.state_dict(), "flappy_bird_dqn.pth")
print("Training completed, final model saved!")
