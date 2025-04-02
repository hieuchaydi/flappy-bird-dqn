import torch

from dqn_model import DQN
from flappy_bird import FlappyBirdEnv


def load_model(model_path):
    # Tạo mô hình DQN với input_size và output_size phù hợp
    model = DQN(input_size=3, output_size=2)  # Điều chỉnh input_size và output_size theo cấu hình của bạn
    model.load_state_dict(torch.load(model_path))  # Tải mô hình đã huấn luyện
    model.eval()  # Chế độ đánh giá (không huấn luyện)
    return model


def play():
    # Tải mô hình đã huấn luyện
    model = load_model('flappy_bird_dqn.pth')
    
    # Tạo môi trường game FlappyBirdEnv
    env = FlappyBirdEnv()

    # Khởi tạo AI với mô hình đã huấn luyện
    ai = FlappyBirdEnv(model, env)

    # Chơi game
    ai.play_game()  # Dùng mô hình để điều khiển game và hiển thị kết quả


if __name__ == "__main__":
    play()
