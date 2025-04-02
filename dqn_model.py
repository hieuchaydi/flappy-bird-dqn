import torch
import torch.nn as nn

class DQN(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=2):  # Đổi từ 4 thành 3
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)  # input_dim = 3
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
