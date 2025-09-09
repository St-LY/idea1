import torch
import torch.nn as nn


class BottomModel(nn.Module):
    """客户端底部模型"""

    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super(BottomModel, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TopModel(nn.Module):
    """服务器端顶部模型"""

    def __init__(self, input_dim, output_dim=10):
        super(TopModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)