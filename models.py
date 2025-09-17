# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import VFLConfig


class BottomModel(nn.Module):
    """客户端底部模型 - 使用卷积层处理图像片段"""

    def __init__(self, input_channels, learning_rate=VFLConfig.learning_rate):
        super(BottomModel, self).__init__()

        # 根据输入通道数构建网络
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # 自适应池化到固定大小

        )

        # 展平层
        self.flatten = nn.Flatten()

        # 全连接层输出64维特征
        self.fc = nn.Linear(64 * 4, 128)  # 128 * 2 * 2 = 512

        # 添加Adam优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class TopModel(nn.Module):
    """服务器端顶部模型"""

    def __init__(self, input_dim, output_dim=10, learning_rate=VFLConfig.learning_rate):
        super(TopModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

        # 添加Adam优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.network(x)
