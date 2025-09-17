import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VFLConfig


class BottomModel(nn.Module):
    """客户端底部模型 - 使用卷积层处理图像片段"""

    def __init__(self, input_channels):
        super(BottomModel, self).__init__()

        # 根据输入通道数构建网络
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # 自适应池化到固定大小


        )

        # 展平层
        self.flatten = nn.Flatten()

        # 全连接层输出64维特征
        self.fc = nn.Linear(32 * 4, 64)  # 128 * 2 * 2 = 512

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class TopModel(nn.Module):
    """服务器端顶部模型"""

    def __init__(self, input_dim, output_dim=10):
        super(TopModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.network(x)
