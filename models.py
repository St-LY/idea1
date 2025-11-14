import torch
import torch.nn as nn
import torch.nn.functional as F


class BottomModel(nn.Module):
    """客户端底部模型 - 根据数据集配置自动构建"""

    def __init__(self, input_channels, dataset_config=None):
        super(BottomModel, self).__init__()

        if dataset_config is None:
            # 默认使用MNIST配置（保持向后兼容）
            dataset_config = {
                'conv1_out': 32,
                'conv2_out': 64,
                'fc_out': 128,
            }

        self.input_channels = input_channels
        self.config = dataset_config

        # 根据配置构建卷积层
        if 'conv3_out' in dataset_config:
            # 三层卷积（如CIFAR-10, SVHN）
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channels, dataset_config['conv1_out'], kernel_size=3, padding=1),
                nn.BatchNorm2d(dataset_config['conv1_out']),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(dataset_config['conv1_out'], dataset_config['conv2_out'], kernel_size=3, padding=1),
                nn.BatchNorm2d(dataset_config['conv2_out']),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(dataset_config['conv2_out'], dataset_config['conv3_out'], kernel_size=3, padding=1),
                nn.BatchNorm2d(dataset_config['conv3_out']),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d((2, 2)),
            )
            fc_in = dataset_config['conv3_out'] * 4
        else:
            # 两层卷积（如MNIST, Fashion-MNIST）
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channels, dataset_config['conv1_out'], kernel_size=3, padding=1),
                nn.BatchNorm2d(dataset_config['conv1_out']),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Conv2d(dataset_config['conv1_out'], dataset_config['conv2_out'], kernel_size=3, padding=1),
                nn.BatchNorm2d(dataset_config['conv2_out']),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d((2, 2)),
            )
            fc_in = dataset_config['conv2_out'] * 4

        # 展平层
        self.flatten = nn.Flatten()

        # 全连接层
        self.fc = nn.Linear(fc_in, dataset_config['fc_out'])

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class TopModel(nn.Module):
    """服务器端顶部模型 - 根据数据集配置自动构建"""

    def __init__(self, input_dim, output_dim=10, dataset_config=None):
        super(TopModel, self).__init__()

        if dataset_config is None:
            # 默认使用MNIST配置（保持向后兼容）
            dataset_config = {
                'hidden1': 64,
                'hidden2': 32,
            }

        self.network = nn.Sequential(
            nn.Linear(input_dim, dataset_config['hidden1']),
            nn.BatchNorm1d(dataset_config['hidden1']),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(dataset_config['hidden1'], dataset_config['hidden2']),
            nn.BatchNorm1d(dataset_config['hidden2']),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(dataset_config['hidden2'], output_dim),
        )

    def forward(self, x):
        return self.network(x)