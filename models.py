#file:C:\Users\24576\Desktop\thesis01\VFL框架\idea1\models.py
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
                'layers': [
                    {
                        'type': 'conv2d',
                        'in_channels': input_channels,
                        'out_channels': 32,
                        'kernel_size': 3,
                        'padding': 1
                    },
                    {
                        'type': 'batchnorm2d',
                        'num_features': 32
                    },
                    {
                        'type': 'relu'
                    },
                    {
                        'type': 'avgpool2d',
                        'kernel_size': 2,
                        'stride': 2
                    },
                    {
                        'type': 'conv2d',
                        'in_channels': 32,
                        'out_channels': 64,
                        'kernel_size': 3,
                        'padding': 1
                    },
                    {
                        'type': 'batchnorm2d',
                        'num_features': 64
                    },
                    {
                        'type': 'relu'
                    },
                    {
                        'type': 'adaptiveavgpool2d',
                        'output_size': (2, 2)
                    }
                ],
                'fc_layers': [
                    {
                        'type': 'linear',
                        'in_features': 256,  # 64*2*2
                        'out_features': 128
                    },
                    {
                        'type': 'batchnorm1d',
                        'num_features': 128
                    },
                    {
                        'type': 'relu'
                    }
                ],
                'fc_out': 128,
                'dropout': 0.0
            }

        self.input_channels = input_channels
        
        # 构建卷积层
        conv_layers = []
        for layer_config in dataset_config['layers']:
            layer_type = layer_config['type']
            if layer_type == 'conv2d':
                conv_layers.append(nn.Conv2d(
                    layer_config['in_channels'],
                    layer_config['out_channels'],
                    layer_config['kernel_size'],
                    padding=layer_config.get('padding', 0),
                    stride=layer_config.get('stride', 1)
                ))
            elif layer_type == 'batchnorm2d':
                conv_layers.append(nn.BatchNorm2d(layer_config['num_features']))
            elif layer_type == 'relu':
                conv_layers.append(nn.ReLU())
            elif layer_type == 'leaky_relu':
                conv_layers.append(nn.LeakyReLU(layer_config.get('negative_slope', 0.01)))
            elif layer_type == 'tanh':
                conv_layers.append(nn.Tanh())
            elif layer_type == 'sigmoid':
                conv_layers.append(nn.Sigmoid())
            elif layer_type == 'maxpool2d':
                conv_layers.append(nn.MaxPool2d(
                    layer_config['kernel_size'],
                    stride=layer_config.get('stride', None),
                    padding=layer_config.get('padding', 0)
                ))
            elif layer_type == 'avgpool2d':
                conv_layers.append(nn.AvgPool2d(
                    layer_config['kernel_size'],
                    stride=layer_config.get('stride', None),
                    padding=layer_config.get('padding', 0)
                ))
            elif layer_type == 'adaptiveavgpool2d':
                conv_layers.append(nn.AdaptiveAvgPool2d(layer_config['output_size']))
            elif layer_type == 'adaptivemaxpool2d':
                conv_layers.append(nn.AdaptiveMaxPool2d(layer_config['output_size']))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 构建全连接层
        fc_layers = []
        for layer_config in dataset_config['fc_layers']:
            layer_type = layer_config['type']
            if layer_type == 'linear':
                fc_layers.append(nn.Linear(
                    layer_config['in_features'],
                    layer_config['out_features']
                ))
            elif layer_type == 'batchnorm1d':
                fc_layers.append(nn.BatchNorm1d(layer_config['num_features']))
            elif layer_type == 'relu':
                fc_layers.append(nn.ReLU())
            elif layer_type == 'leaky_relu':
                fc_layers.append(nn.LeakyReLU(layer_config.get('negative_slope', 0.01)))
            elif layer_type == 'tanh':
                fc_layers.append(nn.Tanh())
            elif layer_type == 'sigmoid':
                fc_layers.append(nn.Sigmoid())
            elif layer_type == 'dropout':
                fc_layers.append(nn.Dropout(layer_config.get('p', 0.5)))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # 展平层
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


class TopModel(nn.Module):
    """服务器端顶部模型 - 根据数据集配置自动构建"""

    def __init__(self, input_dim, output_dim=10, dataset_config=None):
        super(TopModel, self).__init__()

        if dataset_config is None:
            # 默认使用MNIST配置（保持向后兼容）
            dataset_config = {
                'layers': [
                    {
                        'type': 'linear',
                        'in_features': input_dim,
                        'out_features': 64
                    },
                    {
                        'type': 'batchnorm1d',
                        'num_features': 64
                    },
                    {
                        'type': 'relu'
                    },
                    {
                        'type': 'dropout',
                        'p': 0.3
                    },
                    {
                        'type': 'linear',
                        'in_features': 64,
                        'out_features': 32
                    },
                    {
                        'type': 'batchnorm1d',
                        'num_features': 32
                    },
                    {
                        'type': 'relu'
                    },
                    {
                        'type': 'dropout',
                        'p': 0.2
                    },
                    {
                        'type': 'linear',
                        'in_features': 32,
                        'out_features': output_dim
                    }
                ]
            }

        layers = []
        for layer_config in dataset_config['layers']:
            layer_type = layer_config['type']
            if layer_type == 'linear':
                layers.append(nn.Linear(
                    layer_config['in_features'],
                    layer_config['out_features']
                ))
            elif layer_type == 'batchnorm1d':
                layers.append(nn.BatchNorm1d(layer_config['num_features']))
            elif layer_type == 'relu':
                layers.append(nn.ReLU())
            elif layer_type == 'leaky_relu':
                layers.append(nn.LeakyReLU(layer_config.get('negative_slope', 0.01)))
            elif layer_type == 'tanh':
                layers.append(nn.Tanh())
            elif layer_type == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif layer_type == 'dropout':
                layers.append(nn.Dropout(layer_config.get('p', 0.5)))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
