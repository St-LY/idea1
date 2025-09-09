import torch
import torch.optim as optim
import torch.nn as nn
from models import TopModel
from crypto_utils import CryptoUtils


class Server:
    def __init__(self, input_dim, output_dim=10, learning_rate=0.01):
        self.model = TopModel(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.crypto = CryptoUtils()
        self.public_key, _ = self.crypto.generate_keys()

    def get_public_key(self):
        return self.public_key

    def compute_loss(self, predictions, labels):
        """计算损失"""
        return self.criterion(predictions, labels)

    def backward(self, loss):
        """反向传播"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self, intermediate_outputs, labels):
        """训练一步"""
        # 合并来自各客户端的中间输出
        combined_input = torch.cat(intermediate_outputs, dim=1)

        # 前向传播
        predictions = self.model(combined_input)

        # 计算损失
        loss = self.compute_loss(predictions, labels)

        # 反向传播
        self.backward(loss)

        return loss.item(), predictions

    def predict(self, intermediate_outputs):
        """预测"""
        with torch.no_grad():
            combined_input = torch.cat(intermediate_outputs, dim=1)
            predictions = self.model(combined_input)
            return predictions

    def compute_accuracy(self, predictions, labels):
        """计算准确率"""
        _, predicted = torch.max(predictions.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct / total