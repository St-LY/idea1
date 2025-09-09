import torch
import torch.optim as optim
from models import BottomModel
from crypto_utils import CryptoUtils


class Client:
    def __init__(self, client_id, input_dim, hidden_dims=[128, 64], learning_rate=0.01):
        self.client_id = client_id
        self.model = BottomModel(input_dim, hidden_dims)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.crypto = CryptoUtils()
        self.public_key = None

    def set_public_key(self, public_key):
        """设置公钥"""
        self.public_key = public_key
        self.crypto.public_key = public_key

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def compute_intermediate(self, x):
        """计算中间结果"""
        with torch.no_grad():
            return self.forward(x)

    def backward(self, gradient):
        """反向传播"""
        self.optimizer.zero_grad()

        # 假设中间层输出的梯度已计算
        # 这里需要根据实际梯度传播逻辑实现
        # 伪代码: self.model.output.backward(gradient)

        self.optimizer.step()

    def encrypt_intermediate(self, intermediate):
        """加密中间结果"""
        if self.public_key is None:
            raise ValueError("Public key not set. Call set_public_key() first.")

        # 将中间结果转换为numpy数组并加密
        intermediate_np = intermediate.detach().numpy()
        encrypted_intermediate = self.crypto.encrypt_array(intermediate_np.flatten())

        return encrypted_intermediate, intermediate_np.shape

    def train_step(self, x, top_gradient):
        """客户端训练一步（简化版本）"""
        self.optimizer.zero_grad()

        # 前向传播
        output = self.model(x)

        # 反向传播（简化版本，实际需要更复杂的梯度处理）
        output.backward(top_gradient)

        self.optimizer.step()