import torch
import torch.optim as optim
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import pickle
import hashlib
import time

from models import BottomModel
from crypto_utils import CryptoUtils

from config import VFLConfig


class Client:
    def __init__(self, client_id, input_dim, hidden_dims=VFLConfig.hidden_dims, learning_rate=VFLConfig.learning_rate):
        self.client_id = client_id
        self.model = BottomModel(input_dim, hidden_dims)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.crypto = CryptoUtils()
        self.public_key = None

        # 生成RSA密钥对用于环签名
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.rsa_public_key = self.rsa_private_key.public_key()

        # 将公钥转换为可序列化的PEM格式
        self.rsa_public_key_pem = self.rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def set_public_key(self, public_key):
        """设置Paillier公钥"""
        self.public_key = public_key
        self.crypto.public_key = public_key

    def set_ring_public_keys(self, public_keys):
        """设置环中所有客户端的公钥"""
        # 将所有公钥转换为PEM格式以便序列化
        self.ring_public_keys_pem = []
        for key in public_keys:
            if isinstance(key, bytes):
                # 如果已经是PEM格式
                self.ring_public_keys_pem.append(key)
            else:
                # 如果是RSA对象，转换为PEM格式
                pem = key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                self.ring_public_keys_pem.append(pem)

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

    def ring_signature(self, message):
        """
        使用RSA构建环签名

        参数:
            message: 要签名的消息

        返回:
            signature: 环签名
        """
        if not hasattr(self, 'ring_public_keys_pem'):
            raise ValueError("No ring public keys set. Call set_ring_public_keys() first.")

        # 确保消息中包含时间戳
        if isinstance(message, dict):
            if 'timestamp' not in message:
                message['timestamp'] = time.time()
        else:
            message = {'data': message, 'timestamp': time.time()}

        # 生成密钥镜像 (key image)
        key_image = self._generate_key_image()

        # 生成签名数据（只包含可序列化的数据）
        signature_payload = {
            'message': message,
            'key_image': key_image,
            'public_keys_pem': self.ring_public_keys_pem,
            'timestamp': message['timestamp']
        }

        # 使用私钥对签名数据进行签名
        signature_bytes = self.rsa_private_key.sign(
            pickle.dumps(signature_payload),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # 返回完整的签名信息
        return {
            'message': message,
            'signature': signature_bytes,
            'key_image': key_image,
            'public_keys_pem': self.ring_public_keys_pem,
            'timestamp': message['timestamp']
        }

    def _generate_key_image(self):
        """生成密钥镜像 (简化实现)"""
        private_key_bytes = self.rsa_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        return hashlib.sha256(private_key_bytes).digest()
