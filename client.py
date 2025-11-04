import torch
import torch.optim as optim
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.fernet import Fernet
import pickle
import hashlib
import time
import random

from models import BottomModel
from crypto_utils import CryptoUtils
from config import VFLConfig


class Client:
    def __init__(self, client_id, input_channels, learning_rate=VFLConfig.learning_rate):
        self.client_id = client_id

        # 获取设备配置
        self.device = VFLConfig.device

        # 初始化模型并移到GPU
        self.model = BottomModel(input_channels).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.crypto = CryptoUtils()
        self.server_public_key = None

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

        print(f"Client {client_id} initialized on device: {self.device}")

    def set_server_public_key(self, public_key):
        """设置服务器公钥用于加密消息"""
        self.server_public_key = public_key

    def set_ring_public_keys(self, public_keys):
        """设置环中所有客户端的公钥"""
        self.ring_public_keys_pem = []
        for key in public_keys:
            if isinstance(key, bytes):
                self.ring_public_keys_pem.append(key)
            else:
                pem = key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                self.ring_public_keys_pem.append(pem)

    def set_all_clients(self, all_clients):
        """设置所有客户端的引用"""
        self.all_clients = all_clients

    def forward(self, x):
        """前向传播"""
        # 确保输入数据在正确的设备上
        x = x.to(self.device)
        return self.model(x)

    def compute_intermediate(self, x):
        """计算中间结果"""
        with torch.no_grad():
            return self.forward(x)

    def backward(self, gradient):
        """反向传播"""
        self.optimizer.zero_grad()
        self.optimizer.step()

    def encrypt_intermediate(self, intermediate):
        """加密中间结果(使用服务器公钥)"""
        if self.server_public_key is None:
            raise ValueError("Server public key not set. Call set_server_public_key() first.")

        # 将中间结果转换为numpy数组并加密（需要先移到CPU）
        intermediate_np = intermediate.cpu().detach().numpy()
        encrypted_intermediate = self.crypto.encrypt_with_public_key(
            intermediate_np.tolist(),
            self.server_public_key
        )
        return encrypted_intermediate, intermediate_np.shape

    def train_step(self, x, top_gradient):
        """客户端训练一步"""
        # 确保输入和梯度在正确的设备上
        x = x.to(self.device)
        top_gradient = top_gradient.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(x)
        output.backward(top_gradient)
        self.optimizer.step()

    def ring_signature(self, message):
        """
        使用RSA构建环签名
        修复: 密钥镜像包含消息上下文以避免重复
        """
        if not hasattr(self, 'ring_public_keys_pem'):
            raise ValueError("No ring public keys set. Call set_ring_public_keys() first.")

        # 确保消息中包含时间戳
        if isinstance(message, dict):
            if 'timestamp' not in message:
                message['timestamp'] = time.time()
        else:
            message = {'data': message, 'timestamp': time.time()}

        # 生成唯一的密钥镜像(包含epoch, batch_idx和phase信息)
        key_image = self._generate_key_image(message)

        # 加密中间结果
        if isinstance(message['intermediate'], torch.Tensor):
            encrypted_intermediate, shape = self.encrypt_intermediate(message['intermediate'])
            message['intermediate'] = encrypted_intermediate

        # 生成签名数据(用于签名)
        signature_payload = {
            'message': message,
            'key_image': key_image,
            'public_keys_pem': self.ring_public_keys_pem,
            'timestamp': message['timestamp']
        }

        # 序列化用于签名的数据
        signature_bytes = pickle.dumps(signature_payload)

        # 使用私钥对签名数据进行签名
        signature = self.rsa_private_key.sign(
            signature_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # 返回完整的签名信息
        signed_data = {
            'message': message,
            'signature': signature,
            'key_image': key_image,
            'public_keys_pem': self.ring_public_keys_pem,
            'timestamp': message['timestamp'],
            'signature_bytes': signature_bytes
        }

        return signed_data

    def _generate_key_image(self, message):
        """
        生成密钥镜像
        修复: 包含消息的epoch, batch_idx和phase信息以确保唯一性
        """
        private_key_bytes = self.rsa_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        # 创建包含上下文的唯一标识符
        context_str = f"{message.get('epoch', 0)}_{message.get('batch_idx', 0)}_{message.get('phase', 'unknown')}_{message.get('client_id', self.client_id)}"
        context_bytes = context_str.encode('utf-8')

        # 组合私钥和上下文信息生成唯一的密钥镜像
        combined = private_key_bytes + context_bytes
        return hashlib.sha256(combined).digest()

    def encrypt_message_for_server(self, signed_data):
        """使用服务器公钥加密签名后的消息"""
        if self.server_public_key is None:
            raise ValueError("Server public key not set.")

        encrypted_message = self.crypto.encrypt_with_public_key(
            signed_data,
            self.server_public_key
        )
        return encrypted_message

    def send_to_random_client(self, intermediate, batch_idx, epoch):
        """
        先加密中间结果,再对加密数据进行签名,然后随机转发
        """
        # 直接对中间结果进行签名(签名函数内部会处理加密)
        signature = self.ring_signature({
            'intermediate': intermediate,
            'batch_idx': batch_idx,
            'epoch': epoch,
            'client_id': self.client_id,
            'timestamp': time.time()
        })

        # 随机选择一个客户端进行转发以实现匿名
        forwarded_signature = self.send_to_random_client_direct(signature)
        return forwarded_signature

    def send_to_random_client_direct(self, signed_data):
        """
        直接发送签名数据(不加密,因为已经加密过了)
        """
        if not hasattr(self, 'all_clients') or not self.all_clients:
            # 如果没有设置所有客户端,直接返回签名的数据
            return signed_data

        # 随机选择一个客户端(包括自己)
        target_client = random.choice(self.all_clients)

        # 如果选中的是自己,直接返回签名的数据
        if target_client.client_id == self.client_id:
            return signed_data

        # 否则将签名的数据转发给选中的客户端
        return target_client.receive_and_forward_signed(signed_data)

    def receive_and_forward_signed(self, signed_data):
        """
        接收并转发已经签名的数据(被动方行为)
        """
        # 直接返回已经签名的数据,不进行任何处理
        return signed_data