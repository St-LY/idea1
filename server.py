import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.fernet import Fernet
import pickle
import hashlib
import time

from models import TopModel
from crypto_utils import CryptoUtils

from config import VFLConfig


class Server:
    def __init__(self, input_dim, output_dim=VFLConfig.output_dim, learning_rate=VFLConfig.learning_rate):
        self.model = TopModel(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # 生成RSA密钥对
        self.crypto = CryptoUtils()
        self.public_key, self.private_key = self.crypto.generate_keys()

        # 存储客户端公钥和签名验证相关信息
        self.client_public_keys = {}
        self.signature_verification_enabled = True
        self.invalid_signature_count = 0
        self.valid_signature_count = 0

    def register_client_public_key(self, client_id, public_key):
        """注册客户端的RSA公钥"""
        self.client_public_keys[client_id] = public_key

    def get_client_public_keys(self):
        """获取所有客户端的公钥列表"""
        return list(self.client_public_keys.values())

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

    def decrypt_message(self, encrypted_data):
        """解密客户端发送的加密消息"""
        try:
            return self.crypto.decrypt_with_private_key(encrypted_data)
        except Exception as e:
            print(f"Message decryption failed: {e}")
            return None

    def verify_ring_signature(self, signature_data):
        """
        验证环签名（先解密后验证）
        """
        if not self.signature_verification_enabled:
            return True, "Signature verification disabled"

        try:
            # 提取签名和消息
            signature = signature_data['signature']
            message = signature_data['message']
            public_keys_pem = signature_data.get('public_keys_pem', [])
            key_image = signature_data.get('key_image')
            timestamp = signature_data.get('timestamp', 0)

            # 获取签名时使用的数据（如果存在）
            signature_bytes = signature_data.get('signature_bytes')

            # 检查必要的字段是否存在
            if not all([signature, message, public_keys_pem]):
                return False, "Missing required signature fields"

            # 检查时间戳是否合理（允许5分钟内的消息）
            current_time = time.time()
            if abs(current_time - timestamp) > 300:
                return False, "Timestamp too old or in the future"

            # 验证公钥列表不为空
            if not public_keys_pem:
                return False, "No public keys available for verification"

            # 检查密钥镜像是否已使用（防止重放攻击）
            if self._is_key_image_used(key_image):
                stored_timestamp = self._get_key_image_timestamp(key_image)
                if abs(timestamp - stored_timestamp) < 1:
                    pass
                else:
                    return False, "Key image already used (possible replay attack)"

            # 如果没有提供签名时使用的数据，则重新构建
            if signature_bytes is None:
                # 准备签名数据
                signature_payload = {
                    'message': message,
                    'key_image': key_image,
                    'public_keys_pem': public_keys_pem,
                    'timestamp': timestamp
                }

                signature_bytes = pickle.dumps(signature_payload)

            # 尝试使用环中的每个公钥验证签名
            for pub_key_pem in public_keys_pem:
                try:
                    pub_key = serialization.load_pem_public_key(pub_key_pem, backend=default_backend())
                    pub_key.verify(
                        signature,
                        signature_bytes,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                    # 验证成功，标记密钥镜像为已使用
                    self._mark_key_image_used(key_image, timestamp)
                    self.valid_signature_count += 1
                    return True, "Signature verified successfully"
                except Exception:
                    continue

            # 所有公钥验证都失败
            self.invalid_signature_count += 1
            return False, "All public key verifications failed"

        except Exception as e:
            self.invalid_signature_count += 1
            return False, f"Signature verification error: {str(e)}"

    def _is_key_image_used(self, key_image):
        """检查密钥镜像是否已使用"""
        if not hasattr(self, '_used_key_images'):
            self._used_key_images = {}
        return key_image in self._used_key_images

    def _mark_key_image_used(self, key_image, timestamp):
        """标记密钥镜像为已使用"""
        if not hasattr(self, '_used_key_images'):
            self._used_key_images = {}
        self._used_key_images[key_image] = timestamp

    def _get_key_image_timestamp(self, key_image):
        """获取密钥镜像对应的时间戳"""
        if not hasattr(self, '_used_key_images'):
            self._used_key_images = {}
            return 0
        return self._used_key_images.get(key_image, 0)

    def process_encrypted_messages(self, signed_messages):
        """处理签名消息：累加而不是拼接"""
        accumulated_intermediate = None
        valid_count = 0

        for signed_message in signed_messages:
            # 1. 验证环签名
            is_valid, reason = self.verify_ring_signature(signed_message)
            if not is_valid:
                print(f"Invalid signature detected: {reason}")
                continue

            # 2. 解密消息
            try:
                encrypted_intermediate = signed_message['message']['intermediate']
                decrypted_data = self.decrypt_message(encrypted_intermediate)

                if decrypted_data is not None:
                    # 转换为tensor
                    if isinstance(decrypted_data, list):
                        intermediate_tensor = torch.tensor(np.array(decrypted_data)).float()
                    else:
                        intermediate_tensor = torch.tensor(decrypted_data).float()

                    # 累加而不是拼接
                    if accumulated_intermediate is None:
                        accumulated_intermediate = intermediate_tensor
                    else:
                        accumulated_intermediate += intermediate_tensor

                    valid_count += 1
                else:
                    print("Failed to decrypt message")
            except Exception as e:
                print(f"Decryption failed: {e}")

        return accumulated_intermediate, valid_count

    def train_step(self, encrypted_messages, labels):
        """
        训练一步：使用累加而不是拼接
        """
        # 处理加密消息：解密并累加
        accumulated_intermediate, valid_count = self.process_encrypted_messages(encrypted_messages)

        # 如果没有有效的中间结果，返回空值
        if accumulated_intermediate is None:
            print("No valid intermediates received. Skipping training step.")
            return None, None, 0

        # 前向传播 - 直接使用累加结果
        predictions = self.model(accumulated_intermediate)

        # 计算损失
        loss = self.compute_loss(predictions, labels)

        # 反向传播
        self.backward(loss)

        return loss.item(), predictions, valid_count

    def predict(self, encrypted_messages):
        """预测：先解密消息，再验证签名，最后处理中间结果"""
        valid_intermediates, _ = self.process_encrypted_messages(encrypted_messages)

        # 如果没有有效的中间结果，返回空值
        if not valid_intermediates:
            print("No valid intermediates received. Skipping prediction.")
            return None

        with torch.no_grad():
            combined_input = torch.cat(valid_intermediates, dim=1)
            predictions = self.model(combined_input)
            return predictions

    def compute_accuracy(self, predictions, labels):
        """计算准确率"""
        _, predicted = torch.max(predictions.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct / total

    def get_signature_stats(self):
        """获取签名验证统计信息"""
        total = self.valid_signature_count + self.invalid_signature_count
        return {
            'valid_signatures': self.valid_signature_count,
            'invalid_signatures': self.invalid_signature_count,
            'success_rate': self.valid_signature_count / max(1, total)
        }

    def enable_signature_verification(self, enabled=True):
        """启用或禁用签名验证"""
        self.signature_verification_enabled = enabled
