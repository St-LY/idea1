import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
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
        self.crypto = CryptoUtils()
        self.public_key, _ = self.crypto.generate_keys()

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

    def verify_ring_signature(self, signature_data):
        """
        验证环签名

        参数:
            signature_data: 包含签名和消息的字典

        返回:
            is_valid: 签名是否有效
            reason: 无效的原因（如果无效）
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

            # 检查必要的字段是否存在
            if not all([signature, message, public_keys_pem]):
                return False, "Missing required signature fields"

            # 检查时间戳是否合理（允许5分钟内的消息）
            current_time = time.time()
            if abs(current_time - timestamp) > 300:  # 5分钟
                return False, "Timestamp too old or in the future"

            # 验证公钥列表不为空
            if not public_keys_pem:
                return False, "No public keys available for verification"

            # 检查密钥镜像是否已使用（防止重放攻击）
            if self._is_key_image_used(key_image):
                stored_timestamp = self._get_key_image_timestamp(key_image)
                # 如果是同一时间戳的消息，可能是重试，允许通过
                if abs(timestamp - stored_timestamp) < 1:
                    pass  # 允许相同时间戳的消息
                else:
                    return False, "Key image already used (possible replay attack)"

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
                    # 将PEM格式的公钥转换回RSA对象
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
                except Exception as e:
                    continue  # 尝试下一个公钥

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

    def train_step(self, signed_intermediates, labels):
        """
        训练一步，验证签名并处理中间结果

        参数:
            signed_intermediates: 包含签名和中间结果的列表
            labels: 标签

        返回:
            loss: 损失值
            predictions: 预测结果
            valid_count: 有效签名的数量
        """
        valid_intermediates = []
        valid_count = 0

        # 验证每个中间结果的签名
        for signed_data in signed_intermediates:
            is_valid, reason = self.verify_ring_signature(signed_data)

            if is_valid:
                valid_intermediates.append(signed_data['message']['intermediate'])
                valid_count += 1
            else:
                print(f"Invalid signature detected: {reason}")
                # 记录安全事件或采取其他措施

        # 如果没有有效的中间结果，返回空值
        if not valid_intermediates:
            print("No valid intermediates received. Skipping training step.")
            return None, None, 0

        # 合并来自各客户端的中间输出
        combined_input = torch.cat(valid_intermediates, dim=1)

        # 前向传播
        predictions = self.model(combined_input)

        # 计算损失
        loss = self.compute_loss(predictions, labels)

        # 反向传播
        self.backward(loss)

        return loss.item(), predictions, valid_count

    def predict(self, signed_intermediates):
        """预测，验证签名并处理中间结果"""
        valid_intermediates = []

        # 验证每个中间结果的签名
        for signed_data in signed_intermediates:
            is_valid, reason = self.verify_ring_signature(signed_data)

            if is_valid:
                valid_intermediates.append(signed_data['message']['intermediate'])
            else:
                print(f"Invalid signature detected during prediction: {reason}")

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
        return {
            'valid_signatures': self.valid_signature_count,
            'invalid_signatures': self.invalid_signature_count,
            'success_rate': self.valid_signature_count / max(1,
                                                             self.valid_signature_count + self.invalid_signature_count)
        }

    def enable_signature_verification(self, enabled=True):
        """启用或禁用签名验证"""
        self.signature_verification_enabled = enabled
