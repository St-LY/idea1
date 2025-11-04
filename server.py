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
from numba import jit, prange
import warnings

from models import TopModel
from crypto_utils import CryptoUtils
from config import VFLConfig

warnings.filterwarnings('ignore')


# Numba优化的距离计算函数
@jit(nopython=True, fastmath=True, cache=True)
def compute_manhattan_distance_numba(vec1, vec2):
    """使用Numba加速的曼哈顿距离计算"""
    return np.sum(np.abs(vec1 - vec2))


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_distances_to_group_numba(vector, group_vectors):
    """并行计算向量与组内所有向量的距离"""
    n = len(group_vectors)
    distances = np.empty(n, dtype=np.float64)
    for i in prange(n):
        distances[i] = compute_manhattan_distance_numba(vector, group_vectors[i])
    return distances


@jit(nopython=True, fastmath=True, cache=True)
def compute_avg_distance_numba(distances):
    """计算平均距离"""
    if len(distances) == 0:
        return 0.0
    return np.mean(distances)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_group_avg_distances_numba(vectors_array):
    """并行计算组内所有向量的平均距离"""
    n = len(vectors_array)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    avg_distances = np.empty(n, dtype=np.float64)
    for i in prange(n):
        distances = np.empty(n - 1, dtype=np.float64)
        idx = 0
        for j in range(n):
            if i != j:
                distances[idx] = compute_manhattan_distance_numba(vectors_array[i], vectors_array[j])
                idx += 1
        avg_distances[i] = np.mean(distances)

    return avg_distances


@jit(nopython=True, fastmath=True, cache=True)
def find_max_distance_index_numba(distances):
    """找到最大距离的索引"""
    max_val = distances[0]
    max_idx = 0
    for i in range(1, len(distances)):
        if distances[i] > max_val:
            max_val = distances[i]
            max_idx = i
    return max_idx, max_val


class Server:
    def __init__(self, input_dim, output_dim=VFLConfig.output_dim, learning_rate=VFLConfig.learning_rate):
        # 获取设备配置
        self.device = VFLConfig.device

        # 初始化模型并移到GPU
        self.model = TopModel(input_dim, output_dim).to(self.device)
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

        # 防投毒机制相关
        self.output_dim = output_dim
        self.max_vectors_per_label = 100
        self.label_vectors = {i: [] for i in range(output_dim)}
        self.label_vectors_numpy = {i: None for i in range(output_dim)}
        self.label_avg_distances = {i: np.array([], dtype=np.float64) for i in range(output_dim)}

        self.is_pretraining = True
        self.reference_vectors = None
        self.reference_avg_distances = None

        self.poisoning_stats = {
            'rejected_count': 0,
            'accepted_count': 0,
            'replaced_count': 0
        }
        self.BorderPara = 1.2
        self.border_update_count = 0

        print(f"Server initialized on device: {self.device}")

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
        """验证环签名(先解密后验证)"""
        if not self.signature_verification_enabled:
            return True, "Signature verification disabled"

        try:
            signature = signature_data['signature']
            message = signature_data['message']
            public_keys_pem = signature_data.get('public_keys_pem', [])
            key_image = signature_data.get('key_image')
            timestamp = signature_data.get('timestamp', 0)
            signature_bytes = signature_data.get('signature_bytes')

            if not all([signature, message, public_keys_pem]):
                return False, "Missing required signature fields"

            current_time = time.time()
            if abs(current_time - timestamp) > 300:
                return False, "Timestamp too old or in the future"

            if not public_keys_pem:
                return False, "No public keys available for verification"

            if self._is_key_image_used(key_image):
                stored_timestamp = self._get_key_image_timestamp(key_image)
                if abs(timestamp - stored_timestamp) < 0.1:
                    pass
                else:
                    print(f"[Warning] Key image reused with time gap: {abs(timestamp - stored_timestamp):.2f}s")
                    pass

            if signature_bytes is None:
                signature_payload = {
                    'message': message,
                    'key_image': key_image,
                    'public_keys_pem': public_keys_pem,
                    'timestamp': timestamp
                }
                signature_bytes = pickle.dumps(signature_payload)

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
                    self._mark_key_image_used(key_image, timestamp)
                    self.valid_signature_count += 1
                    return True, "Signature verified successfully"
                except Exception:
                    continue

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
        """处理签名消息:累加而不是拼接"""
        accumulated_intermediate = None
        valid_count = 0

        for signed_message in signed_messages:
            is_valid, reason = self.verify_ring_signature(signed_message)
            if not is_valid:
                print(f"Invalid signature detected: {reason}")
                continue

            try:
                encrypted_intermediate = signed_message['message']['intermediate']
                decrypted_data = self.decrypt_message(encrypted_intermediate)

                if decrypted_data is not None:
                    if isinstance(decrypted_data, list):
                        intermediate_tensor = torch.tensor(np.array(decrypted_data, dtype=np.float32)).float().to(
                            self.device)
                    else:
                        intermediate_tensor = torch.tensor(decrypted_data).float().to(self.device)

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

    def _convert_vectors_to_numpy(self, label):
        """将标签的向量列表转换为numpy数组以便Numba处理"""
        vectors = self.label_vectors[label]
        if len(vectors) == 0:
            return np.empty((0, 0), dtype=np.float64)

        numpy_vectors = np.array([v.cpu().numpy() for v in vectors], dtype=np.float64)
        self.label_vectors_numpy[label] = numpy_vectors
        return numpy_vectors

    def compute_manhattan_distance(self, vec1, vec2):
        """计算两个向量的曼哈顿距离 - 使用Numba优化"""
        vec1_np = vec1.cpu().numpy().astype(np.float64) if isinstance(vec1, torch.Tensor) else vec1.astype(np.float64)
        vec2_np = vec2.cpu().numpy().astype(np.float64) if isinstance(vec2, torch.Tensor) else vec2.astype(np.float64)
        return compute_manhattan_distance_numba(vec1_np, vec2_np)

    def compute_avg_distance_to_group(self, vector, group_vectors):
        """计算一个向量与组内所有向量的平均曼哈顿距离 - 使用Numba优化"""
        if len(group_vectors) == 0:
            return 0.0

        vector_np = vector.cpu().numpy().astype(np.float64) if isinstance(vector, torch.Tensor) else vector.astype(
            np.float64)
        group_np = np.array([v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in group_vectors],
                            dtype=np.float64)

        distances = compute_distances_to_group_numba(vector_np, group_np)
        return compute_avg_distance_numba(distances)

    def update_label_vectors_pretraining(self, vector, label):
        """预训练阶段:更新标签对应的向量数组"""
        label = int(label)
        vector_detached = vector.detach().clone()

        current_vectors = self.label_vectors[label]

        if len(current_vectors) < self.max_vectors_per_label:
            current_vectors.append(vector_detached)
            self._update_label_avg_distances(label)
            return True

        vector_np = vector_detached.cpu().numpy().astype(np.float64)
        group_np = self._convert_vectors_to_numpy(label)

        distances = compute_distances_to_group_numba(vector_np, group_np)
        new_vec_avg_dist = compute_avg_distance_numba(distances)

        group_avg_distances = compute_group_avg_distances_numba(group_np)
        max_dist_idx, max_avg_dist = find_max_distance_index_numba(group_avg_distances)

        threshold = max_avg_dist * self.BorderPara
        if self.is_pretraining:
            self.border_update_count += 1
            increment = 0.001 / (1 + 0.08 * (self.border_update_count - 1))
            increment = max(increment, 0.0001)
            self.BorderPara += increment

        if new_vec_avg_dist > threshold:
            self.poisoning_stats['rejected_count'] += 1
            print(f"[Defense] Rejected vector for label {label}: "
                  f"distance {new_vec_avg_dist:.4f} > threshold {threshold:.4f}")
            return False
        else:
            self.poisoning_stats['accepted_count'] += 1

            if new_vec_avg_dist < max_avg_dist:
                current_vectors[max_dist_idx] = vector_detached
                self.poisoning_stats['replaced_count'] += 1
                self._update_label_avg_distances(label)

                updated_avg_distances = self.label_avg_distances[label]
                group_mean_dist = np.mean(updated_avg_distances) if len(updated_avg_distances) > 0 else 0

                print(f"[Defense] Replaced vector for label {label}: "
                      f"new distance {new_vec_avg_dist:.4f} < max distance {max_avg_dist:.4f}, "
                      f"updated group avg: {group_mean_dist:.4f}")

            return True

    def _update_label_avg_distances(self, label):
        """更新指定标签下所有向量的平均距离 - 使用Numba优化"""
        current_vectors = self.label_vectors[label]

        if len(current_vectors) <= 1:
            self.label_avg_distances[label] = np.zeros(len(current_vectors), dtype=np.float64)
            return

        vectors_np = self._convert_vectors_to_numpy(label)
        avg_distances = compute_group_avg_distances_numba(vectors_np)
        self.label_avg_distances[label] = avg_distances

    def check_vector_formal_training(self, vector, label):
        """正式训练阶段:检查向量是否应该被接受 - 使用Numba优化"""
        if self.reference_vectors is None or self.reference_avg_distances is None:
            print("[Warning] Reference vectors not computed. Accepting by default.")
            return True

        label = int(label)
        vector_detached = vector.detach().clone()

        ref_vector = self.reference_vectors[label]
        distance = self.compute_manhattan_distance(vector_detached, ref_vector)

        ref_avg_dist = self.reference_avg_distances[label]
        threshold = ref_avg_dist * self.BorderPara

        if distance > threshold:
            self.poisoning_stats['rejected_count'] += 1
            print(f"[Defense] Rejected vector for label {label}: "
                  f"distance {distance:.4f} > threshold {threshold:.4f}")
            return False
        else:
            self.poisoning_stats['accepted_count'] += 1
            return True

    def finalize_pretraining(self):
        """结束预训练,计算参考向量和平均距离 - 使用Numba优化"""
        print("\n[Defense] Finalizing pretraining phase...")

        self.reference_vectors = {}
        self.reference_avg_distances = {}

        for label in range(self.output_dim):
            vectors = self.label_vectors[label]

            if len(vectors) == 0:
                print(f"[Warning] No vectors collected for label {label}")
                self.reference_vectors[label] = torch.zeros(128).to(self.device)
                self.reference_avg_distances[label] = 0.0
                continue

            stacked_vectors = torch.stack(vectors)
            ref_vector = torch.mean(stacked_vectors, dim=0)
            self.reference_vectors[label] = ref_vector

            vectors_np = self._convert_vectors_to_numpy(label)
            ref_vector_np = ref_vector.cpu().numpy().astype(np.float64)

            distances = compute_distances_to_group_numba(ref_vector_np, vectors_np)
            avg_dist = compute_avg_distance_numba(distances)

            self.reference_avg_distances[label] = avg_dist

            print(f"[Defense] Label {label}: {len(vectors)} vectors, "
                  f"avg distance to reference: {avg_dist:.4f}")

        self.is_pretraining = False
        print("[Defense] Pretraining completed. Starting formal training with defense.\n")
        print(f"[Defense] Pretraining completed. BorderPara value: {self.BorderPara:.4f}")

    def train_step(self, encrypted_messages, labels):
        """训练一步:使用累加而不是拼接,并加入防投毒机制"""
        # 将labels移到GPU
        labels = labels.to(self.device)

        accumulated_intermediate, valid_count = self.process_encrypted_messages(encrypted_messages)

        if accumulated_intermediate is None:
            print("No valid intermediates received. Skipping training step.")
            return None, None, 0

        predictions = self.model(accumulated_intermediate)

        _, predicted_labels = torch.max(predictions.data, 1)

        should_train_flags = []
        for i in range(len(labels)):
            sample_vector = accumulated_intermediate[i]
            true_label = labels[i].item()

            if self.is_pretraining:
                should_train = self.update_label_vectors_pretraining(sample_vector, true_label)
            else:
                predicted_label = predicted_labels[i].item()
                should_train = self.check_vector_formal_training(sample_vector, predicted_label)

            should_train_flags.append(should_train)

        if not any(should_train_flags):
            print("[Defense] All samples rejected. Skipping training step.")
            return None, None, 0

        accepted_indices = [i for i, flag in enumerate(should_train_flags) if flag]
        if len(accepted_indices) < len(labels):
            print(f"[Defense] {len(accepted_indices)}/{len(labels)} samples accepted for training")
            filtered_predictions = predictions[accepted_indices]
            filtered_labels = labels[accepted_indices]
        else:
            filtered_predictions = predictions
            filtered_labels = labels

        loss = self.compute_loss(filtered_predictions, filtered_labels)
        self.backward(loss)

        return loss.item(), predictions, valid_count

    def predict(self, encrypted_messages):
        """预测:先解密消息,再验证签名,最后处理中间结果"""
        valid_intermediates, _ = self.process_encrypted_messages(encrypted_messages)

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

    def get_poisoning_stats(self):
        """获取防投毒统计信息"""
        total = self.poisoning_stats['accepted_count'] + self.poisoning_stats['rejected_count']
        return {
            'accepted': self.poisoning_stats['accepted_count'],
            'rejected': self.poisoning_stats['rejected_count'],
            'replaced': self.poisoning_stats['replaced_count'],
            'acceptance_rate': self.poisoning_stats['accepted_count'] / max(1, total)
        }

    def enable_signature_verification(self, enabled=True):
        """启用或禁用签名验证"""
        self.signature_verification_enabled = enabled