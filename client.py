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
import random
import threading
from queue import Queue, Empty

from models import BottomModel
from crypto_utils import CryptoUtils
from config import VFLConfig
from token_manager import TokenManager


class Client:
    def __init__(self, client_id, input_channels, learning_rate=None, dataset_config=None):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if learning_rate is None:
            learning_rate = 0.001

        # ✅ 正确提取 bottom_model 配置
        if dataset_config is not None:
            bottom_config = dataset_config.get('bottom_model', None)
        else:
            bottom_config = None

        # 传递 bottom_model 配置而不是整个 dataset_config
        self.model = BottomModel(input_channels, bottom_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0001)

        self.crypto = CryptoUtils()
        self.server_public_key = None

        # 生成RSA密钥对用于环签名
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
        self.rsa_public_key_pem = self.rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # 令牌管理器
        self.token_manager = None

        # 并行处理相关
        self.processing_queue = Queue()
        self.result_queue = Queue()
        self.processing_thread = None
        self.stop_flag = False

        print(f"Client {client_id} initialized on device: {self.device}")

    def initialize_token_manager(self, num_clients, chain_secret):
        """初始化令牌管理器"""
        self.token_manager = TokenManager(self.client_id, num_clients)
        self.token_manager.initialize_chain_secret(chain_secret)
        print(f"[Client {self.client_id}] Token manager initialized")

    def start_parallel_processing(self):
        """启动并行处理线程"""
        self.stop_flag = False
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        print(f"[Client {self.client_id}] Parallel processing thread started")

    def stop_parallel_processing(self):
        """停止并行处理线程"""
        self.stop_flag = True
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        print(f"[Client {self.client_id}] Parallel processing thread stopped")

    def _processing_worker(self):
        """并行处理工作线程"""
        while not self.stop_flag:
            try:
                # 从队列获取任务
                try:
                    task = self.processing_queue.get(timeout=0.1)
                except Empty:
                    continue

                if task is None:
                    break

                task_type = task['type']

                if task_type == 'compute_intermediate':
                    # 等待令牌
                    data = task['data']
                    batch_idx = task['batch_idx']
                    epoch = task['epoch']
                    phase = task['phase']

                    # 等待令牌（最多等待30秒）
                    wait_start = time.time()
                    while not self.token_manager.can_send():
                        time.sleep(0.01)
                        if time.time() - wait_start > 30.0:
                            print(f"[Client {self.client_id}] Timeout waiting for token!")
                            self.result_queue.put(None)
                            break
                    else:
                        # 持有令牌，处理任务
                        try:
                            result = self._compute_and_sign(data, batch_idx, epoch, phase)
                            self.result_queue.put(result)
                        except Exception as e:
                            print(f"[Client {self.client_id}] Error in _compute_and_sign: {e}")
                            import traceback
                            traceback.print_exc()
                            self.result_queue.put(None)

            except Exception as e:
                if not self.stop_flag:
                    print(f"[Client {self.client_id}] Processing worker error: {e}")
                    import traceback
                    traceback.print_exc()

    def _compute_and_sign(self, data, batch_idx, epoch, phase):
        """计算中间结果并签名（内部方法）"""
        try:
            # 计算中间结果
            intermediate = self.compute_intermediate(data)

            # 生成令牌证明
            message_data = {
                'batch_idx': batch_idx,
                'epoch': epoch,
                'phase': phase,
                'client_id': self.client_id,
                'timestamp': time.time()
            }

            token_proof = self.token_manager.generate_token_proof(message_data)

            # 签名（包含令牌证明）
            signature = self.ring_signature({
                'intermediate': intermediate,
                'batch_idx': batch_idx,
                'epoch': epoch,
                'phase': phase,
                'client_id': self.client_id,
                'timestamp': message_data['timestamp'],
                'token_proof': token_proof
            })

            # 传递令牌给下一个客户端
            token_data = self.token_manager.pass_token()
            self.pass_token_to_next(token_data)

            # 随机转发
            forwarded_signature = self.send_to_random_client_direct(signature)

            return forwarded_signature

        except Exception as e:
            print(f"[Client {self.client_id}] Error in _compute_and_sign: {e}")
            import traceback
            traceback.print_exc()
            return None

    def submit_task(self, data, batch_idx, epoch, phase):
        """提交计算任务到队列"""
        task = {
            'type': 'compute_intermediate',
            'data': data,
            'batch_idx': batch_idx,
            'epoch': epoch,
            'phase': phase
        }
        self.processing_queue.put(task)

    def get_result(self, timeout=None):
        """获取计算结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None

    def set_server_public_key(self, public_key):
        """设置服务器公钥"""
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
        """加密中间结果"""
        if self.server_public_key is None:
            raise ValueError("Server public key not set.")

        intermediate_np = intermediate.cpu().detach().numpy()
        encrypted_intermediate = self.crypto.encrypt_with_public_key(
            intermediate_np.tolist(),
            self.server_public_key
        )
        return encrypted_intermediate, intermediate_np.shape

    def train_step(self, x, top_gradient):
        """客户端训练一步"""
        x = x.to(self.device)
        top_gradient = top_gradient.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(x)
        output.backward(top_gradient)
        self.optimizer.step()

    def can_process(self):
        """检查是否可以处理（是否持有令牌）"""
        return self.token_manager.can_send()

    def ring_signature(self, message):
        """使用RSA构建环签名"""
        if not hasattr(self, 'ring_public_keys_pem'):
            raise ValueError("No ring public keys set.")

        if isinstance(message, dict):
            if 'timestamp' not in message:
                message['timestamp'] = time.time()
        else:
            message = {'data': message, 'timestamp': time.time()}

        # 生成唯一的密钥镜像
        key_image = self._generate_key_image(message)

        # 加密中间结果
        if 'intermediate' in message and isinstance(message['intermediate'], torch.Tensor):
            encrypted_intermediate, shape = self.encrypt_intermediate(message['intermediate'])
            message['intermediate'] = encrypted_intermediate

        # 生成签名数据
        signature_payload = {
            'message': message,
            'key_image': key_image,
            'public_keys_pem': self.ring_public_keys_pem,
            'timestamp': message['timestamp']
        }

        signature_bytes = pickle.dumps(signature_payload)

        # 签名
        signature = self.rsa_private_key.sign(
            signature_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

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
        """生成密钥镜像"""
        private_key_bytes = self.rsa_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        context_str = f"{message.get('epoch', 0)}_{message.get('batch_idx', 0)}_{message.get('phase', 'unknown')}_{message.get('client_id', self.client_id)}"
        context_bytes = context_str.encode('utf-8')

        combined = private_key_bytes + context_bytes
        return hashlib.sha256(combined).digest()

    def send_to_random_client_direct(self, signed_data):
        """直接发送签名数据"""
        if not hasattr(self, 'all_clients') or not self.all_clients:
            return signed_data

        target_client = random.choice(self.all_clients)

        if target_client.client_id == self.client_id:
            return signed_data

        return target_client.receive_and_forward_signed(signed_data)

    def receive_and_forward_signed(self, signed_data):
        """接收并转发已签名的数据"""
        return signed_data

    def pass_token_to_next(self, token_data):
        """传递令牌给下一个客户端"""
        if not hasattr(self, 'all_clients') or not self.all_clients:
            return

        next_client_id = (self.client_id + 1) % len(self.all_clients)
        next_client = self.all_clients[next_client_id]
        next_client.receive_token(token_data)

    def receive_token(self, token_data):
        """接收令牌"""
        success = self.token_manager.receive_token(token_data)
        if not success:
            print(f"[Client {self.client_id}] Failed to receive token")

    def check_token_timeout(self):
        """检查令牌超时"""
        return self.token_manager.check_timeout()

    def get_token_status(self):
        """获取令牌状态"""
        return self.token_manager.get_status()