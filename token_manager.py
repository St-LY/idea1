import hashlib
import time
import secrets
import random
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import threading


class TokenManager:
    """无中心环形令牌管理器 - 防止DOS攻击"""

    def __init__(self, client_id, num_clients, token_timeout=30.0):
        self.client_id = client_id
        self.num_clients = num_clients
        self.token_timeout = token_timeout

        # 令牌状态
        self.has_token = (client_id == 0)  # 客户端0初始持有令牌
        self.token_data = None  # 完整的令牌数据结构
        self.last_token_time = time.time() if self.has_token else None

        # 全局令牌链密钥
        self.chain_master_secret = None

        # 令牌接收历史
        self.token_history = []
        self.max_history = 100

        # 警报状态
        self.alarm_raised = False
        self.last_alarm_time = None

        # 线程锁
        self.lock = threading.Lock()

        print(f"[TokenManager] Client {client_id} initialized. Initial holder: {'Yes' if self.has_token else 'No'}")

    def initialize_chain_secret(self, master_secret):
        """初始化令牌链主密钥"""
        self.chain_master_secret = master_secret

        if self.has_token:
            # 初始化令牌数据结构
            all_clients = list(range(self.num_clients))
            random.shuffle(all_clients)  # 随机化初始访问顺序

            self.token_data = {
                'counter': 0,  # 批次号
                'round_id': 0,  # 轮次ID
                'visited_clients': [],  # 本轮已访问的客户端
                'remaining_clients': all_clients.copy(),  # 本轮待访问的客户端
                'total_clients': self.num_clients,
                'secret': self._generate_token_secret(0),
                'timestamp': time.time()
            }
            print(f"[TokenManager] Client {self.client_id} initialized token (counter=0, round=0)")
            print(f"[TokenManager] Initial visit order: {all_clients}")

    def _generate_token_secret(self, counter):
        """生成令牌秘密"""
        if self.chain_master_secret is None:
            raise ValueError("Chain master secret not initialized")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=f"token_{counter}".encode(),
            iterations=10000,
            backend=default_backend()
        )
        return kdf.derive(self.chain_master_secret)

    def can_send(self):
        """检查是否可以发送（持有令牌）"""
        with self.lock:
            return self.has_token

    def get_batch_info(self):
        """获取当前批次信息（用于消息签名）"""
        with self.lock:
            if not self.has_token or self.token_data is None:
                return None

            return {
                'counter': self.token_data['counter'],
                'round_id': self.token_data['round_id']
            }

    def generate_token_proof(self, message_data):
        """生成令牌证明"""
        with self.lock:
            if not self.has_token or self.token_data is None:
                raise ValueError(f"Client {self.client_id} cannot generate proof without token")

            # 创建证明数据（包含批次信息）
            timestamp = time.time()
            message_hash = hashlib.sha256(str(message_data).encode()).digest()

            # 生成HMAC作为证明
            proof_data = (message_hash +
                          str(self.token_data['counter']).encode() +
                          str(self.token_data['round_id']).encode() +
                          str(timestamp).encode())
            proof = hashlib.pbkdf2_hmac('sha256', proof_data, self.token_data['secret'], 1000)

            return {
                'proof': proof,
                'counter': self.token_data['counter'],
                'round_id': self.token_data['round_id'],
                'timestamp': timestamp,
                'message_hash': message_hash
            }

    def pass_token(self):
        """传递令牌给下一个客户端"""
        with self.lock:
            if not self.has_token or self.token_data is None:
                raise ValueError(f"Client {self.client_id} cannot pass token without holding it")

            # 将当前客户端标记为已访问
            if self.client_id not in self.token_data['visited_clients']:
                self.token_data['visited_clients'].append(self.client_id)

            # 从待访问列表中移除当前客户端
            if self.client_id in self.token_data['remaining_clients']:
                self.token_data['remaining_clients'].remove(self.client_id)

            # 检查是否完成了一轮
            if len(self.token_data['remaining_clients']) == 0:
                # 一轮完成，批次号增加
                self.token_data['counter'] += 1
                self.token_data['round_id'] += 1

                # 重置访问列表，生成新的随机顺序
                all_clients = list(range(self.num_clients))
                random.shuffle(all_clients)

                self.token_data['visited_clients'] = []
                self.token_data['remaining_clients'] = all_clients.copy()

                # 更新令牌密钥
                self.token_data['secret'] = self._generate_token_secret(self.token_data['counter'])

                print(f"[Token] Client {self.client_id}: Round {self.token_data['round_id'] - 1} completed!")
                print(f"[Token] Starting round {self.token_data['round_id']}, counter={self.token_data['counter']}")
                print(f"[Token] New visit order: {all_clients}")

            # 选择下一个客户端（从remaining_clients中选第一个）
            if len(self.token_data['remaining_clients']) > 0:
                next_client_id = self.token_data['remaining_clients'][0]
            else:
                # 不应该到这里，但作为安全措施
                next_client_id = (self.client_id + 1) % self.num_clients

            # 更新时间戳
            self.token_data['timestamp'] = time.time()

            # 创建传递的令牌数据（深拷贝）
            token_to_pass = {
                'counter': self.token_data['counter'],
                'round_id': self.token_data['round_id'],
                'visited_clients': self.token_data['visited_clients'].copy(),
                'remaining_clients': self.token_data['remaining_clients'].copy(),
                'total_clients': self.token_data['total_clients'],
                'secret': self.token_data['secret'],
                'timestamp': self.token_data['timestamp'],
                'from_client': self.client_id,
                'to_client': next_client_id
            }

            # 释放令牌
            self.has_token = False
            self.token_data = None

            print(f"[Token] Client {self.client_id} -> Client {next_client_id} "
                  f"(counter={token_to_pass['counter']}, round={token_to_pass['round_id']}, "
                  f"remaining={len(token_to_pass['remaining_clients'])})")

            return token_to_pass

    def receive_token(self, token_data):
        """接收令牌"""
        with self.lock:
            # 验证令牌数据
            expected_client = token_data.get('to_client')
            if expected_client != self.client_id:
                print(f"[TokenManager] Token not for this client (for {expected_client}, this is {self.client_id})")
                return False

            if self.has_token:
                print(f"[TokenManager] Warning: Client {self.client_id} already has token, skipping")
                return False

            # 验证客户端在待访问列表中
            if self.client_id not in token_data['remaining_clients']:
                print(f"[TokenManager] Warning: Client {self.client_id} not in remaining clients list")
                return False

            # 接收令牌
            self.has_token = True
            self.token_data = {
                'counter': token_data['counter'],
                'round_id': token_data['round_id'],
                'visited_clients': token_data['visited_clients'].copy(),
                'remaining_clients': token_data['remaining_clients'].copy(),
                'total_clients': token_data['total_clients'],
                'secret': token_data['secret'],
                'timestamp': time.time()
            }
            self.last_token_time = time.time()

            # 记录历史
            self.token_history.append({
                'counter': self.token_data['counter'],
                'round_id': self.token_data['round_id'],
                'timestamp': self.last_token_time,
                'from_client': token_data.get('from_client')
            })

            if len(self.token_history) > self.max_history:
                self.token_history.pop(0)

            # 重置警报
            self.alarm_raised = False

            print(f"[Token] Client {self.client_id} received token "
                  f"(counter={self.token_data['counter']}, round={self.token_data['round_id']}, "
                  f"remaining={len(self.token_data['remaining_clients'])})")
            return True

    def check_timeout(self):
        """检查是否长时间未收到令牌"""
        with self.lock:
            if self.has_token:
                return False, "Has token"

            if not self.token_history:
                return False, "No history"

            last_receive_time = self.token_history[-1]['timestamp']
            time_since_last = time.time() - last_receive_time

            if time_since_last > self.token_timeout:
                if not self.alarm_raised:
                    self.alarm_raised = True
                    self.last_alarm_time = time.time()
                    print(f"[ALARM] Client {self.client_id} timeout: {time_since_last:.2f}s!")
                return True, f"Timeout: {time_since_last:.2f}s"

            return False, "Normal"

    def get_status(self):
        """获取令牌状态"""
        with self.lock:
            status = {
                'client_id': self.client_id,
                'has_token': self.has_token,
                'alarm_raised': self.alarm_raised,
                'history_count': len(self.token_history)
            }

            if self.token_data:
                status.update({
                    'counter': self.token_data['counter'],
                    'round_id': self.token_data['round_id'],
                    'visited': len(self.token_data['visited_clients']),
                    'remaining': len(self.token_data['remaining_clients'])
                })

            return status


class ServerTokenVerifier:
    """服务器端令牌验证器"""

    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.chain_master_secret = None

        # 验证统计
        self.valid_proofs = 0
        self.invalid_proofs = 0
        self.total_verifications = 0

        # 批次验证：记录每个批次收到的客户端ID
        self.batch_records = {}  # {counter: {client_ids: set(), round_id: int}}
        self.max_counter = -1

        # 用于清理旧批次记录
        self.max_batch_history = 1000

        print(f"[ServerTokenVerifier] Initialized for {num_clients} clients")

    def initialize_chain_secret(self, master_secret):
        """初始化令牌链主密钥"""
        self.chain_master_secret = master_secret
        print("[ServerTokenVerifier] Chain secret initialized")

    def _generate_token_secret(self, counter):
        """生成令牌秘密"""
        if self.chain_master_secret is None:
            raise ValueError("Chain master secret not initialized")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=f"token_{counter}".encode(),
            iterations=10000,
            backend=default_backend()
        )
        return kdf.derive(self.chain_master_secret)

    def verify_token_proof(self, proof_data, message_data):
        """验证令牌证明"""
        self.total_verifications += 1

        try:
            proof = proof_data['proof']
            counter = proof_data['counter']
            round_id = proof_data['round_id']
            timestamp = proof_data['timestamp']
            message_hash = proof_data['message_hash']
            client_id = message_data.get('client_id')

            # 检查时间戳
            if abs(time.time() - timestamp) > 60:
                self.invalid_proofs += 1
                return False, "Token proof expired"

            # 检查计数器（允许一定范围的旧批次，因为可能有并行处理）
            if counter < self.max_counter - 50:
                self.invalid_proofs += 1
                return False, f"Token counter too old (counter={counter}, max={self.max_counter})"

            # 生成令牌秘密
            expected_secret = self._generate_token_secret(counter)

            # 验证消息哈希
            computed_hash = hashlib.sha256(str(message_data).encode()).digest()
            if computed_hash != message_hash:
                self.invalid_proofs += 1
                return False, "Message hash mismatch"

            # 验证证明
            proof_data_check = (computed_hash +
                                str(counter).encode() +
                                str(round_id).encode() +
                                str(timestamp).encode())
            expected_proof = hashlib.pbkdf2_hmac('sha256', proof_data_check, expected_secret, 1000)

            if not secrets.compare_digest(proof, expected_proof):
                self.invalid_proofs += 1
                return False, "Invalid token proof"

            # 记录批次信息
            if counter not in self.batch_records:
                self.batch_records[counter] = {
                    'client_ids': set(),
                    'round_id': round_id
                }

            # 验证round_id一致性
            if self.batch_records[counter]['round_id'] != round_id:
                self.invalid_proofs += 1
                return False, f"Round ID mismatch for counter {counter}"

            # 检查客户端是否重复
            if client_id in self.batch_records[counter]['client_ids']:
                self.invalid_proofs += 1
                return False, f"Duplicate client {client_id} in batch {counter}"

            # 记录客户端
            self.batch_records[counter]['client_ids'].add(client_id)

            # 更新最大计数器
            self.max_counter = max(self.max_counter, counter)

            # 清理旧批次记录
            if len(self.batch_records) > self.max_batch_history:
                min_valid = self.max_counter - 100
                self.batch_records = {
                    k: v for k, v in self.batch_records.items()
                    if k > min_valid
                }

            self.valid_proofs += 1
            return True, "Token proof valid"

        except Exception as e:
            self.invalid_proofs += 1
            return False, f"Token verification error: {str(e)}"

    def verify_batch_consistency(self, messages):
        """验证一批消息是否来自同一批次"""
        if not messages:
            return True, "No messages to verify"

        # 提取所有消息的批次号和轮次号
        counters = set()
        round_ids = set()
        client_ids = set()

        for msg in messages:
            token_proof = msg['message'].get('token_proof')
            if token_proof:
                counters.add(token_proof['counter'])
                round_ids.add(token_proof['round_id'])
                client_ids.add(msg['message'].get('client_id'))

        # 验证所有消息来自同一批次
        if len(counters) > 1:
            return False, f"Messages from different batches: {counters}"

        if len(round_ids) > 1:
            return False, f"Messages from different rounds: {round_ids}"

        # 验证没有重复的客户端
        if len(client_ids) != len(messages):
            return False, "Duplicate clients in batch"

        return True, "Batch consistency verified"

    def get_statistics(self):
        """获取验证统计"""
        return {
            'total_verifications': self.total_verifications,
            'valid_proofs': self.valid_proofs,
            'invalid_proofs': self.invalid_proofs,
            'success_rate': self.valid_proofs / max(1, self.total_verifications),
            'active_batches': len(self.batch_records)
        }