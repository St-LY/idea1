import hashlib
import time
import secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import threading


class TokenManager:
    """环形令牌管理器 - 防止DOS攻击"""

    def __init__(self, client_id, num_clients, token_timeout=30.0):
        self.client_id = client_id
        self.num_clients = num_clients
        self.token_timeout = token_timeout

        # 令牌状态
        self.has_token = (client_id == 0)  # 客户端0初始持有令牌
        self.token_secret = None
        self.token_counter = 0
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
            self.token_secret = self._generate_token_secret(0)
            self.token_counter = 0
            print(f"[TokenManager] Client {self.client_id} generated initial token (counter=0)")

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

    def generate_token_proof(self, message_data):
        """生成令牌证明"""
        with self.lock:
            if not self.has_token:
                raise ValueError(f"Client {self.client_id} cannot generate proof without token")

            if self.token_secret is None:
                raise ValueError("Token secret not initialized")

            # 创建证明数据
            timestamp = time.time()
            message_hash = hashlib.sha256(str(message_data).encode()).digest()

            # 生成HMAC作为证明
            proof_data = message_hash + str(self.token_counter).encode() + str(timestamp).encode()
            proof = hashlib.pbkdf2_hmac('sha256', proof_data, self.token_secret, 1000)

            return {
                'proof': proof,
                'counter': self.token_counter,
                'timestamp': timestamp,
                'message_hash': message_hash
            }

    def pass_token(self):
        """传递令牌给下一个客户端"""
        with self.lock:
            if not self.has_token:
                raise ValueError(f"Client {self.client_id} cannot pass token without holding it")

            # 增加计数器
            self.token_counter += 1

            # 生成新令牌秘密
            new_token_secret = self._generate_token_secret(self.token_counter)

            # 创建令牌传递数据
            token_data = {
                'counter': self.token_counter,
                'secret': new_token_secret,
                'timestamp': time.time(),
                'from_client': self.client_id,
                'to_client': (self.client_id + 1) % self.num_clients
            }

            # 释放令牌
            self.has_token = False
            self.token_secret = None

            print(f"[Token] Client {self.client_id} -> Client {token_data['to_client']} (counter={self.token_counter})")

            return token_data

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

            # 接收令牌
            self.has_token = True
            self.token_counter = token_data['counter']
            self.token_secret = token_data['secret']
            self.last_token_time = time.time()

            # 记录历史
            self.token_history.append({
                'counter': self.token_counter,
                'timestamp': self.last_token_time,
                'from_client': token_data.get('from_client')
            })

            if len(self.token_history) > self.max_history:
                self.token_history.pop(0)

            # 重置警报
            self.alarm_raised = False

            print(f"[Token] Client {self.client_id} received token (counter={self.token_counter})")
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
            return {
                'client_id': self.client_id,
                'has_token': self.has_token,
                'counter': self.token_counter,
                'alarm_raised': self.alarm_raised,
                'history_count': len(self.token_history)
            }


class ServerTokenVerifier:
    """服务器端令牌验证器"""

    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.chain_master_secret = None

        # 验证统计
        self.valid_proofs = 0
        self.invalid_proofs = 0
        self.total_verifications = 0

        # 已使用的令牌计数器
        self.used_counters = set()
        self.max_counter = -1

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
            timestamp = proof_data['timestamp']
            message_hash = proof_data['message_hash']

            # 检查时间戳
            if abs(time.time() - timestamp) > 60:
                self.invalid_proofs += 1
                return False, "Token proof expired"

            # 检查计数器（允许一定乱序）
            if counter < self.max_counter - 20:
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
            proof_data_check = computed_hash + str(counter).encode() + str(timestamp).encode()
            expected_proof = hashlib.pbkdf2_hmac('sha256', proof_data_check, expected_secret, 1000)

            if secrets.compare_digest(proof, expected_proof):
                # 记录计数器
                self.used_counters.add(counter)
                self.max_counter = max(self.max_counter, counter)

                # 清理旧计数器
                if len(self.used_counters) > 1000:
                    min_valid = self.max_counter - 100
                    self.used_counters = {c for c in self.used_counters if c > min_valid}

                self.valid_proofs += 1
                return True, "Token proof valid"
            else:
                self.invalid_proofs += 1
                return False, "Invalid token proof"

        except Exception as e:
            self.invalid_proofs += 1
            return False, f"Token verification error: {str(e)}"

    def get_statistics(self):
        """获取验证统计"""
        return {
            'total_verifications': self.total_verifications,
            'valid_proofs': self.valid_proofs,
            'invalid_proofs': self.invalid_proofs,
            'success_rate': self.valid_proofs / max(1, self.total_verifications)
        }