import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.fernet import Fernet
import pickle
import base64


class CryptoUtils:
    def __init__(self, key_size=2048):
        self.key_size = key_size
        self.public_key = None
        self.private_key = None

    def generate_keys(self):
        """生成RSA密钥对"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        return self.public_key, self.private_key

    def encrypt_with_public_key(self, data, public_key=None):
        """使用公钥加密数据（使用混合加密）"""
        if public_key is None:
            if self.public_key is None:
                raise ValueError("Public key not available.")
            public_key = self.public_key

        # 序列化数据
        if isinstance(data, (dict, list, tuple)):
            serialized_data = pickle.dumps(data)
        else:
            serialized_data = data

        # 生成对称密钥
        symmetric_key = Fernet.generate_key()
        fernet = Fernet(symmetric_key)

        # 使用对称密钥加密数据
        encrypted_data = fernet.encrypt(serialized_data)

        # 使用RSA公钥加密对称密钥
        encrypted_key = public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # 将加密的密钥和数据组合在一起
        result = {
            'key': encrypted_key,
            'data': encrypted_data
        }

        return pickle.dumps(result)

    def decrypt_with_private_key(self, encrypted_data):
        """使用私钥解密数据"""
        if self.private_key is None:
            raise ValueError("Private key not available.")

        try:
            # 反序列化结果
            result = pickle.loads(encrypted_data)
            encrypted_key = result['key']
            encrypted_data = result['data']

            # 使用RSA私钥解密对称密钥
            symmetric_key = self.private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            # 使用对称密钥解密数据
            fernet = Fernet(symmetric_key)
            decrypted_data = fernet.decrypt(encrypted_data)

            # 尝试反序列化
            try:
                return pickle.loads(decrypted_data)
            except:
                return decrypted_data

        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")

    def encrypt_array(self, array):
        """加密numpy数组（兼容性方法）"""
        # 转换为列表进行加密
        array_list = array.tolist()
        return self.encrypt_with_public_key(array_list)

    def decrypt_array(self, encrypted_data):
        """解密numpy数组（兼容性方法）"""
        decrypted_data = self.decrypt_with_private_key(encrypted_data)
        return np.array(decrypted_data)
