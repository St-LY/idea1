import numpy as np
import phe as paillier



class CryptoUtils:
    def __init__(self, key_size=1024):
        self.key_size = key_size
        self.public_key = None
        self.private_key = None

    def generate_keys(self):
        """生成Paillier同态加密密钥对"""
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=self.key_size)
        return self.public_key, self.private_key

    def encrypt_array(self, array):
        """加密numpy数组"""
        if self.public_key is None:
            raise ValueError("Public key not generated. Call generate_keys() first.")

        # 将浮点数转换为整数（同态加密通常处理整数）
        array_int = (array * 10000).astype(np.int64)
        encrypted_array = [self.public_key.encrypt(x) for x in array_int]
        return encrypted_array

    def decrypt_array(self, encrypted_array):
        """解密加密数组"""
        if self.private_key is None:
            raise ValueError("Private key not generated. Call generate_keys() first.")

        decrypted_array = [self.private_key.decrypt(x) / 10000.0 for x in encrypted_array]
        return np.array(decrypted_array)

    def partial_decrypt(self, encrypted_array, weight):
        """部分解密（用于客户端）"""
        partial_results = [x * weight for x in encrypted_array]
        return partial_results