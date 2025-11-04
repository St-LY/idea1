import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numba import jit, prange
import warnings

from config import VFLConfig

warnings.filterwarnings('ignore')


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def split_image_spatial_numba(images, split_points):
    """
    使用Numba并行分割图像

    参数:
        images: 图像数据 (N, C, H, W)
        split_points: 分割点列表

    返回:
        分割后的图像片段列表
    """
    n_samples, channels, height, width = images.shape
    n_parties = len(split_points) - 1

    # 预分配内存
    result = []
    for i in range(n_parties):
        start_col = split_points[i]
        end_col = split_points[i + 1]
        slice_width = end_col - start_col
        party_slice = np.empty((n_samples, channels, height, slice_width), dtype=images.dtype)

        # 并行复制数据
        for j in prange(n_samples):
            for c in range(channels):
                for h in range(height):
                    for w in range(slice_width):
                        party_slice[j, c, h, w] = images[j, c, h, start_col + w]

        result.append(party_slice)

    return result


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def standardize_data_numba(data, mean, std):
    """
    使用Numba并行标准化数据

    参数:
        data: 输入数据 (N, features)
        mean: 均值
        std: 标准差

    返回:
        标准化后的数据
    """
    n_samples, n_features = data.shape
    result = np.empty_like(data)

    for i in prange(n_samples):
        for j in range(n_features):
            # 避免除以零
            if std[j] < 1e-8:
                result[i, j] = 0.0
            else:
                result[i, j] = (data[i, j] - mean[j]) / std[j]

    return result


class MNISTDataLoader:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}

    def load_and_split_data(self, num_parties=VFLConfig.num_parties):
        """
        加载MNIST数据并根据空间区域分割给不同参与方
        使用Numba优化加速数据处理
        """
        # 下载并加载MNIST数据集 (保持2D图像格式)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 引入MNIST数据集
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # 转换为numpy数组 (保持图像格式 28x28)
        print("Converting datasets to numpy arrays...")
        X_train = np.array([x.numpy() for x, _ in train_dataset], dtype=np.float32)  # (N, 1, 28, 28)
        y_train = np.array([y for _, y in train_dataset], dtype=np.int64)
        X_test = np.array([x.numpy() for x, _ in test_dataset], dtype=np.float32)  # (N, 1, 28, 28)
        y_test = np.array([y for _, y in test_dataset], dtype=np.int64)

        print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # 按空间区域分割图像给不同参与方 - 使用Numba加速
        print("Splitting images spatially using Numba optimization...")
        train_party_data = self._split_image_spatially_optimized(X_train, num_parties)
        test_party_data = self._split_image_spatially_optimized(X_test, num_parties)

        # 对每个客户端的数据进行标准化 - 使用Numba加速
        print("Standardizing data using Numba optimization...")
        for i in range(num_parties):
            print(f"  Processing party {i + 1}/{num_parties}...")
            original_shape = train_party_data[i].shape

            # 展平数据
            train_flat = train_party_data[i].reshape(original_shape[0], -1)
            test_flat = test_party_data[i].reshape(test_party_data[i].shape[0], -1)

            # 计算均值和标准差
            mean = np.mean(train_flat, axis=0)
            std = np.std(train_flat, axis=0)

            # 使用Numba加速的标准化
            train_standardized = standardize_data_numba(train_flat, mean, std)
            test_standardized = standardize_data_numba(test_flat, mean, std)

            # 恢复原始形状
            train_party_data[i] = train_standardized.reshape(original_shape)
            test_party_data[i] = test_standardized.reshape(test_party_data[i].shape)

            # 保存标准化参数（用于后续可能的需求）
            self.scalers[i] = {'mean': mean, 'std': std}

        # 创建特征索引列表用于参考
        features_per_party = [party_data.shape[1:] for party_data in train_party_data]

        print("Data loading and splitting completed!")
        return train_party_data, test_party_data, y_train, y_test, features_per_party

    def _split_image_spatially_optimized(self, images, num_parties):
        """
        使用Numba优化的空间分割

        参数:
            images: 图像数据 (N, 1, 28, 28)
            num_parties: 参与方数量

        返回:
            party_data: 每个参与方的图像片段列表
        """
        _, channels, height, width = images.shape

        # 计算分割点
        cols_per_party = width // num_parties
        remainder = width % num_parties

        split_points = [0]
        current_col = 0
        for i in range(num_parties):
            current_col += cols_per_party
            if i < remainder:
                current_col += 1
            split_points.append(current_col)

        # 使用传统方法进行分割（因为Numba的列表返回限制）
        party_data = []
        for i in range(num_parties):
            start_col = split_points[i]
            end_col = split_points[i + 1]
            image_slice = images[:, :, :, start_col:end_col].copy()
            party_data.append(image_slice)

        return party_data

    def _split_image_spatially(self, images, num_parties):
        """
        按空间区域将图像分割给不同参与方（兼容性方法）

        参数:
            images: 图像数据 (N, 1, 28, 28)
            num_parties: 参与方数量

        返回:
            party_data: 每个参与方的图像片段列表
        """
        return self._split_image_spatially_optimized(images, num_parties)