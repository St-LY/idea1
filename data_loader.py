import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import VFLConfig


class MNISTDataLoader:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}

    def load_and_split_data(self, num_parties=VFLConfig.num_parties):
        """
        加载MNIST数据并根据空间区域分割给不同参与方
        """
        # 下载并加载MNIST数据集 (保持2D图像格式)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 引入MNIST数据集
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # 转换为numpy数组 (保持图像格式 28x28)
        X_train = np.array([x.numpy() for x, _ in train_dataset])  # (N, 1, 28, 28)
        y_train = np.array([y for _, y in train_dataset])
        X_test = np.array([x.numpy() for x, _ in test_dataset])  # (N, 1, 28, 28)
        y_test = np.array([y for _, y in test_dataset])

        # 按空间区域分割图像给不同参与方
        # 将28x28图像分割成5个垂直条带 (大约5-6像素宽)
        train_party_data = self._split_image_spatially(X_train, num_parties)
        test_party_data = self._split_image_spatially(X_test, num_parties)

        # 对每个客户端的数据进行标准化
        for i in range(num_parties):
            scaler = StandardScaler()
            # 将图像数据展平进行标准化
            original_shape = train_party_data[i].shape
            train_party_data[i] = scaler.fit_transform(
                train_party_data[i].reshape(original_shape[0], -1)
            ).reshape(original_shape)

            test_party_data[i] = scaler.transform(
                test_party_data[i].reshape(test_party_data[i].shape[0], -1)
            ).reshape(test_party_data[i].shape)

            self.scalers[i] = scaler

        # 创建特征索引列表用于参考
        features_per_party = [party_data.shape[1:] for party_data in train_party_data]

        return train_party_data, test_party_data, y_train, y_test, features_per_party

    def _split_image_spatially(self, images, num_parties):
        """
        按空间区域将图像分割给不同参与方

        参数:
            images: 图像数据 (N, 1, 28, 28)
            num_parties: 参与方数量

        返回:
            party_data: 每个参与方的图像片段列表
        """
        _, channels, height, width = images.shape
        party_data = []

        # 计算每个客户端应该拥有的列数
        cols_per_party = width // num_parties
        remainder = width % num_parties

        start_col = 0
        for i in range(num_parties):
            # 分配列数，确保总和等于width
            end_col = start_col + cols_per_party
            if i < remainder:
                end_col += 1

            # 提取图像片段
            image_slice = images[:, :, :, start_col:end_col]
            party_data.append(image_slice)
            start_col = end_col

        return party_data
