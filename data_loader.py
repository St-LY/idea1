import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MNISTDataLoader:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}

    def load_and_split_data(self, num_parties=2):
        """
        加载MNIST数据并根据特征列分割给不同参与方
        """
        # 下载并加载MNIST数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # 展平图像
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # 转换为numpy数组
        X_train = np.array([x.numpy() for x, _ in train_dataset])
        y_train = np.array([y for _, y in train_dataset])
        X_test = np.array([x.numpy() for x, _ in test_dataset])
        y_test = np.array([y for _, y in test_dataset])

        # 分割特征给不同参与方
        feature_dim = X_train.shape[1]
        features_per_party = np.array_split(range(feature_dim), num_parties)

        # 标准化数据
        train_party_data = []
        test_party_data = []

        for i, party_features in enumerate(features_per_party):
            scaler = StandardScaler()
            X_train_party = scaler.fit_transform(X_train[:, party_features])
            X_test_party = scaler.transform(X_test[:, party_features])

            self.scalers[i] = scaler
            train_party_data.append(X_train_party)
            test_party_data.append(X_test_party)

        return train_party_data, test_party_data, y_train, y_test, features_per_party