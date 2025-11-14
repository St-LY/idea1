import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numba import jit, prange
import warnings

warnings.filterwarnings('ignore')


class VFLDataset(Dataset):
    """自定义Dataset类,支持多方数据和GPU缓存"""

    def __init__(self, party_data_list, labels, device=None, preload_to_gpu=False):
        """
        Args:
            party_data_list: 每个参与方的数据列表 [party0_data, party1_data, ...]
            labels: 标签数据
            device: 目标设备
            preload_to_gpu: 是否预加载到GPU
        """
        self.num_parties = len(party_data_list)
        self.labels = labels
        self.device = device
        self.preload_to_gpu = preload_to_gpu and device is not None

        if self.preload_to_gpu:
            # 预加载所有数据到GPU
            print(f"Preloading {self.num_parties} parties' data to GPU...")
            self.party_data = [torch.tensor(data, dtype=torch.float32).to(device)
                               for data in party_data_list]
            self.labels = torch.tensor(labels, dtype=torch.long).to(device)
            print("Data preloaded to GPU successfully!")
        else:
            # 保持在CPU,转换为tensor
            self.party_data = [torch.tensor(data, dtype=torch.float32)
                               for data in party_data_list]
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回所有参与方的数据和标签
        party_samples = [party[idx] for party in self.party_data]
        label = self.labels[idx]
        return party_samples, label


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def standardize_data_numba(data, mean, std):
    """使用Numba并行标准化数据"""
    n_samples, n_features = data.shape
    result = np.empty_like(data)

    for i in prange(n_samples):
        for j in range(n_features):
            if std[j] < 1e-8:
                result[i, j] = 0.0
            else:
                result[i, j] = (data[i, j] - mean[j]) / std[j]

    return result


class MultiDatasetLoader:
    """支持多种数据集的加载器"""

    def __init__(self, dataset_name='MNIST', test_size=0.2, random_state=42):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}

    def load_and_split_data(self, num_parties=5):
        """根据数据集名称加载并分割数据"""
        if self.dataset_name == 'MNIST':
            return self._load_mnist(num_parties)
        elif self.dataset_name == 'CIFAR10':
            return self._load_cifar10(num_parties)
        elif self.dataset_name == 'FashionMNIST':
            return self._load_fashion_mnist(num_parties)
        elif self.dataset_name == 'SVHN':
            return self._load_svhn(num_parties)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _load_mnist(self, num_parties):
        """加载MNIST数据集"""
        print(f"Loading {self.dataset_name} dataset...")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        return self._process_dataset(train_dataset, test_dataset, num_parties)

    def _load_fashion_mnist(self, num_parties):
        """加载Fashion-MNIST数据集"""
        print(f"Loading {self.dataset_name} dataset...")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        return self._process_dataset(train_dataset, test_dataset, num_parties)

    def _load_cifar10(self, num_parties):
        """加载CIFAR-10数据集"""
        print(f"Loading {self.dataset_name} dataset...")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        return self._process_dataset(train_dataset, test_dataset, num_parties)

    def _load_svhn(self, num_parties):
        """加载SVHN数据集"""
        print(f"Loading {self.dataset_name} dataset...")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

        return self._process_dataset(train_dataset, test_dataset, num_parties)

    def _process_dataset(self, train_dataset, test_dataset, num_parties):
        """处理数据集：转换格式、分割、标准化"""
        # 转换为numpy数组
        print("Converting datasets to numpy arrays...")
        X_train = np.array([x.numpy() for x, _ in train_dataset], dtype=np.float32)
        y_train = np.array([y for _, y in train_dataset], dtype=np.int64)
        X_test = np.array([x.numpy() for x, _ in test_dataset], dtype=np.float32)
        y_test = np.array([y for _, y in test_dataset], dtype=np.int64)

        print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # 按空间区域分割图像
        print("Splitting images spatially...")
        train_party_data = self._split_image_spatially(X_train, num_parties)
        test_party_data = self._split_image_spatially(X_test, num_parties)

        # 对每个客户端的数据进行标准化
        print("Standardizing data using Numba optimization...")
        for i in range(num_parties):
            print(f"  Processing party {i + 1}/{num_parties}...")
            original_shape = train_party_data[i].shape

            train_flat = train_party_data[i].reshape(original_shape[0], -1)
            test_flat = test_party_data[i].reshape(test_party_data[i].shape[0], -1)

            mean = np.mean(train_flat, axis=0)
            std = np.std(train_flat, axis=0)

            train_standardized = standardize_data_numba(train_flat, mean, std)
            test_standardized = standardize_data_numba(test_flat, mean, std)

            train_party_data[i] = train_standardized.reshape(original_shape)
            test_party_data[i] = test_standardized.reshape(test_party_data[i].shape)

            self.scalers[i] = {'mean': mean, 'std': std}

        features_per_party = [party_data.shape[1:] for party_data in train_party_data]

        print("Data loading and splitting completed!")
        return train_party_data, test_party_data, y_train, y_test, features_per_party

    def _split_image_spatially(self, images, num_parties):
        """按空间区域将图像分割给不同参与方"""
        _, channels, height, width = images.shape

        cols_per_party = width // num_parties
        remainder = width % num_parties

        split_points = [0]
        current_col = 0
        for i in range(num_parties):
            current_col += cols_per_party
            if i < remainder:
                current_col += 1
            split_points.append(current_col)

        party_data = []
        for i in range(num_parties):
            start_col = split_points[i]
            end_col = split_points[i + 1]
            image_slice = images[:, :, :, start_col:end_col].copy()
            party_data.append(image_slice)

        return party_data

    def create_dataloaders(self, train_party_data, test_party_data, y_train, y_test,
                           batch_size=128, device=None, config=None):
        """创建优化的DataLoader"""

        if config is None:
            # 使用默认配置
            pin_memory = True
            num_workers = 4
            prefetch_factor = 3
            persistent_workers = True
            preload_to_gpu = False
        else:
            pin_memory = config.pin_memory
            num_workers = config.num_workers
            prefetch_factor = config.prefetch_factor
            persistent_workers = config.persistent_workers
            preload_to_gpu = config.preload_to_gpu

        # 创建Dataset
        train_dataset = VFLDataset(
            train_party_data, y_train,
            device=device,
            preload_to_gpu=preload_to_gpu
        )
        test_dataset = VFLDataset(
            test_party_data, y_test,
            device=device,
            preload_to_gpu=preload_to_gpu
        )

        # 如果数据已预加载到GPU,不需要pin_memory和workers
        if preload_to_gpu:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # GPU数据不需要workers
                pin_memory=False
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
        else:
            # CPU数据使用优化的DataLoader配置
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers
            )

        return train_loader, test_loader


# 保持向后兼容的别名
MNISTDataLoader = MultiDatasetLoader