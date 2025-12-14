# file: data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numba import jit, prange
import warnings
import os

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

            # 根据标签类型选择数据类型
            if labels.dtype == np.float32 or labels.dtype == np.float64:
                self.labels = torch.tensor(labels, dtype=torch.float32).to(device)
            else:
                self.labels = torch.tensor(labels, dtype=torch.long).to(device)
            print("Data preloaded to GPU successfully!")
        else:
            # 保持在CPU,转换为tensor
            self.party_data = [torch.tensor(data, dtype=torch.float32)
                               for data in party_data_list]

            if labels.dtype == np.float32 or labels.dtype == np.float64:
                self.labels = torch.tensor(labels, dtype=torch.float32)
            else:
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
        elif self.dataset_name == 'NUS_WIDE':
            return self._load_nus_wide(num_parties)
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
        print(f"Loading {self.dataset_name} dataset...")

        # 训练集使用数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # 测试集只做标准化
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

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

    def _load_nus_wide(self, num_parties):
        """
        加载NUS-WIDE数据集

        数据集结构预期：
        ./data/NUS-WIDE/
            ├── features/
            │   └── CH_features.npy (634维颜色直方图特征)
            └── labels/
                └── labels.npy (81个类别的多标签)
        """
        print(f"Loading {self.dataset_name} dataset...")

        data_root = './data/NUS-WIDE'

        # 检查数据集是否存在，如果不存在则生成合成数据
        if not os.path.exists(data_root):
            print(f"NUS-WIDE dataset not found at {data_root}")
            print("Generating synthetic data for testing...")
            return self._generate_synthetic_nus_wide(num_parties)

        # 尝试加载特征
        feature_file = os.path.join(data_root, 'features', 'CH_features.npy')
        if not os.path.exists(feature_file):
            # 尝试其他可能的路径
            alt_paths = [
                os.path.join(data_root, 'NUS_WID_Low_Level_Features', 'Low_Level_Features', 'CH_Train.dat'),
                os.path.join(data_root, 'CH_features.npy'),
            ]

            feature_file = None
            for path in alt_paths:
                if os.path.exists(path):
                    feature_file = path
                    break

            if feature_file is None:
                print("Feature file not found. Generating synthetic data...")
                return self._generate_synthetic_nus_wide(num_parties)

        # 加载特征
        try:
            if feature_file.endswith('.dat'):
                features = self._load_dat_file(feature_file)
            else:
                features = np.load(feature_file)
        except Exception as e:
            print(f"Error loading features: {e}")
            print("Generating synthetic data for testing...")
            return self._generate_synthetic_nus_wide(num_parties)

        # 加载标签
        label_file = os.path.join(data_root, 'labels', 'labels.npy')
        if not os.path.exists(label_file):
            # 尝试其他可能的路径
            alt_label_paths = [
                os.path.join(data_root, 'Concepts81.txt'),
                os.path.join(data_root, 'labels.npy'),
            ]

            label_file = None
            for path in alt_label_paths:
                if os.path.exists(path):
                    label_file = path
                    break

            if label_file is None:
                print("Label file not found. Generating synthetic data...")
                return self._generate_synthetic_nus_wide(num_parties)

        try:
            if label_file.endswith('.txt'):
                labels = self._load_concept_labels(label_file)
            else:
                labels = np.load(label_file)
        except Exception as e:
            print(f"Error loading labels: {e}")
            print("Generating synthetic data for testing...")
            return self._generate_synthetic_nus_wide(num_parties)

        # 数据预处理和分割
        return self._process_nus_wide_data(features, labels, num_parties)

    def _generate_synthetic_nus_wide(self, num_parties):
        """生成合成的NUS-WIDE数据用于测试"""
        print("=" * 80)
        print("Generating Synthetic NUS-WIDE Dataset".center(80))
        print("=" * 80)

        n_train = 10000
        n_test = 2000
        n_features = 634  # CH特征维度
        n_classes = 81

        print(f"Configuration:")
        print(f"  Training samples: {n_train}")
        print(f"  Test samples: {n_test}")
        print(f"  Feature dimension: {n_features}")
        print(f"  Number of classes: {n_classes}")
        print(f"  Multi-label: Yes (3-5 labels per sample)")

        # 生成合成特征
        train_features = np.random.randn(n_train, n_features).astype(np.float32)
        test_features = np.random.randn(n_test, n_features).astype(np.float32)

        # 生成合成多标签（每个样本平均3-5个标签）
        train_labels = np.zeros((n_train, n_classes), dtype=np.float32)
        test_labels = np.zeros((n_test, n_classes), dtype=np.float32)

        for i in range(n_train):
            n_labels = np.random.randint(3, 6)
            label_indices = np.random.choice(n_classes, n_labels, replace=False)
            train_labels[i, label_indices] = 1.0

        for i in range(n_test):
            n_labels = np.random.randint(3, 6)
            label_indices = np.random.choice(n_classes, n_labels, replace=False)
            test_labels[i, label_indices] = 1.0

        print(f"\n✓ Synthetic data generated successfully!")
        print("=" * 80 + "\n")

        return self._process_nus_wide_data(train_features, train_labels, num_parties,
                                           test_features, test_labels)

    def _process_nus_wide_data(self, train_features, train_labels, num_parties,
                               test_features=None, test_labels=None):
        """处理NUS-WIDE数据"""

        # 如果没有提供测试集，则分割训练集
        if test_features is None or test_labels is None:
            train_features, test_features, train_labels, test_labels = train_test_split(
                train_features, train_labels, test_size=0.2, random_state=42
            )

        print(f"Processing NUS-WIDE data:")
        print(f"  Train samples: {len(train_features)}")
        print(f"  Test samples: {len(test_features)}")
        print(f"  Feature dimension: {train_features.shape[1]}")
        print(f"  Number of classes: {train_labels.shape[1]}")

        # 特征标准化
        print("  Standardizing features...")
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)
        std[std < 1e-8] = 1.0

        train_features = (train_features - mean) / std
        test_features = (test_features - mean) / std

        # 垂直分割文本特征给不同参与方
        print("  Splitting features vertically across parties...")
        train_text_party = self._split_features_vertically(train_features, num_parties)
        test_text_party = self._split_features_vertically(test_features, num_parties)

        # 创建伪图像表示（将1D特征reshape成2D）
        print("  Creating pseudo-image representations...")
        train_image_party = []
        test_image_party = []

        for i in range(num_parties):
            # 将1D特征reshape成2D "图像"
            party_dim = train_text_party[i].shape[1]
            side_len = int(np.ceil(np.sqrt(party_dim)))

            # Padding到平方数
            pad_dim = side_len * side_len

            train_padded = np.pad(train_text_party[i],
                                  ((0, 0), (0, pad_dim - party_dim)),
                                  mode='constant')
            test_padded = np.pad(test_text_party[i],
                                 ((0, 0), (0, pad_dim - party_dim)),
                                 mode='constant')

            # Reshape成伪图像格式 (N, 1, H, W)
            train_img = train_padded.reshape(-1, 1, side_len, side_len)
            test_img = test_padded.reshape(-1, 1, side_len, side_len)

            train_image_party.append(train_img)
            test_image_party.append(test_img)

            print(f"    Party {i}: {train_text_party[i].shape[1]} features → {side_len}x{side_len} image")

        features_per_party = [party.shape[1:] for party in train_image_party]

        print(f"\n✓ NUS-WIDE data processing completed!")

        return train_image_party, test_image_party, train_labels, test_labels, features_per_party

    def _split_features_vertically(self, features, num_parties):
        """垂直分割特征"""
        n_samples, n_features = features.shape
        features_per_party = n_features // num_parties
        remainder = n_features % num_parties

        party_data = []
        start_idx = 0

        for i in range(num_parties):
            end_idx = start_idx + features_per_party
            if i < remainder:
                end_idx += 1

            party_features = features[:, start_idx:end_idx].copy()
            party_data.append(party_features)
            start_idx = end_idx

        return party_data

    def _load_dat_file(self, filepath):
        """加载.dat格式的特征文件"""
        features = []
        with open(filepath, 'r') as f:
            for line in f:
                values = line.strip().split()
                features.append([float(v) for v in values])
        return np.array(features, dtype=np.float32)

    def _load_concept_labels(self, filepath):
        """加载81个概念标签"""
        labels = []
        with open(filepath, 'r') as f:
            for line in f:
                values = line.strip().split()
                labels.append([int(v) for v in values])
        return np.array(labels, dtype=np.float32)

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