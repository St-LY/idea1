import torch


class DatasetConfig:
    """数据集特定配置"""

    # MNIST配置 - 保持不变
    MNIST = {
        'name': 'MNIST',
        'input_channels': 1,
        'image_size': (28, 28),
        'num_classes': 10,
        'bottom_model': {
            'layers': [
                {'type': 'conv2d', 'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 32},
                {'type': 'relu'},
                {'type': 'avgpool2d', 'kernel_size': 2, 'stride': 2},
                {'type': 'conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 64},
                {'type': 'relu'},
                {'type': 'maxpool', 'kernel_size': 2},
                {'type': 'adaptiveavgpool2d', 'output_size': (2, 2)}
            ],
            'fc_layers': [
                {'type': 'linear', 'in_features': 256, 'out_features': 128},
                {'type': 'batchnorm1d', 'num_features': 128},
                {'type': 'relu'}
            ],
            'fc_out': 128,
            'dropout': 0.0
        },
        'top_model': {
            'layers': [
                {'type': 'linear', 'in_features': 128, 'out_features': 64},
                {'type': 'batchnorm1d', 'num_features': 64},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.3},
                {'type': 'linear', 'in_features': 64, 'out_features': 32},
                {'type': 'batchnorm1d', 'num_features': 32},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.2},
                {'type': 'linear', 'in_features': 32, 'out_features': 10}
            ]
        }
    }

    FASHION_MNIST = {
        'name': 'FashionMNIST',
        'input_channels': 1,
        'image_size': (28, 28),
        'num_classes': 10,
        'bottom_model': {
            'layers': [
                {'type': 'conv2d', 'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 32},
                {'type': 'relu'},
                {'type': 'conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 64},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'conv2d', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 128},
                {'type': 'relu'},
                {'type': 'conv2d', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 256},
                {'type': 'relu'},
                {'type': 'adaptiveavgpool2d', 'output_size': (4, 4)}
            ],
            'fc_layers': [
                {'type': 'linear', 'in_features': 256 * 4 * 4, 'out_features': 512},
                {'type': 'batchnorm1d', 'num_features': 512},
                {'type': 'relu'},
            ],
            'fc_out': 512,
            'dropout': 0.0
        },
        'top_model': {
            'layers': [
                {'type': 'linear', 'in_features': 512, 'out_features': 128},
                {'type': 'batchnorm1d', 'num_features': 128},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.2},
                {'type': 'linear', 'in_features': 128, 'out_features': 32},
                {'type': 'batchnorm1d', 'num_features': 32},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.1},
                {'type': 'linear', 'in_features': 32, 'out_features': 10}
            ]
        }
    }

    CIFAR10 = {
        'name': 'CIFAR10',
        'input_channels': 3,
        'image_size': (32, 32),
        'num_classes': 10,
        'bottom_model': {
            'layers': [
                {'type': 'conv2d', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 64},
                {'type': 'relu'},
                {'type': 'conv2d', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 64},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'conv2d', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 128},
                {'type': 'relu'},
                {'type': 'conv2d', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 128},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'conv2d', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 256},
                {'type': 'relu'},
                {'type': 'conv2d', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 256},
                {'type': 'relu'},
                {'type': 'adaptiveavgpool2d', 'output_size': (4, 4)}
            ],
            'fc_layers': [
                {'type': 'linear', 'in_features': 256 * 4 * 4, 'out_features': 1024},
                {'type': 'batchnorm1d', 'num_features': 1024},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.3}
            ],
            'fc_out': 1024,
            'dropout': 0.3
        },
        'top_model': {
            'layers': [
                {'type': 'linear', 'in_features': 1024, 'out_features': 256},
                {'type': 'batchnorm1d', 'num_features': 256},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.2},
                {'type': 'linear', 'in_features': 256, 'out_features': 128},
                {'type': 'batchnorm1d', 'num_features': 128},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.1},
                {'type': 'linear', 'in_features': 128, 'out_features': 10}
            ]
        }
    }

    SVHN = {
        'name': 'SVHN',
        'input_channels': 3,
        'image_size': (32, 32),
        'num_classes': 10,
        'bottom_model': {
            'layers': [
                {'type': 'conv2d', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 64},
                {'type': 'relu'},
                {'type': 'conv2d', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 64},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'conv2d', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 128},
                {'type': 'relu'},
                {'type': 'conv2d', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 128},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'conv2d', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 256},
                {'type': 'relu'},
                {'type': 'conv2d', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 256},
                {'type': 'relu'},
                {'type': 'adaptiveavgpool2d', 'output_size': (4, 4)}
            ],
            'fc_layers': [
                {'type': 'linear', 'in_features': 256 * 4 * 4, 'out_features': 1024},
                {'type': 'batchnorm1d', 'num_features': 1024},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.3}
            ],
            'fc_out': 1024,
            'dropout': 0.3
        },
        'top_model': {
            'layers': [
                {'type': 'linear', 'in_features': 1024, 'out_features': 256},
                {'type': 'batchnorm1d', 'num_features': 256},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.2},
                {'type': 'linear', 'in_features': 256, 'out_features': 128},
                {'type': 'batchnorm1d', 'num_features': 128},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.1},
                {'type': 'linear', 'in_features': 128, 'out_features': 10}
            ]
        }
    }

    # NUS-WIDE 多模态数据集配置
    NUS_WIDE = {
        'name': 'NUS_WIDE',
        'input_channels': 1,  # 伪图像通道数
        'image_size': (26, 26),  # 伪图像大小（根据特征维度计算）
        'num_classes': 81,
        'is_multimodal': True,
        'is_multilabel': True,
        'feature_dim': 634,

        # 底部模型配置（处理文本特征的伪图像表示）
        'bottom_model': {
            'layers': [
                {'type': 'conv2d', 'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 32},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                {'type': 'batchnorm2d', 'num_features': 64},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'adaptiveavgpool2d', 'output_size': (4, 4)}
            ],
            'fc_layers': [
                {'type': 'linear', 'in_features': 64 * 4 * 4, 'out_features': 512},
                {'type': 'batchnorm1d', 'num_features': 512},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.4},
                {'type': 'linear', 'in_features': 512, 'out_features': 256},
                {'type': 'batchnorm1d', 'num_features': 256},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.3}
            ],
            'fc_out': 256,
            'dropout': 0.3
        },

        # Top model配置（多标签分类）
        'top_model': {
            'layers': [
                {'type': 'linear', 'in_features': 256, 'out_features': 512},
                {'type': 'batchnorm1d', 'num_features': 512},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.4},
                {'type': 'linear', 'in_features': 512, 'out_features': 256},
                {'type': 'batchnorm1d', 'num_features': 256},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.3},
                {'type': 'linear', 'in_features': 256, 'out_features': 81}
            ]
        }
    }


class VFLConfig:
    def __init__(self, dataset_name='MNIST'):
        # ========== 数据集配置 ==========
        self.dataset_name = dataset_name
        self.dataset_config = self._get_dataset_config(dataset_name)

        # ========== 基础配置 ==========
        self.num_parties = 5
        self.output_dim = self.dataset_config['num_classes']

        # ========== 训练配置 ==========
        self.batch_size = 512
        self.learning_rate = 0.01

        self.pretraining_epochs = 1
        self.epochs = 5

        self.top_model_input_dim = self.dataset_config['bottom_model']['fc_out']

        # ========== 防投毒配置 ==========
        self.max_vectors_per_label = 300
        self.distance_threshold_multiplier = 1.2

        # ========== CUDA配置 ==========
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # ========== GPU优化配置 ==========
        self.pin_memory = True
        self.num_workers = 4
        self.prefetch_factor = 3
        self.persistent_workers = True

        self.use_amp = self.use_cuda
        self.amp_dtype = torch.float16

        self.gradient_accumulation_steps = 1

        self.preload_to_gpu = True
        self.gpu_cache_size = 0.9

        # ========== 令牌机制配置 ==========
        self.token_timeout = 30.0
        self.token_verification_enabled = True

        # ========== 并行处理配置 ==========
        self.client_parallel_workers = 5
        self.processing_queue_size = 100

        # ========== CUDA优化选项 ==========
        if self.use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # ========== 打印配置信息 ==========
        self._print_configuration()

    def _get_dataset_config(self, dataset_name):
        """获取数据集配置"""
        dataset_configs = {
            'MNIST': DatasetConfig.MNIST,
            'CIFAR10': DatasetConfig.CIFAR10,
            'FashionMNIST': DatasetConfig.FASHION_MNIST,
            'SVHN': DatasetConfig.SVHN,
            'NUS_WIDE': DatasetConfig.NUS_WIDE,
        }

        if dataset_name not in dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                             f"Supported: {list(dataset_configs.keys())}")

        return dataset_configs[dataset_name]

    def _print_configuration(self):
        """打印配置信息"""
        print("\n" + "=" * 80)
        print("VFL Configuration".center(80))
        print("=" * 80)

        print(f"\n📊 Dataset: {self.dataset_name}")
        print(f"  Input Channels: {self.dataset_config['input_channels']}")
        print(f"  Image Size: {self.dataset_config['image_size']}")
        print(f"  Number of Classes: {self.dataset_config['num_classes']}")

        if self.dataset_config.get('is_multimodal'):
            print(f"  📌 Multimodal Dataset")
            print(f"  Feature Dimension: {self.dataset_config.get('feature_dim', 'N/A')}")

        print(f"  Bottom Model Output: {self.dataset_config['bottom_model']['fc_out']}")
        print(f"  Top Model Input: {self.top_model_input_dim} (using accumulation)")

        if self.use_cuda:
            print(f"\n✓ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")

            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"  Total GPU Memory: {total_memory:.2f} GB")
            print(f"  GPU Cache Size: {total_memory * self.gpu_cache_size:.2f} GB ({self.gpu_cache_size * 100:.0f}%)")

            print(f"\nGPU Optimizations:")
            print(f"  ✓ Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")
            print(f"  ✓ Data Preload to GPU: {'Enabled' if self.preload_to_gpu else 'Disabled'}")
            print(f"  ✓ TF32: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"  ✓ CuDNN Benchmark: {torch.backends.cudnn.benchmark}")

            compute_capability = torch.cuda.get_device_properties(0).major
            print(f"  Compute Capability: {compute_capability}.{torch.cuda.get_device_properties(0).minor}")
        else:
            print("⚠ CUDA is not available. Using CPU.")

        print(f"\nTraining Configuration:")
        print(f"  Parties: {self.num_parties}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Pretraining Epochs: {self.pretraining_epochs}")
        print(f"  Training Epochs: {self.epochs}")
        print(f"  Gradient Accumulation Steps: {self.gradient_accumulation_steps}")

        print(f"\nDataLoader Configuration:")
        print(f"  Pin Memory: {self.pin_memory}")
        print(f"  Num Workers: {self.num_workers}")
        print(f"  Prefetch Factor: {self.prefetch_factor}")
        print(f"  Persistent Workers: {self.persistent_workers}")

        print(f"\nSecurity Configuration:")
        print(f"  ✓ Ring Signature: Enabled")
        print(f"  ✓ Token Mechanism: {'Enabled' if self.token_verification_enabled else 'Disabled'}")
        print(f"  ✓ Poisoning Defense: Enabled")
        print(f"  Token Timeout: {self.token_timeout}s")
        print(f"  Aggregation Method: Accumulation (Sum)")

        print(f"\nParallel Processing:")
        print(f"  Client Workers: {self.client_parallel_workers}")
        print(f"  Queue Size: {self.processing_queue_size}")

        print("=" * 80 + "\n")

    def estimate_memory_usage(self, num_samples):
        """估算内存使用"""
        if not self.use_cuda:
            return

        img_size = self.dataset_config['image_size']
        channels = self.dataset_config['input_channels']
        bytes_per_sample = img_size[0] * img_size[1] * channels * 4
        total_data_bytes = num_samples * bytes_per_sample * self.num_parties
        total_data_gb = total_data_bytes / 1024 ** 3

        model_params_gb = 0.1

        activation_gb = (self.batch_size * self.top_model_input_dim * 4) / 1024 ** 3 * 10

        total_estimated = total_data_gb + model_params_gb + activation_gb

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        print(f"\n📊 Estimated Memory Usage:")
        print(f"  Data: {total_data_gb:.2f} GB")
        print(f"  Model: {model_params_gb:.2f} GB")
        print(f"  Activation: {activation_gb:.2f} GB")
        print(f"  Total Estimated: {total_estimated:.2f} GB")
        print(f"  Available: {gpu_memory:.2f} GB")

        if total_estimated > gpu_memory * 0.9:
            print(f"  ⚠ Warning: Estimated usage exceeds 90% of GPU memory!")
            print(f"  Consider reducing batch_size or disabling preload_to_gpu")
        else:
            utilization = (total_estimated / gpu_memory) * 100
            print(f"  ✓ Estimated GPU utilization: {utilization:.1f}%")


# 全局配置实例（默认使用MNIST）
default_vfl_config = VFLConfig('MNIST')