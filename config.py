import torch


# 联邦学习配置
class VFLConfig:
    def __init__(self):
        self.num_parties = 5  # 参与方数量
        self.batch_size = 64
        self.learning_rate = 0.001

        # 训练轮数配置
        self.pretraining_epochs = 1  # 预训练轮数(用于收集参考向量)
        self.epochs = 5  # 正式训练轮数

        # 每个客户端输出128维特征
        self.top_model_input_dim = 128

        self.output_dim = 10  # MNIST有10个类别

        # 防投毒配置
        self.max_vectors_per_label = 100  # 每个标签最多存储的向量数
        self.distance_threshold_multiplier = 1.2  # 距离阈值倍数

        # CUDA配置
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 打印设备信息
        if self.use_cuda:
            print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("CUDA is not available. Using CPU.")


VFLConfig = VFLConfig()