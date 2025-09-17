# 联邦学习配置
class VFLConfig:
    def __init__(self):
        self.num_parties = 5  # 参与方数量
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 10

        # 每个客户端输出64维特征
        self.top_model_input_dim = 128 * self.num_parties

        self.output_dim = 10  # MNIST有10个类别

VFLConfig = VFLConfig()
