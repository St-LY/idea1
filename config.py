# 联邦学习配置
class VFLConfig:
    def __init__(self):
        self.num_parties = 5  # 参与方数量
        self.batch_size = 64
        self.learning_rate = 0.01
        self.epochs = 10
        self.hidden_dims = [64, 128, 64]  # 底部模型隐藏层维度


        # 根据MNIST特征分割计算顶部模型输入维度
        # MNIST有784个特征，平均分给各参与方
        feature_dim = 784
        features_per_party = feature_dim // self.num_parties
        self.top_model_input_dim = 64 * self.num_parties  # 每个客户端输出64维特征

        self.output_dim = 10  # MNIST有10个类别

VFLConfig = VFLConfig()
