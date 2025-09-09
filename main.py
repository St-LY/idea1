import torch
import numpy as np
from data_loader import MNISTDataLoader
from server import Server
from client import Client
from config import VFLConfig



def main():
    # 加载配置
    config = VFLConfig()

    # 加载和预处理数据
    data_loader = MNISTDataLoader()
    train_party_data, test_party_data, y_train, y_test, features_per_party = data_loader.load_and_split_data(
        config.num_parties
    )

    # 转换为PyTorch张量
    y_train_tensor = torch.tensor(y_train).long()
    y_test_tensor = torch.tensor(y_test).long()

    train_party_tensors = [torch.tensor(data).float() for data in train_party_data]
    test_party_tensors = [torch.tensor(data).float() for data in test_party_data]

    # 初始化协调器和客户端
    coordinator = Server(config.top_model_input_dim, config.output_dim, config.learning_rate)
    clients = []

    for i, party_data in enumerate(train_party_tensors):
        input_dim = party_data.shape[1]
        client = Client(i, input_dim, config.hidden_dims, config.learning_rate)
        client.set_public_key(coordinator.get_public_key())
        clients.append(client)

    # 训练过程
    for epoch in range(config.epochs):
        epoch_loss = 0
        num_batches = len(train_party_tensors[0]) // config.batch_size

        for batch_idx in range(num_batches):
            # 获取当前批次数据
            batch_data = []
            batch_labels = []
            batch_start = batch_idx * config.batch_size
            batch_end = (batch_idx + 1) * config.batch_size

            for party_tensor in train_party_tensors:
                batch_data.append(party_tensor[batch_start:batch_end])

            batch_labels = y_train_tensor[batch_start:batch_end]

            # 客户端计算中间结果
            intermediate_outputs = []
            for client, data in zip(clients, batch_data):
                intermediate = client.compute_intermediate(data)
                intermediate_outputs.append(intermediate)

            # 协调器训练
            loss, predictions = coordinator.train_step(intermediate_outputs, batch_labels)
            epoch_loss += loss

            # 计算准确率
            if batch_idx % 100 == 0:
                accuracy = coordinator.compute_accuracy(predictions, batch_labels)
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        print(f"Epoch {epoch + 1}/{config.epochs}, Average Loss: {epoch_loss / num_batches:.4f}")

    # 测试过程
    print("Testing model...")
    test_intermediates = []
    for client, data in zip(clients, test_party_tensors):
        test_intermediate = client.compute_intermediate(data)
        test_intermediates.append(test_intermediate)

    test_predictions = coordinator.predict(test_intermediates)
    test_accuracy = coordinator.compute_accuracy(test_predictions, y_test_tensor)
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()