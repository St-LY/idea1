import torch
import numpy as np
import time
from data_loader import MNISTDataLoader
from server import Server
from client import Client
from config import VFLConfig

def main():
    # 加载配置
    config = VFLConfig
    print("Initializing Vertical Federated Learning with Ring Signatures...")
    print(f"Number of parties: {config.num_parties}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")

    # 加载和预处理数据
    print("Loading and preprocessing MNIST data...")
    data_loader = MNISTDataLoader()
    train_party_data, test_party_data, y_train, y_test, features_per_party = data_loader.load_and_split_data(
        config.num_parties
    )

    # 转换为PyTorch张量
    y_train_tensor = torch.tensor(y_train).long()
    y_test_tensor = torch.tensor(y_test).long()

    train_party_tensors = [torch.tensor(data).float() for data in train_party_data]
    test_party_tensors = [torch.tensor(data).float() for data in test_party_data]

    print(f"Training data shape: {[t.shape for t in train_party_tensors]}")
    print(f"Test data shape: {[t.shape for t in test_party_tensors]}")

    # 初始化协调器和客户端
    print("Initializing server and clients...")
    server = Server(config.top_model_input_dim, config.output_dim, config.learning_rate)
    clients = []

    # 收集所有客户端的RSA公钥（PEM格式）
    client_public_keys_pem = []

    for i, party_data in enumerate(train_party_tensors):
        input_dim = party_data.shape[1]
        client = Client(i, input_dim, config.hidden_dims, config.learning_rate)
        client.set_public_key(server.get_public_key())
        clients.append(client)
        # 收集PEM格式的公钥
        client_public_keys_pem.append(client.rsa_public_key_pem)

        # 向协调器注册客户端公钥（原始对象）
        server.register_client_public_key(i, client.rsa_public_key)
        print(f"Client {i} initialized with input dimension {input_dim}")

    # 为每个客户端设置环公钥（使用PEM格式）
    for client in clients:
        client.set_ring_public_keys(client_public_keys_pem.copy())
        # 设置所有客户端的引用以支持匿名转发
        client.set_all_clients(clients)

    print("Starting training process with ring signature verification...")

    # 训练过程
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        num_batches = len(train_party_tensors[0]) // config.batch_size
        valid_steps = 0
        batch_times = []

        for batch_idx in range(num_batches):
            batch_start_time = time.time()

            # 获取当前批次数据
            batch_data = []
            batch_start = batch_idx * config.batch_size
            batch_end = (batch_idx + 1) * config.batch_size

            for party_tensor in train_party_tensors:
                batch_data.append(party_tensor[batch_start:batch_end])

            batch_labels = y_train_tensor[batch_start:batch_end]

            # 客户端计算中间结果并使用环签名
            signed_intermediates = []

            for client, data in zip(clients, batch_data):
                intermediate = client.compute_intermediate(data)

                # 使用环签名保护消息
                signature = client.ring_signature({
                    'intermediate': intermediate,
                    'batch_idx': batch_idx,
                    'epoch': epoch,
                    'client_id': client.client_id,
                    'timestamp': time.time()
                })

                # 随机选择一个客户端进行转发以实现匿名
                forwarded_signature = client.send_to_random_client(signature)
                signed_intermediates.append(forwarded_signature)

            # 协调器训练（会自动验证签名）
            result = server.train_step(signed_intermediates, batch_labels)

            if result[0] is not None:  # 如果有有效的签名
                loss, predictions, valid_count = result
                epoch_loss += loss
                valid_steps += 1

                # 计算准确率
                if batch_idx % 100 == 0:
                    accuracy = server.compute_accuracy(predictions, batch_labels)
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss:.4f}, "
                          f"Accuracy: {accuracy:.4f}, Valid clients: {valid_count}/{len(clients)}, "
                          f"Time: {batch_time:.2f}s")

            # 每500个批次打印一次签名统计信息
            if batch_idx % 500 == 0 and batch_idx > 0:
                stats = server.get_signature_stats()
                print(f"Signature stats: Valid: {stats['valid_signatures']}, "
                      f"Invalid: {stats['invalid_signatures']}, "
                      f"Success rate: {stats['success_rate']:.2%}")

        # 计算 epoch 统计信息
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0

        # 打印 epoch 总结
        if valid_steps > 0:
            avg_loss = epoch_loss / valid_steps
        else:
            avg_loss = 0
            print(f"Epoch {epoch + 1}: No valid training steps completed!")

        stats = server.get_signature_stats()
        print(f"Epoch {epoch + 1}/{config.epochs} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Valid Steps: {valid_steps}/{num_batches}")
        print(f"  Signature Success Rate: {stats['success_rate']:.2%}")
        print(f"  Epoch Time: {epoch_time:.2f}s, Avg Batch Time: {avg_batch_time:.2f}s")
        print("-" * 60)

    # 训练完成后打印最终签名统计
    final_stats = server.get_signature_stats()
    print("\nTraining completed!")
    print(f"Final Signature Statistics:")
    print(f"  Total Valid Signatures: {final_stats['valid_signatures']}")
    print(f"  Total Invalid Signatures: {final_stats['invalid_signatures']}")
    print(f"  Overall Success Rate: {final_stats['success_rate']:.2%}")

    # 测试过程
    print("\nStarting testing process...")
    test_start_time = time.time()
    test_signed_intermediates = []

    for client, data in zip(clients, test_party_tensors):
        test_intermediate = client.compute_intermediate(data)

        # 使用环签名保护测试消息
        signature = client.ring_signature({
            'intermediate': test_intermediate,
            'is_test': True,
            'client_id': client.client_id,
            'timestamp': time.time()
        })

        # 随机选择一个客户端进行转发以实现匿名
        forwarded_signature = client.send_to_random_client(signature)
        test_signed_intermediates.append(forwarded_signature)

    test_predictions = server.predict(test_signed_intermediates)
    test_time = time.time() - test_start_time

    if test_predictions is not None:
        test_accuracy = server.compute_accuracy(test_predictions, y_test_tensor)
        print(f"Test completed in {test_time:.2f}s")
        print(f"Final Test Accuracy: {test_accuracy:.4f}")

        # 计算其他测试指标
        _, predicted = torch.max(test_predictions.data, 1)
        correct = (predicted == y_test_tensor).sum().item()
        total = y_test_tensor.size(0)

        print(f"Correct predictions: {correct}/{total}")
    else:
        print("Test failed: no valid signatures received during testing")

    # 保存模型（可选）
    print("\nSaving models...")
    try:
        # 保存协调器模型
        torch.save(server.model.state_dict(), 'coordinator_model.pth')

        # 保存各客户端模型
        for i, client in enumerate(clients):
            torch.save(client.model.state_dict(), f'client_{i}_model.pth')

        print("Models saved successfully!")
    except Exception as e:
        print(f"Error saving models: {e}")

    print("\nVertical Federated Learning with Ring Signatures completed!")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    main()
