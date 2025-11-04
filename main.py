import torch
import numpy as np
import time
from data_loader import MNISTDataLoader
from server import Server
from client import Client
from config import VFLConfig
import matplotlib.pyplot as plt
import sys


def print_separator(title="", char="=", length=80):
    """打印分隔线"""
    if title:
        print("\n" + char * length)
        print(title.center(length))
        print(char * length)
    else:
        print(char * length)
    sys.stdout.flush()


def print_info(message):
    """打印信息并立即刷新"""
    print(message)
    sys.stdout.flush()


def main():
    # 加载配置
    config = VFLConfig
    print_separator("INITIALIZATION", "=")
    print_info("Initializing Vertical Federated Learning with Ring Signatures and Poisoning Defense...")
    print_info(f"Number of parties: {config.num_parties}")
    print_info(f"Batch size: {config.batch_size}")
    print_info(f"Pretraining epochs: {config.pretraining_epochs}")
    print_info(f"Formal training epochs: {config.epochs}")
    print_info(f"Device: {config.device}")

    # 打印GPU内存信息
    if config.use_cuda:
        print_info(f"\nGPU Memory Info:")
        print_info(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print_info(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
        print_info(f"  Cached: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB\n")

    # 加载和预处理数据
    print_separator("DATA LOADING", "-")
    print_info("Loading and preprocessing MNIST data...")
    data_loader = MNISTDataLoader()
    train_party_data, test_party_data, y_train, y_test, features_per_party = data_loader.load_and_split_data(
        config.num_parties
    )

    # 转换为PyTorch张量并移到GPU
    y_train_tensor = torch.tensor(y_train).long().to(config.device)
    y_test_tensor = torch.tensor(y_test).long().to(config.device)

    train_party_tensors = [torch.tensor(data).float().to(config.device) for data in train_party_data]
    test_party_tensors = [torch.tensor(data).float().to(config.device) for data in test_party_data]

    print_info(f"Training data shape: {[t.shape for t in train_party_tensors]}")
    print_info(f"Test data shape: {[t.shape for t in test_party_tensors]}")

    # 打印GPU内存使用
    if config.use_cuda:
        print_info(f"\nGPU Memory after loading data:")
        print_info(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
        print_info(f"  Cached: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # 初始化协调器和客户端
    print_separator("MODEL INITIALIZATION", "-")
    print_info("Initializing server and clients...")
    server = Server(config.top_model_input_dim, config.output_dim, config.learning_rate)
    clients = []

    # 创建用于记录损失的列表
    train_losses = []
    pretraining_losses = []

    # 收集所有客户端的RSA公钥(PEM格式)
    client_public_keys_pem = []

    for i, party_data in enumerate(train_party_tensors):
        input_dim = party_data.shape[1]
        client = Client(i, input_dim, config.learning_rate)
        client.set_server_public_key(server.get_public_key())
        clients.append(client)
        client_public_keys_pem.append(client.rsa_public_key_pem)
        server.register_client_public_key(i, client.rsa_public_key)
        print_info(f"Client {i} initialized with input dimension {input_dim}")

    # 为每个客户端设置环公钥(使用PEM格式)
    for client in clients:
        client.set_ring_public_keys(client_public_keys_pem.copy())
        client.set_all_clients(clients)

    print_info("\nAll models initialized successfully!")

    # ========== 预训练阶段 ==========
    print_separator("PHASE 1: PRETRAINING WITH POISONING DEFENSE", "=")
    print_info("Starting pretraining phase to collect reference vectors...")
    print_info(f"Epochs: {config.pretraining_epochs}")
    print_info(f"Purpose: Building defense baseline\n")

    for epoch in range(config.pretraining_epochs):
        print_separator(f"PRETRAINING EPOCH {epoch + 1}/{config.pretraining_epochs}", "-")
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
                # 计算中间结果
                intermediate = client.compute_intermediate(data)

                # 签名中间结果(包含epoch和batch信息以避免密钥镜像冲突)
                signature = client.ring_signature({
                    'intermediate': intermediate,
                    'batch_idx': batch_idx,
                    'epoch': epoch,
                    'phase': 'pretrain',
                    'client_id': client.client_id,
                    'timestamp': time.time()
                })

                # 随机转发
                forwarded_signature = client.send_to_random_client_direct(signature)
                signed_intermediates.append(forwarded_signature)

            # 协调器训练(会自动验证签名并应用防投毒机制)
            result = server.train_step(signed_intermediates, batch_labels)

            if result[0] is not None:
                loss, predictions, valid_count = result
                epoch_loss += loss
                valid_steps += 1
                pretraining_losses.append(loss)

                if batch_idx % 100 == 0:
                    accuracy = server.compute_accuracy(predictions, batch_labels)
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    poison_stats = server.get_poisoning_stats()

                    print_info(f"[Pretrain] Epoch {epoch + 1}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}, "
                               f"Acc: {accuracy:.4f}, Valid: {valid_count}/{len(clients)}, "
                               f"Defense: {poison_stats['accepted']}/{poison_stats['accepted'] + poison_stats['rejected']} accepted")

                    # 显示GPU内存使用
                    if config.use_cuda and batch_idx % 500 == 0:
                        print_info(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

            # 定期清理GPU缓存
            if config.use_cuda and batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        avg_loss = epoch_loss / valid_steps if valid_steps > 0 else 0

        stats = server.get_signature_stats()
        poison_stats = server.get_poisoning_stats()

        print_separator(f"PRETRAINING EPOCH {epoch + 1} SUMMARY", "-")
        print_info(f"  Average Loss: {avg_loss:.4f}")
        print_info(f"  Valid Steps: {valid_steps}/{num_batches}")
        print_info(f"  Signature Success Rate: {stats['success_rate']:.2%}")
        print_info(f"  Defense Acceptance Rate: {poison_stats['acceptance_rate']:.2%}")
        print_info(f"  Vectors Replaced: {poison_stats['replaced']}")
        print_info(f"  Time: {epoch_time:.2f}s")
        print_info(f"  Avg Batch Time: {avg_batch_time:.3f}s")

    # 完成预训练
    print_separator("FINALIZING PRETRAINING", "-")
    server.finalize_pretraining()
    print_info("Pretraining completed! Reference vectors established.")
    print_info(f"Final BorderPara value: {server.BorderPara:.4f}\n")

    # ========== 正式训练阶段 ==========
    print_separator("PHASE 2: FORMAL TRAINING WITH REFERENCE VECTORS", "=")
    print_info("Starting formal training phase with poisoning defense...")
    print_info(f"Epochs: {config.epochs}")
    print_info(f"Defense: Active (using reference vectors)\n")

    for epoch in range(config.epochs):
        print_separator(f"FORMAL TRAINING EPOCH {epoch + 1}/{config.epochs}", "-")
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

                # 签名(包含不同的phase标识)
                signature = client.ring_signature({
                    'intermediate': intermediate,
                    'batch_idx': batch_idx,
                    'epoch': epoch,
                    'phase': 'formal',
                    'client_id': client.client_id,
                    'timestamp': time.time()
                })

                forwarded_signature = client.send_to_random_client_direct(signature)
                signed_intermediates.append(forwarded_signature)

            # 协调器训练
            result = server.train_step(signed_intermediates, batch_labels)

            if result[0] is not None:
                loss, predictions, valid_count = result
                epoch_loss += loss
                valid_steps += 1
                train_losses.append(loss)

                if batch_idx % 100 == 0:
                    accuracy = server.compute_accuracy(predictions, batch_labels)
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    poison_stats = server.get_poisoning_stats()

                    print_info(f"[Formal] Epoch {epoch + 1}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}, "
                               f"Acc: {accuracy:.4f}, Valid: {valid_count}/{len(clients)}, "
                               f"Defense: {poison_stats['accepted']}/{poison_stats['accepted'] + poison_stats['rejected']} accepted")

                    # 显示GPU内存使用
                    if config.use_cuda and batch_idx % 500 == 0:
                        print_info(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

            if batch_idx % 500 == 0 and batch_idx > 0:
                stats = server.get_signature_stats()
                print_info(f"  Signature stats - Valid: {stats['valid_signatures']}, "
                           f"Invalid: {stats['invalid_signatures']}, Rate: {stats['success_rate']:.2%}")

            # 定期清理GPU缓存
            if config.use_cuda and batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        avg_loss = epoch_loss / valid_steps if valid_steps > 0 else 0

        stats = server.get_signature_stats()
        poison_stats = server.get_poisoning_stats()

        print_separator(f"FORMAL TRAINING EPOCH {epoch + 1} SUMMARY", "-")
        print_info(f"  Average Loss: {avg_loss:.4f}")
        print_info(f"  Valid Steps: {valid_steps}/{num_batches}")
        print_info(f"  Signature Success Rate: {stats['success_rate']:.2%}")
        print_info(f"  Defense Acceptance Rate: {poison_stats['acceptance_rate']:.2%}")
        print_info(f"  Time: {epoch_time:.2f}s")
        print_info(f"  Avg Batch Time: {avg_batch_time:.3f}s")

    # 最终统计
    final_stats = server.get_signature_stats()
    final_poison_stats = server.get_poisoning_stats()

    print_separator("TRAINING COMPLETED!", "=")
    print_info("\n📊 Final Statistics:")
    print_info("\nSignature Verification:")
    print_info(f"  ✓ Valid Signatures: {final_stats['valid_signatures']}")
    print_info(f"  ✗ Invalid Signatures: {final_stats['invalid_signatures']}")
    print_info(f"  Success Rate: {final_stats['success_rate']:.2%}")

    print_info("\nPoisoning Defense:")
    print_info(f"  ✓ Accepted Samples: {final_poison_stats['accepted']}")
    print_info(f"  ✗ Rejected Samples: {final_poison_stats['rejected']}")
    print_info(f"  ↻ Replaced Vectors: {final_poison_stats['replaced']}")
    print_info(f"  Acceptance Rate: {final_poison_stats['acceptance_rate']:.2%}")

    # 绘制损失曲线
    print_separator("GENERATING VISUALIZATIONS", "-")
    print_info("Creating loss curves...")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    if pretraining_losses:
        plt.plot(pretraining_losses, color='blue', linewidth=1.5)
        plt.title('Pretraining Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    if train_losses:
        plt.plot(train_losses, color='green', linewidth=1.5)
        plt.title('Formal Training Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    all_losses = pretraining_losses + train_losses
    if all_losses:
        plt.plot(all_losses, color='purple', linewidth=1.5)
        if pretraining_losses:
            plt.axvline(x=len(pretraining_losses), color='r', linestyle='--',
                        linewidth=2, label='Pretrain End')
        plt.title('Complete Training Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
    print_info("✓ Loss curves saved as 'training_loss_curves.png'")

    # 保存模型
    print_separator("SAVING MODELS", "-")
    try:
        torch.save(server.model.state_dict(), 'coordinator_model.pth')
        print_info("✓ Server model saved: coordinator_model.pth")

        for i, client in enumerate(clients):
            torch.save(client.model.state_dict(), f'client_{i}_model.pth')
            print_info(f"✓ Client {i} model saved: client_{i}_model.pth")

        print_info("\nAll models saved successfully!")
    except Exception as e:
        print_info(f"✗ Error saving models: {e}")
        return

    # ========== 测试阶段 ==========
    print_separator("PHASE 3: TESTING", "=")
    print_info("Evaluating model on test dataset...\n")

    test_start_time = time.time()

    trained_server = Server(config.top_model_input_dim, config.output_dim, config.learning_rate)
    trained_server.model.load_state_dict(torch.load('coordinator_model.pth'))
    trained_server.model.eval()

    trained_clients = []
    for i, party_data in enumerate(train_party_tensors):
        input_dim = party_data.shape[1]
        trained_client = Client(i, input_dim, config.learning_rate)
        trained_client.model.load_state_dict(torch.load(f'client_{i}_model.pth'))
        trained_client.model.eval()
        trained_clients.append(trained_client)

    test_predictions_list = []
    test_labels_list = []

    num_test_batches = (len(test_party_tensors[0]) + config.batch_size - 1) // config.batch_size

    print_info(f"Processing {num_test_batches} test batches...")

    with torch.no_grad():
        for batch_idx in range(num_test_batches):
            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, len(test_party_tensors[0]))

            batch_test_data = []
            for party_tensor in test_party_tensors:
                batch_test_data.append(party_tensor[batch_start:batch_end])

            batch_test_labels = y_test_tensor[batch_start:batch_end]
            test_labels_list.append(batch_test_labels)

            # 累加中间结果
            accumulated_intermediate = None
            for client, data in zip(trained_clients, batch_test_data):
                intermediate = client.compute_intermediate(data)
                if accumulated_intermediate is None:
                    accumulated_intermediate = intermediate
                else:
                    accumulated_intermediate += intermediate

            # 预测
            if accumulated_intermediate is not None:
                batch_predictions = trained_server.model(accumulated_intermediate)
                test_predictions_list.append(batch_predictions)

            if (batch_idx + 1) % 50 == 0:
                print_info(f"  Processed {batch_idx + 1}/{num_test_batches} batches...")

    test_predictions = torch.cat(test_predictions_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)

    test_time = time.time() - test_start_time
    test_accuracy = trained_server.compute_accuracy(test_predictions, test_labels)

    _, predicted = torch.max(test_predictions.data, 1)
    correct = (predicted == test_labels).sum().item()
    total = test_labels.size(0)

    print_separator("TEST RESULTS", "-")
    print_info(f"  Test Time: {test_time:.2f}s")
    print_info(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print_info(f"  Correct Predictions: {correct}/{total}")
    print_info(f"  Incorrect Predictions: {total - correct}/{total}")

    # 最终GPU内存统计
    if config.use_cuda:
        print_separator("GPU MEMORY STATISTICS", "-")
        print_info(f"  Current Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
        print_info(f"  Peak Allocated: {torch.cuda.max_memory_allocated(0) / 1024 ** 3:.2f} GB")
        print_info(f"  Current Cached: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    print_separator("ALL TASKS COMPLETED SUCCESSFULLY! ✓", "=")
    print_info("\n🎉 Training pipeline finished!")
    print_info(f"📁 Models saved in current directory")
    print_info(f"📊 Loss curves saved as 'training_loss_curves.png'\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置CUDA随机种子以确保可重复性
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # 可选: 设置确定性算法(可能会降低性能)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    main()
