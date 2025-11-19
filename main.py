import torch
import numpy as np
import time
import secrets
import threading
import argparse
from data_loader import MultiDatasetLoader
from server import Server
from client import Client
from config import VFLConfig
import matplotlib.pyplot as plt
import sys

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VFL Training with Multiple Dataset Support')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN'],
                        help='Dataset to use for training')
    parser.add_argument('--num_parties', type=int, default=5,
                        help='Number of parties')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--pretrain_epochs', type=int, default=1,
                        help='Number of pretraining epochs')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of formal training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    return parser.parse_args()


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


def monitor_token_timeouts(clients, stop_event):
    """监控令牌超时的线程"""
    while not stop_event.is_set():
        time.sleep(5.0)

        for client in clients:
            is_timeout, reason = client.check_token_timeout()
            if is_timeout:
                print(f"\n[WARNING] Token timeout for Client {client.client_id}: {reason}")

        if stop_event.wait(5.0):
            break


def parallel_client_processing(clients, party_samples_list, batch_idx, epoch, phase):
    """
    并行处理所有客户端的计算任务
    使用线程队列系统
    """
    # 提交所有客户端的任务
    for client, data in zip(clients, party_samples_list):
        client.submit_task(data, batch_idx, epoch, phase)

    # 收集所有结果
    results = []
    timeout_per_client = 60.0  # 每个客户端最多等待60秒

    for i, client in enumerate(clients):
        result = client.get_result(timeout=timeout_per_client)
        if result is not None:
            results.append(result)
        else:
            print(f"[Warning] Client {client.client_id} returned None or timed out")

    return results


def main():
    # 解析命令行参数
    args = parse_args()

    # 使用指定的数据集创建配置
    config = VFLConfig(dataset_name=args.dataset)

    # 覆盖配置中的参数（如果命令行提供）
    config.num_parties = args.num_parties
    config.batch_size = args.batch_size
    config.pretraining_epochs = args.pretrain_epochs
    config.epochs = args.epochs
    config.learning_rate = args.lr

    print_separator(f"INITIALIZATION WITH TOKEN MECHANISM - {args.dataset}", "=")
    print_info("Initializing Vertical Federated Learning with:")
    print_info(f"  ✓ Dataset: {args.dataset}")
    print_info(f"  ✓ Token-based DOS Protection")
    print_info(f"  ✓ Ring Token Circulation")
    print_info(f"  ✓ Parallel Client Processing")
    print_info(f"  ✓ GPU Optimization")
    print_info(f"Number of parties: {config.num_parties}")
    print_info(f"Batch size: {config.batch_size}")
    print_info(f"Device: {config.device}")

    # 打印GPU信息
    if config.use_cuda:
        print_info(f"\nGPU Memory Info:")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print_info(f"  Total: {total_mem:.2f} GB")
        print_info(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

    # 加载数据
    print_separator("DATA LOADING", "-")
    print_info(f"Loading and preprocessing {args.dataset} data...")
    data_loader = MultiDatasetLoader(dataset_name=args.dataset)
    train_party_data, test_party_data, y_train, y_test, features_per_party = data_loader.load_and_split_data(
        config.num_parties
    )

    print_info("Creating optimized DataLoaders...")
    train_loader, test_loader = data_loader.create_dataloaders(
        train_party_data, test_party_data, y_train, y_test,
        batch_size=config.batch_size,
        device=config.device if config.preload_to_gpu else None,
        config=config
    )

    print_info(f"Train batches: {len(train_loader)}")
    print_info(f"Test batches: {len(test_loader)}")

    # 初始化服务器和客户端
    print_separator("MODEL INITIALIZATION WITH TOKEN SYSTEM", "-")
    print_info("Initializing server and clients with token mechanism...")

    # 生成全局令牌链主密钥
    chain_master_secret = secrets.token_bytes(32)
    print_info(f"Generated chain master secret")

    # 在main.py中找到服务器初始化部分，确保使用正确的输入维度
    server = Server(
        config.top_model_input_dim,  # 确保这个值正确
        config.output_dim,
        config.learning_rate,
        num_clients=config.num_parties,
        dataset_config=config.dataset_config
    )
    server.initialize_token_verifier(chain_master_secret)

    clients = []
    client_public_keys_pem = []

    # 获取第一个batch确定input_dim
    first_batch = next(iter(train_loader))
    party_samples_list, _ = first_batch

    for i, party_sample in enumerate(party_samples_list):
        input_dim = party_sample.shape[1]
        client = Client(
            i,
            input_dim,
            config.learning_rate,
            dataset_config=config.dataset_config
        )
        client.set_server_public_key(server.get_public_key())

        # 初始化令牌管理器
        client.initialize_token_manager(config.num_parties, chain_master_secret)

        clients.append(client)
        client_public_keys_pem.append(client.rsa_public_key_pem)
        server.register_client_public_key(i, client.rsa_public_key)
        print_info(f"Client {i} initialized")

    # 为每个客户端设置环公钥和客户端引用
    # 为每个客户端设置环公钥和客户端引用
    for client in clients:
        client.set_ring_public_keys(client_public_keys_pem.copy())
        client.set_all_clients(clients)

    # 调试信息：检查客户端输出维度和服务器输入维度
    print(f"Server top model input dim: {config.top_model_input_dim}")
    for i, client in enumerate(clients):
        # 测试客户端输出维度 - 使用多个样本避免BatchNorm问题
        if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
            # 灰度图像使用1个通道
            test_input = torch.randn(2, 1, 28, 28).to(config.device)
        else:
            # 彩色图像使用3个通道
            test_input = torch.randn(2, 3, 32, 32).to(config.device)  # 使用2个样本
        client.model.eval()  # 设置为评估模式
        with torch.no_grad():  # 不计算梯度
            test_output = client.model(test_input)
        client.model.train()  # 恢复训练模式
        print(f"Client {i} output dim: {test_output.shape[1]}")

    # 启动所有客户端的并行处理线程
    print_info("\nStarting parallel processing threads...")
    for client in clients:
        client.start_parallel_processing()
    print_info("All client processing threads started!")

    # 启动令牌超时监控线程
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_token_timeouts,
        args=(clients, stop_monitoring),
        daemon=True
    )
    monitor_thread.start()
    print_info("Token timeout monitoring started!")

    # 打印令牌初始状态
    print_info("\nInitial Token Status:")
    for client in clients:
        status = client.get_token_status()
        print_info(f"  Client {status['client_id']}: Has Token = {status['has_token']}")

    train_losses = []
    pretraining_losses = []

    # ========== 预训练阶段 ==========
    print_separator("PHASE 1: PRETRAINING WITH TOKEN PROTECTION", "=")
    print_info("Starting pretraining phase with parallel processing...")
    print_info("Clients will process in parallel, coordinated by token ring\n")

    for epoch in range(config.pretraining_epochs):
        print_separator(f"PRETRAINING EPOCH {epoch + 1}/{config.pretraining_epochs}", "-")
        epoch_start_time = time.time()
        epoch_loss = 0
        valid_steps = 0
        batch_times = []

        for batch_idx, (party_samples_list, batch_labels) in enumerate(train_loader):
            batch_start_time = time.time()

            # 如果数据未预加载到GPU,现在移到GPU
            if not config.preload_to_gpu:
                party_samples_list = [data.to(config.device) for data in party_samples_list]
                batch_labels = batch_labels.to(config.device)

            # 并行处理所有客户端
            signed_intermediates = parallel_client_processing(
                clients, party_samples_list, batch_idx, epoch, 'pretrain'
            )

            if len(signed_intermediates) == 0:
                print_info(f"[Warning] No valid results at batch {batch_idx}")
                continue

            # 服务器训练
            result = server.train_step(signed_intermediates, batch_labels)

            if result[0] is not None:
                loss, predictions, valid_count = result
                epoch_loss += loss
                valid_steps += 1
                pretraining_losses.append(loss)

                if batch_idx % 50 == 0:
                    accuracy = server.compute_accuracy(predictions, batch_labels)
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    poison_stats = server.get_poisoning_stats()
                    token_stats = server.get_token_stats()

                    print_info(f"[Pretrain] Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                               f"Loss: {loss:.4f}, Acc: {accuracy:.4f}, "
                               f"Valid: {valid_count}/{len(clients)}, "
                               f"Defense: {poison_stats['accepted']}/{poison_stats['accepted'] + poison_stats['rejected']}, "
                               f"Token: {token_stats['valid_proofs']}/{token_stats['total_verifications']}, "
                               f"Time: {batch_time:.3f}s")

                    if config.use_cuda and batch_idx % 200 == 0:
                        print_info(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        avg_loss = epoch_loss / valid_steps if valid_steps > 0 else 0
        throughput = len(train_loader) * config.batch_size / epoch_time if epoch_time > 0 else 0

        stats = server.get_signature_stats()
        poison_stats = server.get_poisoning_stats()
        token_stats = server.get_token_stats()

        print_separator(f"PRETRAINING EPOCH {epoch + 1} SUMMARY", "-")
        print_info(f"  Average Loss: {avg_loss:.4f}")
        print_info(f"  Valid Steps: {valid_steps}/{len(train_loader)}")
        print_info(f"  Signature Success: {stats['success_rate']:.2%}")
        print_info(f"  Token Verification: {token_stats['success_rate']:.2%}")
        print_info(f"  Defense Acceptance: {poison_stats['acceptance_rate']:.2%}")
        print_info(f"  Time: {epoch_time:.2f}s, Throughput: {throughput:.0f} samples/sec")

    # 完成预训练
    print_separator("FINALIZING PRETRAINING", "-")
    server.finalize_pretraining()
    print_info(f"Final BorderPara: {server.BorderPara:.4f}\n")

    # ========== 正式训练阶段 ==========
    print_separator("PHASE 2: FORMAL TRAINING WITH FULL PROTECTION", "=")
    print_info("Starting formal training...")

    for epoch in range(config.epochs):
        print_separator(f"FORMAL TRAINING EPOCH {epoch + 1}/{config.epochs}", "-")
        epoch_start_time = time.time()
        epoch_loss = 0
        valid_steps = 0
        batch_times = []

        for batch_idx, (party_samples_list, batch_labels) in enumerate(train_loader):
            batch_start_time = time.time()

            if not config.preload_to_gpu:
                party_samples_list = [data.to(config.device) for data in party_samples_list]
                batch_labels = batch_labels.to(config.device)

            # 并行处理
            signed_intermediates = parallel_client_processing(
                clients, party_samples_list, batch_idx, epoch, 'formal'
            )

            if len(signed_intermediates) == 0:
                continue

            result = server.train_step(signed_intermediates, batch_labels)

            if result[0] is not None:
                loss, predictions, valid_count = result
                epoch_loss += loss
                valid_steps += 1
                train_losses.append(loss)

                if batch_idx % 50 == 0:
                    accuracy = server.compute_accuracy(predictions, batch_labels)
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    poison_stats = server.get_poisoning_stats()
                    token_stats = server.get_token_stats()

                    print_info(f"[Formal] Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                               f"Loss: {loss:.4f}, Acc: {accuracy:.4f}, "
                               f"Valid: {valid_count}/{len(clients)}, "
                               f"Defense: {poison_stats['accepted']}/{poison_stats['accepted'] + poison_stats['rejected']}, "
                               f"Token: {token_stats['valid_proofs']}/{token_stats['total_verifications']}, "
                               f"Time: {batch_time:.3f}s")

        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        avg_loss = epoch_loss / valid_steps if valid_steps > 0 else 0
        throughput = len(train_loader) * config.batch_size / epoch_time if epoch_time > 0 else 0

        stats = server.get_signature_stats()
        poison_stats = server.get_poisoning_stats()
        token_stats = server.get_token_stats()

        print_separator(f"FORMAL TRAINING EPOCH {epoch + 1} SUMMARY", "-")
        print_info(f"  Average Loss: {avg_loss:.4f}")
        print_info(f"  Valid Steps: {valid_steps}/{len(train_loader)}")
        print_info(f"  Signature Success: {stats['success_rate']:.2%}")
        print_info(f"  Token Verification: {token_stats['success_rate']:.2%}")
        print_info(f"  Defense Acceptance: {poison_stats['acceptance_rate']:.2%}")
        print_info(f"  Time: {epoch_time:.2f}s, Throughput: {throughput:.0f} samples/sec")

    # 停止所有线程
    print_info("\nStopping threads...")
    stop_monitoring.set()
    monitor_thread.join(timeout=2)

    for client in clients:
        client.stop_parallel_processing()
    print_info("All threads stopped!")

    # 最终统计
    final_stats = server.get_signature_stats()
    final_poison_stats = server.get_poisoning_stats()
    final_token_stats = server.get_token_stats()

    print_separator(f"TRAINING COMPLETED - {args.dataset}!", "=")
    print_info("\n📊 Final Statistics:")

    print_info("\nSignature Verification:")
    print_info(f"  ✓ Valid: {final_stats['valid_signatures']}")
    print_info(f"  ✗ Invalid: {final_stats['invalid_signatures']}")
    print_info(f"  Success Rate: {final_stats['success_rate']:.2%}")

    print_info("\nToken Verification (DOS Protection):")
    print_info(f"  ✓ Valid Tokens: {final_token_stats['valid_proofs']}")
    print_info(f"  ✗ Invalid Tokens: {final_token_stats['invalid_proofs']}")
    print_info(f"  Success Rate: {final_token_stats['success_rate']:.2%}")

    print_info("\nPoisoning Defense:")
    print_info(f"  ✓ Accepted: {final_poison_stats['accepted']}")
    print_info(f"  ✗ Rejected: {final_poison_stats['rejected']}")
    print_info(f"  ↻ Replaced: {final_poison_stats['replaced']}")
    print_info(f"  Acceptance Rate: {final_poison_stats['acceptance_rate']:.2%}")

    # 绘制损失曲线
    print_separator("GENERATING VISUALIZATIONS", "-")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    if pretraining_losses:
        plt.plot(pretraining_losses, color='blue', linewidth=1.5)
        plt.title(f'{args.dataset} - Pretraining Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    if train_losses:
        plt.plot(train_losses, color='green', linewidth=1.5)
        plt.title(f'{args.dataset} - Formal Training Loss', fontsize=12, fontweight='bold')
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
        plt.title(f'{args.dataset} - Complete Training Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'training_loss_{args.dataset.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print_info(f"✓ Loss curves saved to {filename}!")

    # 保存模型
    print_separator("SAVING MODELS", "-")
    torch.save(server.model.state_dict(), f'coordinator_model_{args.dataset.lower()}.pth')
    print_info(f"✓ Server model saved")

    for i, client in enumerate(clients):
        torch.save(client.model.state_dict(), f'client_{i}_model_{args.dataset.lower()}.pth')
        print_info(f"✓ Client {i} model saved")

    # ========== 测试阶段 ==========
    print_separator("PHASE 3: TESTING", "=")
    print_info("Evaluating model on test dataset...")

    test_start_time = time.time()

    trained_server = Server(
        config.top_model_input_dim,
        config.output_dim,
        config.learning_rate,
        num_clients=config.num_parties,
        dataset_config=config.dataset_config
    )
    trained_server.model.load_state_dict(
        torch.load(f'coordinator_model_{args.dataset.lower()}.pth', weights_only=True)
    )
    trained_server.model.eval()

    trained_clients = []
    for i, client in enumerate(clients):
        trained_client = Client(
            i,
            client.model.conv_layers[0].in_channels,
            config.learning_rate,
            dataset_config=config.dataset_config
        )
        trained_client.model.load_state_dict(
            torch.load(f'client_{i}_model_{args.dataset.lower()}.pth', weights_only=True)
        )
        trained_client.model.eval()
        trained_clients.append(trained_client)

    test_predictions_list = []
    test_labels_list = []

    with torch.no_grad():
        for batch_idx, (party_samples_list, batch_labels) in enumerate(test_loader):
            if not config.preload_to_gpu:
                party_samples_list = [data.to(config.device) for data in party_samples_list]
                batch_labels = batch_labels.to(config.device)

            test_labels_list.append(batch_labels)

            # 使用累加方式：将所有客户端的中间结果相加
            accumulated_intermediate = None
            for client, data in zip(trained_clients, party_samples_list):
                intermediate = client.compute_intermediate(data)
                if accumulated_intermediate is None:
                    accumulated_intermediate = intermediate
                else:
                    accumulated_intermediate += intermediate  # 累加

            if accumulated_intermediate is not None:
                if config.use_amp:
                    with torch.cuda.amp.autocast(dtype=config.amp_dtype):
                        batch_predictions = trained_server.model(accumulated_intermediate)
                else:
                    batch_predictions = trained_server.model(accumulated_intermediate)
                test_predictions_list.append(batch_predictions)

            if (batch_idx + 1) % 50 == 0:
                print_info(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")

    test_predictions = torch.cat(test_predictions_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)

    test_time = time.time() - test_start_time
    test_accuracy = trained_server.compute_accuracy(test_predictions, test_labels)

    _, predicted = torch.max(test_predictions.data, 1)
    correct = (predicted == test_labels).sum().item()
    total = test_labels.size(0)

    print_separator("TEST RESULTS", "-")
    print_info(f"  Dataset: {args.dataset}")
    print_info(f"  Test Time: {test_time:.2f}s")
    print_info(f"  Throughput: {total / test_time:.0f} samples/sec")
    print_info(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print_info(f"  Correct: {correct}/{total}")

    if config.use_cuda:
        print_separator("GPU MEMORY STATISTICS", "-")
        print_info(f"  Current: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
        print_info(f"  Peak: {torch.cuda.max_memory_allocated(0) / 1024 ** 3:.2f} GB")

    print_separator("ALL TASKS COMPLETED! ✓", "=")
    print_info(f"\n🎉 Training on {args.dataset} finished!")
    print_info("📁 Models saved in current directory")
    print_info("📊 Loss curves saved\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    main()