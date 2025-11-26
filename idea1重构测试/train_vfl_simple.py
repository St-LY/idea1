# -*- coding: utf-8 -*-
"""
带随机打乱防御的联邦学习训练代码
目标: 测试中间向量随机打乱对重构攻击的防御效果
作者: Based on VFL Security Framework
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import time
from tqdm import tqdm

# 导入项目模块
from data_loader import MultiDatasetLoader
from models import BottomModel, TopModel
from config import VFLConfig



class SimplifiedClient:
    """简化的客户端(不使用加密和签名)"""
    def __init__(self, client_id, input_channels, dataset_config):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用config中的bottom_model配置
        bottom_config = dataset_config.get('bottom_model', None)
        self.model = BottomModel(input_channels, bottom_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        print(f"[Client {client_id}] Initialized with input_channels={input_channels}")

    def compute_intermediate(self, x):
        """计算中间特征向量"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            intermediate = self.model(x)
        return intermediate

    def train_step(self, x, gradient):
        """训练一步"""
        x = x.to(self.device)
        gradient = gradient.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(x)
        output.backward(gradient)
        self.optimizer.step()


class SimplifiedServer:
    """简化的服务器(不使用加密和防御机制)"""
    def __init__(self, input_dim, output_dim, dataset_config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用config中的top_model配置
        top_config = dataset_config.get('top_model', None)
        self.model = TopModel(input_dim, output_dim, top_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        print(f"[Server] Initialized with input_dim={input_dim}, output_dim={output_dim}")

    def train_step(self, aggregated_intermediate, labels):
        """训练一步"""
        aggregated_intermediate = aggregated_intermediate.to(self.device)
        labels = labels.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        predictions = self.model(aggregated_intermediate)
        loss = self.criterion(predictions, labels)
        loss.backward()

        # 计算梯度用于回传给客户端
        gradient = aggregated_intermediate.grad if aggregated_intermediate.requires_grad else None

        self.optimizer.step()

        # 计算准确率
        _, predicted = torch.max(predictions.data, 1)
        accuracy = (predicted == labels).float().mean().item()

        return loss.item(), accuracy, gradient


def train_federated_with_shuffle_defense(
    dataset_name='CIFAR10',
    num_clients=5,
    epochs=20,
    batch_size=128,
    save_vectors_every=5,  # 每N个epoch保存一次中间向量
    save_dir='shuffle_defense_vectors'
):
    """
    联邦学习训练 - 带中间向量随机打乱防御

    Args:
        dataset_name: 数据集名称 ('MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN')
        num_clients: 被动方数量
        epochs: 训练轮数
        batch_size: 批次大小
        save_vectors_every: 每隔多少epoch保存一次中间向量
        save_dir: 中间向量保存目录
    """

    print("="*80)
    print(f"联邦学习训练 - 带随机打乱防御".center(80))
    print("="*80)
    print(f"数据集: {dataset_name}")
    print(f"被动方数量: {num_clients}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"防御方式: 中间向量随机打乱")
    print("="*80 + "\n")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'vectors'), exist_ok=True)

    # 加载配置
    config = VFLConfig(dataset_name=dataset_name)
    config.num_parties = num_clients
    config.batch_size = batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 加载数据
    print("正在加载数据...")
    data_loader = MultiDatasetLoader(dataset_name=dataset_name)
    train_party_data, test_party_data, y_train, y_test, features_per_party = \
        data_loader.load_and_split_data(num_clients)

    train_loader, test_loader = data_loader.create_dataloaders(
        train_party_data, test_party_data, y_train, y_test,
        batch_size=batch_size,
        device=device if config.preload_to_gpu else None,
        config=config
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}\n")

    # 初始化客户端
    print("正在初始化客户端...")
    clients = []
    first_batch = next(iter(train_loader))
    party_samples_list, _ = first_batch

    for i, party_sample in enumerate(party_samples_list):
        input_channels = party_sample.shape[1]
        client = SimplifiedClient(i, input_channels, config.dataset_config)
        clients.append(client)

    # 初始化服务器
    print("\n正在初始化服务器...")
    server_input_dim = config.dataset_config['bottom_model']['fc_out']
    server = SimplifiedServer(
        input_dim=server_input_dim,
        output_dim=config.dataset_config['num_classes'],
        dataset_config=config.dataset_config
    )

    print("\n" + "="*80)
    print("开始训练".center(80))
    print("="*80 + "\n")

    # 训练循环
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        # 用于保存中间向量的列表
        if (epoch + 1) % save_vectors_every == 0:
            saved_vectors_before_shuffle = []  # 打乱前的向量
            saved_vectors_after_shuffle = []   # 打乱后的向量
            saved_labels = []
            saved_shuffle_indices = []  # 保存打乱的索引

        # 训练循环
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (party_samples_list, labels) in enumerate(pbar):

            # 移动数据到设备
            if not config.preload_to_gpu:
                party_samples_list = [data.to(device) for data in party_samples_list]
                labels = labels.to(device)

            # 步骤1: 各被动方生成中间特征向量
            intermediate_vectors = []
            for client_id, (client, data) in enumerate(zip(clients, party_samples_list)):
                intermediate = client.compute_intermediate(data)
                intermediate_vectors.append(intermediate)

            # 保存打乱前的中间向量(用于攻击测试)
            if (epoch + 1) % save_vectors_every == 0 and batch_idx < 50:  # 只保存前50个batch
                vectors_before = torch.stack(intermediate_vectors, dim=0)  # [num_clients, batch_size, feature_dim]
                saved_vectors_before_shuffle.append(vectors_before.cpu().detach())
                saved_labels.append(labels.cpu())

            # === 关键防御: 主动方随机打乱中间向量顺序 ===
            shuffle_indices = list(range(num_clients))
            np.random.shuffle(shuffle_indices)
            shuffled_vectors = [intermediate_vectors[i] for i in shuffle_indices]

            # 保存打乱后的向量和打乱索引
            if (epoch + 1) % save_vectors_every == 0 and batch_idx < 50:
                vectors_after = torch.stack(shuffled_vectors, dim=0)
                saved_vectors_after_shuffle.append(vectors_after.cpu().detach())
                saved_shuffle_indices.append(shuffle_indices)

            # 步骤2: 聚合(累加)
            aggregated = shuffled_vectors[0]
            for vec in shuffled_vectors[1:]:
                aggregated = aggregated + vec

            # 启用梯度追踪用于反向传播
            aggregated.requires_grad = True

            # 步骤3: 服务器训练
            loss, acc, gradient = server.train_step(aggregated, labels)

            # 步骤4: 将梯度分发给各客户端(反向打乱)
            if gradient is not None:
                # 反向打乱: 恢复原始顺序
                reverse_indices = [0] * num_clients
                for i, idx in enumerate(shuffle_indices):
                    reverse_indices[idx] = i

                # 客户端训练
                for client_id, client in enumerate(clients):
                    original_pos = reverse_indices[client_id]
                    client.train_step(party_samples_list[client_id], gradient)

            # 统计
            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}'})

        # Epoch统计
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        epoch_time = time.time() - epoch_start

        print(f"\n[Epoch {epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.4f}, "
              f"Acc: {avg_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")

        # 保存中间向量
        if (epoch + 1) % save_vectors_every == 0:
            vectors_file = os.path.join(save_dir, 'vectors', f'epoch_{epoch+1}.pkl')

            # 合并所有batch
            vectors_before_all = torch.cat(saved_vectors_before_shuffle, dim=1)  # [num_clients, total_samples, feature_dim]
            vectors_after_all = torch.cat(saved_vectors_after_shuffle, dim=1)
            labels_all = torch.cat(saved_labels, dim=0)

            save_data = {
                'epoch': epoch + 1,
                'vectors_before_shuffle': vectors_before_all.numpy(),  # [5, N, feature_dim]
                'vectors_after_shuffle': vectors_after_all.numpy(),
                'labels': labels_all.numpy(),
                'shuffle_indices_per_batch': saved_shuffle_indices,  # 每个batch的打乱索引
                'num_clients': num_clients,
                'dataset': dataset_name
            }

            with open(vectors_file, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"  ✓ 已保存中间向量到 {vectors_file}")
            print(f"    - 打乱前向量形状: {vectors_before_all.shape}")
            print(f"    - 打乱后向量形状: {vectors_after_all.shape}")
            print(f"    - 样本数: {labels_all.shape[0]}")

    # 保存模型
    print("\n" + "="*80)
    print("保存模型".center(80))
    print("="*80)

    server_model_path = os.path.join(save_dir, 'models', f'server_{dataset_name.lower()}.pth')
    torch.save(server.model.state_dict(), server_model_path)
    print(f"✓ 服务器模型已保存: {server_model_path}")

    for i, client in enumerate(clients):
        client_model_path = os.path.join(save_dir, 'models', f'client_{i}_{dataset_name.lower()}.pth')
        torch.save(client.model.state_dict(), client_model_path)
        print(f"✓ 客户端 {i} 模型已保存: {client_model_path}")

    # 测试
    print("\n" + "="*80)
    print("测试模型性能".center(80))
    print("="*80)

    server.model.eval()
    for client in clients:
        client.model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for party_samples_list, labels in tqdm(test_loader, desc="测试"):
            if not config.preload_to_gpu:
                party_samples_list = [data.to(device) for data in party_samples_list]
                labels = labels.to(device)

            # 生成中间向量并聚合
            intermediate_vectors = []
            for client, data in zip(clients, party_samples_list):
                intermediate = client.compute_intermediate(data)
                intermediate_vectors.append(intermediate)

            # 聚合
            aggregated = intermediate_vectors[0]
            for vec in intermediate_vectors[1:]:
                aggregated = aggregated + vec

            # 预测
            predictions = server.model(aggregated)
            _, predicted = torch.max(predictions.data, 1)

            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = test_correct / test_total
    print(f"\n测试准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"正确: {test_correct}/{test_total}")

    print("\n" + "="*80)
    print("训练完成!".center(80))
    print("="*80)
    print(f"\n模型保存目录: {os.path.join(save_dir, 'models')}")
    print(f"中间向量保存目录: {os.path.join(save_dir, 'vectors')}")
    print("\n可以使用这些中间向量进行黑盒重构攻击测试")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='训练带随机打乱防御的联邦学习模型')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN'],
                        help='数据集名称')
    parser.add_argument('--num_clients', type=int, default=5,
                        help='被动方数量')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--save_vectors_every', type=int, default=5,
                        help='每隔多少epoch保存一次中间向量')
    parser.add_argument('--save_dir', type=str, default='shuffle_defense_vectors',
                        help='保存目录')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 训练
    train_federated_with_shuffle_defense(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_vectors_every=args.save_vectors_every,
        save_dir=args.save_dir
    )