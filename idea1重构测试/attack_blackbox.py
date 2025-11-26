# -*- coding: utf-8 -*-
"""
黑盒重构攻击 - 测试随机打乱防御的有效性
作者: Based on Model Inversion Attack Framework
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim


def get_psnr(img1, img2, peak=1.0):
    """计算PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(peak ** 2 / mse)


def get_ssim(img1, img2):
    """计算SSIM"""
    # 确保图像格式正确
    if img1.ndim == 3:
        # 如果是彩色图像 (C, H, W), 转换为 (H, W, C)
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        channel_axis = -1  # 通道在最后一维
    else:
        channel_axis = None  # 灰度图像

    data_range = max(img1.max() - img1.min(), img2.max() - img2.min())

    # 使用较小的win_size以适应小图像
    try:
        return compare_ssim(img1, img2, data_range=data_range, channel_axis=channel_axis)
    except ValueError:
        # 如果默认窗口太大，使用较小的窗口
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win_size < 3:
            win_size = 3
        return compare_ssim(img1, img2, data_range=data_range,
                            channel_axis=channel_axis, win_size=win_size)


class DecoderNetwork(nn.Module):
    """反向解码器网络 - 从中间特征重构原始输入"""

    def __init__(self, feature_dim, output_channels, output_size):
        """
        Args:
            feature_dim: 中间特征维度
            output_channels: 输出通道数(1 for MNIST, 3 for CIFAR10)
            output_size: 输出图像大小 (H, W)
        """
        super(DecoderNetwork, self).__init__()

        self.feature_dim = feature_dim
        self.output_channels = output_channels
        self.output_size = output_size

        # 全连接层将特征映射到合适的维度
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)

        # 计算转置卷积的起始大小
        if output_size[0] == 28:  # MNIST
            self.start_size = 7
            self.fc_out = nn.Linear(2048, 128 * self.start_size * self.start_size)

            # 转置卷积层
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7->14
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14->28
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, output_channels, kernel_size=3, padding=1),  # 28->28
                nn.Tanh()
            )
        else:  # CIFAR10
            self.start_size = 8
            self.fc_out = nn.Linear(2048, 256 * self.start_size * self.start_size)

            # 转置卷积层
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8->16
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16->32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, output_channels, kernel_size=3, padding=1),  # 32->32
                nn.Tanh()
            )

    def forward(self, x):
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc_out(x)

        # 重塑为卷积输入
        if self.output_size[0] == 28:
            x = x.view(-1, 128, self.start_size, self.start_size)
        else:
            x = x.view(-1, 256, self.start_size, self.start_size)

        # 转置卷积
        x = self.deconv_layers(x)
        return x


def train_decoder(
        vectors_file,
        original_images,
        target_client_id=0,
        epochs=100,
        batch_size=64,
        lr=0.001,
        device='cuda'
):
    """
    训练反向解码器

    Args:
        vectors_file: 保存的中间向量文件路径
        original_images: 原始图像数据
        target_client_id: 目标攻击的客户端ID (0-4)
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        device: 计算设备
    """

    print("\n" + "=" * 80)
    print(f"训练反向解码器 - 目标客户端 {target_client_id}".center(80))
    print("=" * 80)

    # 加载中间向量数据
    print(f"加载中间向量: {vectors_file}")
    with open(vectors_file, 'rb') as f:
        data = pickle.load(f)

    vectors_before_shuffle = data['vectors_before_shuffle']  # [5, N, feature_dim]
    labels = data['labels']
    num_clients = data['num_clients']
    dataset_name = data['dataset']

    print(f"数据集: {dataset_name}")
    print(f"客户端数: {num_clients}")
    print(f"向量形状: {vectors_before_shuffle.shape}")
    print(f"样本数: {len(labels)}")

    # === 关键假设: 攻击者错误地认为位置固定 ===
    # 攻击者假设第target_client_id个位置的向量始终来自客户端target_client_id
    # 但实际上由于打乱,这个假设是错误的
    assumed_vectors = vectors_before_shuffle[target_client_id]  # [N, feature_dim]

    print(f"\n攻击者假设:")
    print(f"  - 位置 {target_client_id} 的向量始终来自客户端 {target_client_id}")
    print(f"  - 提取的向量形状: {assumed_vectors.shape}")

    # 确定图像参数
    if dataset_name in ['MNIST', 'FashionMNIST']:
        output_channels = 1
        output_size = (28, 28)
    else:  # CIFAR10, SVHN
        output_channels = 3
        output_size = (32, 32)

    # 获取完整的原始图像
    # 注意：所有客户端对应同一份完整图像数据
    target_images = original_images[0][:len(assumed_vectors)]  # [N, C, H, W]

    print(f"\n原始图像形状: {target_images.shape}")
    print(f"  - 使用完整图像作为重构目标")
    print(f"  - 图像数量: {len(target_images)}")

    # 将图像归一化到[-1, 1]以匹配Tanh输出
    if dataset_name in ['MNIST', 'FashionMNIST']:
        # MNIST图像已经在[0, 1]范围，转换到[-1, 1]
        target_images = target_images * 2 - 1
    else:
        # CIFAR10已经被normalize，需要先反归一化再重新归一化到[-1,1]
        # 反归一化
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        target_images = target_images * std + mean  # 恢复到[0, 1]
        target_images = target_images * 2 - 1  # 转换到[-1, 1]

    print(f"  - 图像范围: [{target_images.min():.3f}, {target_images.max():.3f}]")

    # 创建数据集
    num_samples = len(assumed_vectors)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_size = int(0.8 * num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_vectors = torch.FloatTensor(assumed_vectors[train_indices]).to(device)
    train_images = target_images[train_indices].to(device)
    val_vectors = torch.FloatTensor(assumed_vectors[val_indices]).to(device)
    val_images = target_images[val_indices].to(device)

    print(f"\n训练集: {len(train_indices)} 样本")
    print(f"验证集: {len(val_indices)} 样本")

    # 创建解码器
    feature_dim = assumed_vectors.shape[1]
    decoder = DecoderNetwork(feature_dim, output_channels, output_size).to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\n解码器参数: ")
    print(f"  - 输入维度: {feature_dim}")
    print(f"  - 输出通道: {output_channels}")
    print(f"  - 输出大小: {output_size}")
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  - 总参数量: {total_params:,}")

    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        decoder.train()
        epoch_loss = 0.0
        num_batches = 0

        # 训练
        for i in range(0, len(train_indices), batch_size):
            end_idx = min(i + batch_size, len(train_indices))
            batch_vectors = train_vectors[i:end_idx]
            batch_images = train_images[i:end_idx]

            optimizer.zero_grad()
            reconstructed = decoder(batch_vectors)
            loss = criterion(reconstructed, batch_images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # 验证
        decoder.eval()
        with torch.no_grad():
            val_reconstructed = decoder(val_vectors)
            val_loss = criterion(val_reconstructed, val_images).item()
            val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = decoder.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")

    # 加载最佳模型
    decoder.load_state_dict(best_model_state)

    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")

    return decoder, train_losses, val_losses


def evaluate_reconstruction(
        decoder,
        vectors_file,
        original_images,
        target_client_id=0,
        num_samples=100,
        save_dir='attack_results',
        device='cuda'
):
    """
    评估重构效果

    Args:
        decoder: 训练好的解码器
        vectors_file: 中间向量文件
        original_images: 原始图像
        target_client_id: 目标客户端ID
        num_samples: 评估样本数
        save_dir: 结果保存目录
        device: 计算设备
    """

    print("\n" + "=" * 80)
    print("评估重构效果".center(80))
    print("=" * 80)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)

    # 加载数据
    with open(vectors_file, 'rb') as f:
        data = pickle.load(f)

    vectors_before_shuffle = data['vectors_before_shuffle'][target_client_id]  # [N, feature_dim]
    dataset_name = data['dataset']

    # 获取完整的原始图像
    target_images = original_images[0][:len(vectors_before_shuffle)]  # [N, C, H, W]

    # 将图像归一化到[-1, 1]
    if dataset_name in ['MNIST', 'FashionMNIST']:
        target_images = target_images * 2 - 1
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        target_images = target_images * std + mean
        target_images = target_images * 2 - 1

    # 随机选择样本
    indices = np.random.choice(len(vectors_before_shuffle), num_samples, replace=False)

    decoder.eval()

    psnr_list = []
    ssim_list = []

    print(f"\n评估 {num_samples} 个样本...")

    with torch.no_grad():
        for idx in tqdm(indices):
            # 获取向量和原始图像
            vector = torch.FloatTensor(vectors_before_shuffle[idx:idx + 1]).to(device)
            original = target_images[idx].cpu().numpy()  # [C, H, W], 范围[-1, 1]

            # 重构
            reconstructed = decoder(vector).cpu().numpy()[0]  # [C, H, W], 范围[-1, 1]

            # 归一化到[0, 1]用于计算指标和显示
            original_np = np.clip((original + 1) / 2, 0, 1)
            reconstructed_np = np.clip((reconstructed + 1) / 2, 0, 1)

            # 计算指标
            psnr = get_psnr(original_np, reconstructed_np)
            ssim = get_ssim(original_np, reconstructed_np)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

    # 统计
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)

    print(f"\n重构质量评估:")
    print(f"  PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

    # 可视化一些样本
    print(f"\n保存可视化结果...")
    num_vis = min(10, num_samples)
    vis_indices = indices[:num_vis]

    fig, axes = plt.subplots(3, num_vis, figsize=(num_vis * 2, 6))

    with torch.no_grad():
        for i, idx in enumerate(vis_indices):
            vector = torch.FloatTensor(vectors_before_shuffle[idx:idx + 1]).to(device)
            original = target_images[idx].cpu().numpy()  # [C, H, W], 范围[-1, 1]
            reconstructed = decoder(vector).cpu().numpy()[0]  # [C, H, W], 范围[-1, 1]

            # 归一化到[0, 1]
            original_np = np.clip((original + 1) / 2, 0, 1)
            reconstructed_np = np.clip((reconstructed + 1) / 2, 0, 1)
            difference = np.abs(original_np - reconstructed_np)

            # 显示
            if original_np.shape[0] == 1:  # 灰度图
                axes[0, i].imshow(original_np[0], cmap='gray')
                axes[1, i].imshow(reconstructed_np[0], cmap='gray')
                axes[2, i].imshow(difference[0], cmap='hot')
            else:  # 彩色图
                axes[0, i].imshow(np.transpose(original_np, (1, 2, 0)))
                axes[1, i].imshow(np.transpose(reconstructed_np, (1, 2, 0)))
                axes[2, i].imshow(np.transpose(difference, (1, 2, 0)))

            axes[0, i].axis('off')
            axes[1, i].axis('off')
            axes[2, i].axis('off')

            if i == 0:
                axes[0, i].set_ylabel('Original', fontsize=12)
                axes[1, i].set_ylabel('Reconstructed', fontsize=12)
                axes[2, i].set_ylabel('Difference', fontsize=12)

    plt.tight_layout()
    vis_path = os.path.join(save_dir, 'images', f'reconstruction_client_{target_client_id}.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 可视化结果已保存: {vis_path}")

    # 保存评估结果
    results = {
        'target_client_id': target_client_id,
        'dataset': dataset_name,
        'num_samples': num_samples,
        'psnr_mean': avg_psnr,
        'psnr_std': std_psnr,
        'ssim_mean': avg_ssim,
        'ssim_std': std_ssim,
        'psnr_list': psnr_list,
        'ssim_list': ssim_list
    }

    results_path = os.path.join(save_dir, f'results_client_{target_client_id}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"  ✓ 评估结果已保存: {results_path}")

    return results


def load_full_images(dataset_name, num_samples=None):
    """
    加载完整的原始图像（未分割）

    Args:
        dataset_name: 数据集名称
        num_samples: 加载的样本数（None表示全部）

    Returns:
        images: [N, C, H, W] 形状的完整图像
    """
    import torchvision
    import torchvision.transforms as transforms

    print(f"\n加载完整原始图像: {dataset_name}")

    # 定义转换（与训练时保持一致）
    if dataset_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:  # CIFAR10, SVHN
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    # 加载数据集
    if dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
    elif dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
    elif dataset_name == 'SVHN':
        dataset = torchvision.datasets.SVHN(
            root='./data', split='train', download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 提取所有图像
    images = []
    labels = []

    if num_samples is None:
        num_samples = len(dataset)

    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        images.append(img)
        labels.append(label)

    images = torch.stack(images)  # [N, C, H, W]
    labels = torch.LongTensor(labels)

    print(f"  加载完成: {images.shape}")
    print(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")

    return images, labels


def main():
    """主函数 - 黑盒重构攻击"""
    import argparse

    parser = argparse.ArgumentParser(description='黑盒重构攻击 - 测试随机打乱防御')
    parser.add_argument('--vectors_file', type=str, required=True,
                        help='中间向量文件路径')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN'],
                        help='数据集名称')
    parser.add_argument('--target_client', type=int, default=0,
                        help='目标攻击的客户端ID (0-4)')
    parser.add_argument('--decoder_epochs', type=int, default=100,
                        help='解码器训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='评估样本数')
    parser.add_argument('--save_dir', type=str, default='attack_results',
                        help='结果保存目录')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载完整的原始图像数据（未分割）
    full_images, full_labels = load_full_images(args.dataset)

    # 为了兼容后续代码，我们创建一个列表，每个客户端都指向同一份完整图像
    # （因为我们的目标是重构完整图像，而不是分割后的部分图像）
    original_images = [full_images for _ in range(5)]

    print(f"\n原始图像数据已准备:")
    print(f"  形状: {full_images.shape}")
    print(f"  注意: 攻击目标是重构完整图像，而不是分割后的部分")

    # 训练解码器
    decoder, train_losses, val_losses = train_decoder(
        vectors_file=args.vectors_file,
        original_images=original_images,
        target_client_id=args.target_client,
        epochs=args.decoder_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    # 保存解码器
    decoder_path = os.path.join(args.save_dir, f'decoder_client_{args.target_client}.pth')
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"\n✓ 解码器已保存: {decoder_path}")

    # 评估重构效果
    results = evaluate_reconstruction(
        decoder=decoder,
        vectors_file=args.vectors_file,
        original_images=original_images,
        target_client_id=args.target_client,
        num_samples=args.eval_samples,
        save_dir=args.save_dir,
        device=device
    )

    print("\n" + "=" * 80)
    print("攻击完成!".center(80))
    print("=" * 80)
    print(f"\n由于随机打乱防御,攻击者使用了错误的向量-客户端映射")
    print(f"预期效果: 重构质量应该显著降低 (低PSNR/SSIM)")
    print(f"\n实际结果:")
    print(f"  PSNR: {results['psnr_mean']:.2f} dB")
    print(f"  SSIM: {results['ssim_mean']:.4f}")

    # 判断防御效果
    if results['ssim_mean'] < 0.3:
        print(f"\n✓ 防御有效! 重构质量很差 (SSIM < 0.3)")
    elif results['ssim_mean'] < 0.6:
        print(f"\n△ 防御部分有效 (0.3 <= SSIM < 0.6)")
    else:
        print(f"\n✗ 防御可能失效 (SSIM >= 0.6)")


if __name__ == '__main__':
    main()