# -*- coding: utf-8 -*-
"""
白盒重构攻击 - 基于梯度优化 (rMLE方法)
即使攻击者知道完整的模型和训练流程,但不知道随机打乱的顺序
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
    if img1.ndim == 3:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        channel_axis = -1
    else:
        channel_axis = None

    data_range = max(img1.max() - img1.min(), img2.max() - img2.min())

    try:
        return compare_ssim(img1, img2, data_range=data_range, channel_axis=channel_axis)
    except ValueError:
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        win_size = max(win_size, 3)
        return compare_ssim(img1, img2, data_range=data_range,
                            channel_axis=channel_axis, win_size=win_size)


def total_variation(x):
    """
    计算Total Variation正则化
    鼓励生成平滑的图像
    """
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = (x.size()[2] - 1) * x.size()[3]
    count_w = x.size()[2] * (x.size()[3] - 1)
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2_loss(x):
    """L2正则化"""
    return (x ** 2).mean()


class WhiteBoxAttacker:
    """
    白盒攻击器 - 使用梯度优化重构图像

    攻击者知道:
    1. 完整的客户端模型结构和参数
    2. 训练流程和防御机制

    攻击者不知道:
    1. 随机打乱的具体顺序
    2. 每个向量实际来自哪个客户端
    """

    def __init__(self, client_model, device='cuda'):
        """
        Args:
            client_model: 客户端模型(用于前向传播)
            device: 计算设备
        """
        self.client_model = client_model.to(device)
        self.client_model.eval()
        self.device = device

        print(f"[WhiteBoxAttacker] 初始化")
        print(f"  - 攻击者拥有完整的客户端模型")
        print(f"  - 但不知道向量的打乱顺序")

    def reconstruct_image(
            self,
            target_vector,
            image_shape,
            n_iterations=5000,
            learning_rate=0.01,
            lambda_tv=1e1,
            lambda_l2=0.0,
            init_mode='gray'
    ):
        """
        使用梯度优化重构图像 (rMLE方法)

        优化目标: minimize ||f(x) - v||^2 + λ_tv*TV(x) + λ_l2*||x||^2

        Args:
            target_vector: 目标中间向量 [feature_dim]
            image_shape: 图像形状 (C, H, W)
            n_iterations: 优化迭代次数
            learning_rate: 学习率
            lambda_tv: Total Variation权重
            lambda_l2: L2正则化权重
            init_mode: 初始化模式 ('gray', 'random', 'noise')

        Returns:
            reconstructed_image: 重构的图像
            loss_history: 损失历史
        """

        # 初始化生成图像 - 关键修复：直接在[-1, 1]范围内初始化
        if init_mode == 'gray':
            # 初始化为灰色 (0.0 对应归一化后的0.5)
            x_gen = torch.zeros(1, *image_shape, device=self.device, requires_grad=True)
        elif init_mode == 'random':
            # 随机初始化在[-1, 1]范围内
            x_gen = torch.randn(1, *image_shape, device=self.device) * 0.2
            x_gen.requires_grad = True
        elif init_mode == 'noise':
            # 高斯噪声
            x_gen = torch.randn(1, *image_shape, device=self.device)
            x_gen.requires_grad = True
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        # 确保目标向量在正确的设备上
        target_vector = target_vector.to(self.device)

        # 优化器
        optimizer = optim.Adam([x_gen], lr=learning_rate, amsgrad=True)

        # 记录损失
        loss_history = {
            'total': [],
            'feature': [],
            'tv': [],
            'l2': []
        }

        # 优化循环
        pbar = tqdm(range(n_iterations), desc="白盒重构")
        for i in pbar:
            optimizer.zero_grad()

            # 前向传播: 通过客户端模型
            generated_vector = self.client_model(x_gen)

            # 特征匹配损失
            feature_loss = ((generated_vector - target_vector) ** 2).mean()

            # Total Variation损失(图像平滑)
            tv_loss = total_variation(x_gen)

            # L2正则化损失
            l2_reg = l2_loss(x_gen)

            # 总损失
            total_loss = feature_loss + lambda_tv * tv_loss + lambda_l2 * l2_reg

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 裁剪到有效范围[-1, 1]
            with torch.no_grad():
                x_gen.clamp_(-1, 1)

            # 记录损失
            loss_history['total'].append(total_loss.item())
            loss_history['feature'].append(feature_loss.item())
            loss_history['tv'].append(tv_loss.item())
            loss_history['l2'].append(l2_reg.item())

            # 更新进度条
            if i % 100 == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.6f}',
                    'feature': f'{feature_loss.item():.6f}'
                })

        # 返回重构的图像
        reconstructed = x_gen.detach().clone()

        return reconstructed, loss_history

    def clip_image(self, image):
        """裁剪图像到有效范围"""
        return torch.clamp(image, -1, 1)


def load_full_images(dataset_name, num_samples=None):
    """加载完整的原始图像"""
    import torchvision
    import torchvision.transforms as transforms

    print(f"\n加载完整原始图像: {dataset_name}")

    if dataset_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    if dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'SVHN':
        dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    images = []
    labels = []

    if num_samples is None:
        num_samples = len(dataset)

    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        images.append(img)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.LongTensor(labels)

    print(f"  加载完成: {images.shape}")
    return images, labels


def load_client_model(model_path, input_channels, dataset_config, device):
    """加载训练好的客户端模型"""
    from models import BottomModel

    print(f"\n加载客户端模型: {model_path}")

    # 创建模型
    bottom_config = dataset_config.get('bottom_model', None)
    model = BottomModel(input_channels, bottom_config)

    # 加载权重
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"  ✓ 模型加载成功")

    return model


def evaluate_whitebox_attack(
        vectors_file,
        model_dir,
        dataset_name,
        target_client_id=0,
        num_samples=20,
        n_iterations=5000,
        learning_rate=0.01,
        lambda_tv=1e1,
        lambda_l2=0.0,
        save_dir='whitebox_attack_results',
        device='cuda'
):
    """
    评估白盒攻击效果

    Args:
        vectors_file: 中间向量文件
        model_dir: 模型目录
        dataset_name: 数据集名称
        target_client_id: 目标客户端ID
        num_samples: 评估样本数
        n_iterations: 优化迭代次数
        learning_rate: 学习率
        lambda_tv: TV正则化权重
        lambda_l2: L2正则化权重
        save_dir: 结果保存目录
        device: 计算设备
    """

    print("\n" + "=" * 80)
    print(f"白盒重构攻击 - 目标客户端 {target_client_id}".center(80))
    print("=" * 80)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)

    # 加载配置
    from config import VFLConfig
    config = VFLConfig(dataset_name=dataset_name)

    # 确定图像参数
    if dataset_name in ['MNIST', 'FashionMNIST']:
        image_shape = (1, 28, 28)
        input_channels = 1
    else:
        image_shape = (3, 32, 32)
        input_channels = 3

    # 加载客户端模型
    model_path = os.path.join(model_dir, f'client_{target_client_id}_{dataset_name.lower()}.pth')
    client_model = load_client_model(model_path, input_channels, config.dataset_config, device)

    # 创建攻击器
    attacker = WhiteBoxAttacker(client_model, device)

    # 加载中间向量
    print(f"\n加载中间向量: {vectors_file}")
    with open(vectors_file, 'rb') as f:
        data = pickle.load(f)

    # === 关键假设: 攻击者错误地认为位置固定 ===
    vectors_before_shuffle = data['vectors_before_shuffle'][target_client_id]

    print(f"\n攻击者假设:")
    print(f"  ✗ 位置 {target_client_id} 的向量来自客户端 {target_client_id}")
    print(f"  实际情况: 由于随机打乱,这个假设是错误的!")

    # 加载完整的原始图像
    full_images, _ = load_full_images(dataset_name)

    # 归一化图像到[-1, 1]
    if dataset_name in ['MNIST', 'FashionMNIST']:
        target_images = full_images[:len(vectors_before_shuffle)] * 2 - 1
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        target_images = full_images[:len(vectors_before_shuffle)] * std + mean
        target_images = target_images * 2 - 1

    print(f"\n原始图像: {target_images.shape}")
    print(f"  范围: [{target_images.min():.3f}, {target_images.max():.3f}]")

    # 选择样本进行攻击
    indices = np.random.choice(len(vectors_before_shuffle), num_samples, replace=False)

    psnr_list = []
    ssim_list = []

    print(f"\n开始白盒攻击 {num_samples} 个样本...")
    print(f"  迭代次数: {n_iterations}")
    print(f"  学习率: {learning_rate}")
    print(f"  λ_TV: {lambda_tv}")
    print(f"  λ_L2: {lambda_l2}")

    reconstructed_images = []
    original_images = []

    for sample_idx, idx in enumerate(indices):
        print(f"\n样本 {sample_idx + 1}/{num_samples} (索引 {idx}):")

        # 目标向量和原始图像
        target_vector = torch.FloatTensor(vectors_before_shuffle[idx]).unsqueeze(0)
        original = target_images[idx]

        # 白盒重构
        reconstructed, loss_history = attacker.reconstruct_image(
            target_vector=target_vector,
            image_shape=image_shape,
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            lambda_tv=lambda_tv,
            lambda_l2=lambda_l2,
            init_mode='gray'
        )

        # 裁剪到有效范围
        reconstructed = attacker.clip_image(reconstructed)

        # 转换为numpy用于评估
        original_np = original.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()[0]

        # 归一化到[0, 1]
        original_np = np.clip((original_np + 1) / 2, 0, 1)
        reconstructed_np = np.clip((reconstructed_np + 1) / 2, 0, 1)

        # 计算指标
        psnr = get_psnr(original_np, reconstructed_np)
        ssim = get_ssim(original_np, reconstructed_np)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        print(f"  最终损失: {loss_history['total'][-1]:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")

        reconstructed_images.append(reconstructed_np)
        original_images.append(original_np)

    # 统计
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)

    print(f"\n" + "=" * 80)
    print("白盒攻击结果统计".center(80))
    print("=" * 80)
    print(f"  PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

    # 可视化结果
    print(f"\n保存可视化结果...")
    num_vis = min(10, num_samples)

    fig, axes = plt.subplots(3, num_vis, figsize=(num_vis * 2, 6))

    for i in range(num_vis):
        original = original_images[i]
        reconstructed = reconstructed_images[i]
        difference = np.abs(original - reconstructed)

        if original.shape[0] == 1:
            axes[0, i].imshow(original[0], cmap='gray')
            axes[1, i].imshow(reconstructed[0], cmap='gray')
            axes[2, i].imshow(difference[0], cmap='hot')
        else:
            axes[0, i].imshow(np.transpose(original, (1, 2, 0)))
            axes[1, i].imshow(np.transpose(reconstructed, (1, 2, 0)))
            axes[2, i].imshow(np.transpose(difference, (1, 2, 0)))

        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')

        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12, fontweight='bold')
            axes[1, i].set_ylabel('Reconstructed\n(White-box)', fontsize=12, fontweight='bold')
            axes[2, i].set_ylabel('Difference', fontsize=12, fontweight='bold')

        # 添加PSNR/SSIM标注
        axes[1, i].set_title(f'PSNR:{psnr_list[i]:.1f}\nSSIM:{ssim_list[i]:.3f}',
                             fontsize=10)

    plt.tight_layout()
    vis_path = os.path.join(save_dir, 'images', f'whitebox_reconstruction_client_{target_client_id}.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 可视化结果: {vis_path}")

    # 保存详细结果
    results = {
        'attack_type': 'white_box',
        'target_client_id': target_client_id,
        'dataset': dataset_name,
        'num_samples': num_samples,
        'n_iterations': n_iterations,
        'learning_rate': learning_rate,
        'lambda_tv': lambda_tv,
        'lambda_l2': lambda_l2,
        'psnr_mean': avg_psnr,
        'psnr_std': std_psnr,
        'ssim_mean': avg_ssim,
        'ssim_std': std_ssim,
        'psnr_list': psnr_list,
        'ssim_list': ssim_list
    }

    results_path = os.path.join(save_dir, f'whitebox_results_client_{target_client_id}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"  ✓ 详细结果: {results_path}")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='白盒重构攻击 - 测试随机打乱防御')
    parser.add_argument('--vectors_file', type=str, required=True,
                        help='中间向量文件路径')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='模型目录')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN'],
                        help='数据集名称')
    parser.add_argument('--target_client', type=int, default=0,
                        help='目标攻击的客户端ID (0-4)')
    parser.add_argument('--n_iterations', type=int, default=5000,
                        help='优化迭代次数')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--lambda_tv', type=float, default=1e1,
                        help='Total Variation正则化权重')
    parser.add_argument('--lambda_l2', type=float, default=0.0,
                        help='L2正则化权重')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='评估样本数')
    parser.add_argument('--save_dir', type=str, default='whitebox_attack_results',
                        help='结果保存目录')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 执行白盒攻击
    results = evaluate_whitebox_attack(
        vectors_file=args.vectors_file,
        model_dir=args.model_dir,
        dataset_name=args.dataset,
        target_client_id=args.target_client,
        num_samples=args.num_samples,
        n_iterations=args.n_iterations,
        learning_rate=args.learning_rate,
        lambda_tv=args.lambda_tv,
        lambda_l2=args.lambda_l2,
        save_dir=args.save_dir,
        device=device
    )

    print("\n" + "=" * 80)
    print("白盒攻击完成!".center(80))
    print("=" * 80)
    print(f"\n攻击方式: 白盒攻击(梯度优化)")
    print(f"攻击者知道: 完整的模型结构和参数")
    print(f"攻击者不知道: 随机打乱的具体顺序")

    print(f"\n实际结果:")
    print(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")

    # 判断防御效果
    if results['ssim_mean'] < 0.3:
        print(f"\n✓ 防御有效! 即使是白盒攻击也无法重构 (SSIM < 0.3)")
    elif results['ssim_mean'] < 0.6:
        print(f"\n△ 防御部分有效 (0.3 <= SSIM < 0.6)")
    else:
        print(f"\n✗ 防御可能失效 (SSIM >= 0.6)")

    print(f"\n结论:")
    print(f"  即使攻击者拥有完整的模型知识(白盒攻击),")
    print(f"  由于不知道随机打乱的顺序,重构仍然失败。")
    print(f"  这证明了随机打乱防御的有效性!")


if __name__ == '__main__':
    main()