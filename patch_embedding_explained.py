"""
Patch Embedding 详细解释：图像分块和嵌入生成的完整过程
重点解释维度变化的每一步
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class DetailedPatchEmbedding(nn.Module):
    """
    带详细调试信息的Patch Embedding类
    每一步都打印维度变化，便于理解过程
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 384, in_channels: int = 3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        print("🔧 Patch Embedding 初始化:")
        print(f"  输入图像尺寸: {img_size} x {img_size}")
        print(f"  每个Patch尺寸: {patch_size} x {patch_size}")
        print(f"  Patch数量: {img_size // patch_size} x {img_size // patch_size} = {self.num_patches}")
        print(f"  嵌入维度: {embed_dim}")
        print(f"  输入通道数: {in_channels}")
        print()

        # 核心卷积层：负责将每个patch映射到embedding向量
        self.proj = nn.Conv2d(
            in_channels,    # 输入通道数：RGB=3
            embed_dim,      # 输出通道数：每个patch映射到embed_dim维向量
            kernel_size=patch_size,  # 卷积核大小=patch_size
            stride=patch_size        # 步长=patch_size，确保不重叠
        )

        print("🎯 卷积层配置:")
        print(f"  卷积核: {in_channels} -> {embed_dim}")
        print(f"  核大小: {patch_size} x {patch_size}")
        print(f"  步长: {patch_size} x {patch_size}")
        print(f"  参数数量: {embed_dim * in_channels * patch_size * patch_size:,}")
        print()

        # 可学习的位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        print("📍 位置编码:")
        print(f"  形状: [1, {self.num_patches}, {embed_dim}]")
        print(f"  可学习参数: {self.num_patches * embed_dim:,}")
        print()

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, debug: bool = True) -> torch.Tensor:
        """
        详细的前向传播过程，打印每一步的维度变化

        Args:
            x: 输入图像 [batch_size, channels, height, width]
            debug: 是否打印调试信息
        Returns:
            patch embeddings [batch_size, num_patches, embed_dim]
        """
        if debug:
            print("=" * 60)
            print("🚀 Patch Embedding 前向传播过程")
            print("=" * 60)

        # 步骤1: 输入检查
        B, C, H, W = x.shape
        if debug:
            print(f"📥 输入维度:")
            print(f"  批次大小: {B}")
            print(f"  通道数: {C}")
            print(f"  高度: {H}")
            print(f"  宽度: {W}")
            print(f"  完整形状: {x.shape}")
            print()

        # 步骤2: 卷积操作 - 最重要的维度变化
        if debug:
            print("🔄 步骤2: 卷积操作 (核心步骤)")
            print(f"  卷积核滑动: 每个patch_size x patch_size区域 -> 一个{self.embed_dim}维向量")
            print(f"  输出特征图尺寸: {H // self.patch_size} x {W // self.patch_size}")

        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]

        if debug:
            print(f"  卷积后形状: {x.shape}")
            print(f"  维度变化: [{B}, {C}, {H}, {W}] -> [{B}, {self.embed_dim}, {H//self.patch_size}, {W//self.patch_size}]")
            print()

        # 步骤3: 展平操作
        if debug:
            print("📐 步骤3: 展平操作")
            print(f"  将最后两个维度合并: (H//patch_size) * (W//patch_size)")
            print(f"  {H//self.patch_size} x {W//self.patch_size} = {(H//self.patch_size) * (W//self.patch_size)}")

        x = x.flatten(2)  # [B, embed_dim, (H//patch_size)*(W//patch_size)]

        if debug:
            print(f"  展平后形状: {x.shape}")
            print(f"  维度变化: [{B}, {self.embed_dim}, {H//self.patch_size}, {W//self.patch_size}] -> [{B}, {self.embed_dim}, {(H//self.patch_size)*(W//self.patch_size)}]")
            print()

        # 步骤4: 转置为Transformer需要的格式
        if debug:
            print("🔄 步骤4: 转置维度")
            print(f"  将patch维度移到前面，embed_dim移到最后")
            print(f"  为Transformer准备: [batch, patches, features]")

        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]

        if debug:
            print(f"  转置后形状: {x.shape}")
            print(f"  维度变化: [{B}, {self.embed_dim}, {(H//self.patch_size)*(W//self.patch_size)}] -> [{B}, {(H//self.patch_size)*(W//self.patch_size)}, {self.embed_dim}]")
            print()

        # 步骤5: 添加位置编码
        if debug:
            print("📍 步骤5: 添加位置编码")
            print(f"  位置编码形状: {self.pos_embed.shape}")
            print(f"  广播机制: [1, {self.num_patches}, {self.embed_dim}] -> [{B}, {self.num_patches}, {self.embed_dim}]")

        x = x + self.pos_embed  # [B, num_patches, embed_dim]

        if debug:
            print(f"  最终输出形状: {x.shape}")
            print(f"  每个patch: {self.embed_dim}维向量，包含位置信息")
            print()

        return x

def demonstrate_patch_embedding():
    """演示Patch Embedding的完整过程"""
    print("=" * 60)
    print("🎨 Patch Embedding 完整演示")
    print("=" * 60)
    print()

    # 创建实例
    patch_embed = DetailedPatchEmbedding(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        in_channels=3
    )

    # 创建示例输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    print("📸 创建示例输入:")
    print(f"  输入图像: {x.shape}")
    print(f"  随机数值范围: [{x.min():.3f}, {x.max():.3f}]")
    print()

    # 执行前向传播
    embeddings = patch_embed(x, debug=True)

    print("✅ Patch Embedding 完成!")
    print(f"最终输出: {embeddings.shape}")

def visualize_patch_process():
    """可视化patch分割过程"""
    print("\n" + "=" * 60)
    print("🔍 Patch分割过程可视化")
    print("=" * 60)

    img_size, patch_size = 224, 16
    num_patches_per_dim = img_size // patch_size

    print(f"输入图像: {img_size} x {img_size}")
    print(f"每个Patch: {patch_size} x {patch_size}")
    print(f"每行Patch数量: {num_patches_per_dim}")
    print(f"每列Patch数量: {num_patches_per_dim}")
    print(f"总Patch数量: {num_patches_per_dim} x {num_patches_per_dim} = {num_patches_per_dim**2}")

    print("\n📐 Patch编号方案 (从左到右，从上到下):")
    for row in range(min(4, num_patches_per_dim)):  # 只显示前4行
        row_patches = []
        for col in range(min(4, num_patches_per_dim)):  # 只显示前4列
            patch_idx = row * num_patches_per_dim + col
            row_patches.append(f"{patch_idx:3d}")
        print(f"行{row:02d}: {row_patches}")

    print("\n🎯 每个Patch包含的像素:")
    print(f"  Patch 0:  行[0:16], 列[0:16]      (左上角)")
    print(f"  Patch 1:  行[0:16], 列[16:32]     (右上方向)")
    print(f"  Patch 16: 行[16:32], 列[0:16]     (下一行)")
    print(f"  Patch 195: 行[208:224], 列[208:224] (右下角)")

def explain_convolution_details():
    """详细解释卷积操作的原理"""
    print("\n" + "=" * 60)
    print("🧮 卷积操作详细解释")
    print("=" * 60)

    print("📝 卷积核参数:")
    print("  输入通道: 3 (RGB)")
    print("  输出通道: 384 (embedding维度)")
    print("  核大小: 16 x 16")
    print("  步长: 16 x 16")
    print()

    print("🔄 卷积过程:")
    print("  1. 卷积核在每个16x16区域滑动")
    print("  2. 每个卷积核包含: 3 x 16 x 16 = 768个权重")
    print("  3. 输出通道有384个，所以总参数: 384 x 768 = 294,912")
    print("  4. 每个输出通道学习不同的特征模式")
    print()

    print("💡 与手动分割的对比:")
    print("  手动方法: reshape -> 展平 -> 线性变换")
    print("  卷积方法: 一步到位，更高效")
    print("  优势: 代码简洁，计算优化，支持任意输入尺寸")

def dimension_change_summary():
    """维度变化总结"""
    print("\n" + "=" * 60)
    print("📊 维度变化总结")
    print("=" * 60)

    print("🔄 完整的维度变换链:")
    print()

    print("📥 输入图像:")
    print("  形状: [batch_size, channels, height, width]")
    print("  示例: [2, 3, 224, 224]")
    print()

    print("🔄 卷积操作:")
    print("  变化: [2, 3, 224, 224] -> [2, 384, 14, 14]")
    print("  解释: 每个16x16 patch → 384维特征向量")
    print()

    print("📐 展平操作:")
    print("  变化: [2, 384, 14, 14] -> [2, 384, 196]")
    print("  解释: 14x14 = 196个patch，展平为序列")
    print()

    print("🔄 转置操作:")
    print("  变化: [2, 384, 196] -> [2, 196, 384]")
    print("  解释: Transformer需要的格式 [batch, sequence, features]")
    print()

    print("📍 添加位置编码:")
    print("  变化: [2, 196, 384] + [1, 196, 384] -> [2, 196, 384]")
    print("  解释: 广播机制，每个位置获得唯一编码")
    print()

    print("🎯 最终输出:")
    print("  形状: [batch_size, num_patches, embed_dim]")
    print("  含义: 每个patch是一个384维的嵌入向量")
    print("  用途: 输入给Transformer进行注意力计算")

def memory_analysis():
    """内存使用分析"""
    print("\n" + "=" * 60)
    print("💾 内存使用分析")
    print("=" * 60)

    batch_size = 2
    img_size = 224
    patch_size = 16
    embed_dim = 384

    # 计算内存
    input_memory = batch_size * 3 * img_size * img_size * 4  # float32
    output_memory = batch_size * (img_size // patch_size) ** 2 * embed_dim * 4
    param_memory = 3 * embed_dim * patch_size * patch_size * 4 + (img_size // patch_size) ** 2 * embed_dim * 4

    print(f"📊 内存占用 (batch_size={batch_size}):")
    print(f"  输入图像: {input_memory / 1024 / 1024:.2f} MB")
    print(f"  输出嵌入: {output_memory / 1024 / 1024:.2f} MB")
    print(f"  模型参数: {param_memory / 1024 / 1024:.2f} MB")
    print(f"  总计: {(input_memory + output_memory + param_memory) / 1024 / 1024:.2f} MB")
    print()

    print("🔍 不同batch_size的内存需求:")
    for bs in [1, 4, 8, 16, 32]:
        total = (bs * 3 * img_size * img_size +
                bs * (img_size // patch_size) ** 2 * embed_dim +
                3 * embed_dim * patch_size * patch_size +
                (img_size // patch_size) ** 2 * embed_dim) * 4 / 1024 / 1024
        print(f"  batch_size={bs:2d}: {total:.1f} MB")

if __name__ == "__main__":
    print("🎨 Patch Embedding 深度解析")
    print("详细解释图像如何分割为patch并转换为嵌入向量")
    print()

    # 运行所有演示
    demonstrate_patch_embedding()
    visualize_patch_process()
    explain_convolution_details()
    dimension_change_summary()
    memory_analysis()

    print("\n" + "=" * 60)
    print("✨ 总结")
    print("=" * 60)
    print("Patch Embedding是视觉Transformer的核心步骤:")
    print("1. 将2D图像分割为固定大小的patches")
    print("2. 将每个patch转换为高维embedding向量")
    print("3. 添加位置编码保留空间信息")
    print("4. 输出为Transformer可以处理的序列格式")
    print()
    print("关键优势:")
    print("- 保持空间信息的结构化表示")
    print("- 支持高效的并行计算")
    print("- 可学习的位置编码适应不同任务")
    print("- 与自然语言处理中的tokenization概念统一")