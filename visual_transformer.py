"""
Visual Transformer - 精简版Hiera架构
基于SAM2的Hiera Transformer思想，实现了一个简单的视觉Transformer模型

主要特点：
- 基于Patch Embedding的图像分块处理
- 窗口自注意力机制（Window-based Self-Attention）
- 多尺度特征融合
- 阶梯式下采样策略

作者：AI Assistant
日期：2025-12-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层 (Patch Embedding)

    将输入图像分割成固定大小的patch，然后将每个patch线性投影到embedding空间
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 384, in_channels: int = 3):
        """
        Args:
            img_size: 输入图像尺寸 (假设为正方形)
            patch_size: 每个patch的尺寸
            embed_dim: embedding维度
            in_channels: 输入通道数 (RGB=3)
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积进行patch embedding：一次处理所有patch
        # kernel_size=patch_size, stride=patch_size 确保不重叠的分割
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 可学习的位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # Xavier初始化卷积层
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        # 截断正态分布初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [batch_size, channels, height, width]
        Returns:
            patch embeddings [batch_size, num_patches, embed_dim]
        """
        B, C, H, W = x.shape

        # 确保输入尺寸正确
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"

        # 卷积操作将每个patch映射到embedding向量
        # [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size]
        x = self.proj(x)

        # 展平并转置为序列格式
        # [B, embed_dim, H//patch_size, W//patch_size] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # 添加位置编码
        x = x + self.pos_embed

        return x


class WindowAttention(nn.Module):
    """
    窗口多头自注意力机制 (Window-based Multi-Head Self-Attention)

    Hiera的核心思想：只在局部窗口内计算注意力，减少计算复杂度
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 6, window_size: int = 7):
        """
        Args:
            embed_dim: embedding维度
            num_heads: 注意力头数
            window_size: 窗口尺寸 (假设为正方形)
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # Q, K, V线性变换
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, num_patches, embed_dim]
            H, W: 特征图的高度和宽度 (用于重构窗口)
        Returns:
            输出特征 [batch_size, num_patches, embed_dim]
        """
        B, N, C = x.shape

        # 计算Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # 重构为2D格式用于窗口注意力
        q = q.view(B, self.num_heads, H, W, self.head_dim)
        k = k.view(B, self.num_heads, H, W, self.head_dim)
        v = v.view(B, self.num_heads, H, W, self.head_dim)

        # 简化的窗口注意力：使用固定大小的非重叠窗口
        window_size = self.window_size
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        # 填充以适应窗口
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, 0, 0, pad_w, 0, pad_h))
            k = F.pad(k, (0, 0, 0, pad_w, 0, pad_h))
            v = F.pad(v, (0, 0, 0, pad_w, 0, pad_h))

        H_padded, W_padded = H + pad_h, W + pad_w

        # 重塑为窗口格式
        q_windows = q.view(B, self.num_heads,
                          H_padded // window_size, window_size,
                          W_padded // window_size, window_size, self.head_dim)
        k_windows = k.view(B, self.num_heads,
                          H_padded // window_size, window_size,
                          W_padded // window_size, window_size, self.head_dim)
        v_windows = v.view(B, self.num_heads,
                          H_padded // window_size, window_size,
                          W_padded // window_size, window_size, self.head_dim)

        # 转置以进行批量窗口计算
        q_windows = q_windows.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # [B, num_heads, H_w, W_w, window_h, window_w, head_dim]
        k_windows = k_windows.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        v_windows = v_windows.permute(0, 1, 2, 4, 3, 5, 6).contiguous()

        # 展平为窗口序列
        num_windows = (H_padded // window_size) * (W_padded // window_size)
        q_windows = q_windows.view(B * self.num_heads, num_windows, window_size * window_size, self.head_dim)
        k_windows = k_windows.view(B * self.num_heads, num_windows, window_size * window_size, self.head_dim)
        v_windows = v_windows.view(B * self.num_heads, num_windows, window_size * window_size, self.head_dim)

        # 计算窗口内注意力
        attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale  # [B*heads, num_windows, window_size^2, window_size^2]
        attn = F.softmax(attn, dim=-1)

        # 应用注意力权重
        window_output = attn @ v_windows  # [B*heads, num_windows, window_size^2, head_dim]

        # 重塑回原始形状
        window_output = window_output.view(B, self.num_heads,
                                          H_padded // window_size, W_padded // window_size,
                                          window_size, window_size, self.head_dim)

        # 转置回去并重组为完整特征图
        window_output = window_output.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        window_output = window_output.view(B, self.num_heads, H_padded, W_padded, self.head_dim)

        # 移除填充
        if pad_h > 0 or pad_w > 0:
            window_output = window_output[:, :, :H, :W, :]

        # 重塑为序列格式
        window_output = window_output.contiguous().view(B, self.num_heads, H * W, self.head_dim).transpose(1, 2)
        window_output = window_output.contiguous().view(B, N, C)

        # 最终线性变换
        output = self.proj(window_output)

        return output


class MLP(nn.Module):
    """
    多层感知机 (MLP)
    前馈网络，包含两个线性层和GELU激活
    """

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        Args:
            embed_dim: 输入维度
            mlp_ratio: MLP扩展比例 (隐藏层维度 = embed_dim * mlp_ratio)
            dropout: Dropout概率
        """
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, num_patches, embed_dim]
        Returns:
            输出特征 [batch_size, num_patches, embed_dim]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class HieraBlock(nn.Module):
    """
    Hiera Transformer块

    包含窗口注意力和MLP，采用Pre-Normalization结构
    """

    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 7,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        Args:
            embed_dim: embedding维度
            num_heads: 注意力头数
            window_size: 注意力窗口大小
            mlp_ratio: MLP扩展比例
            dropout: Dropout概率
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(embed_dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, num_patches, embed_dim]
            H, W: 特征图空间维度
        Returns:
            输出特征 [batch_size, num_patches, embed_dim]
        """
        # Pre-Norm: 注意力 + 残差连接
        x = x + self.attn(self.norm1(x), H, W)

        # Pre-Norm: MLP + 残差连接
        x = x + self.mlp(self.norm2(x))

        return x


class PatchMerging(nn.Module):
    """
    Patch合并层 (Patch Merging)

    Hiera的多尺度特征融合：将2x2的patch合并为一个，实现下采样
    """

    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: 输入embedding维度
        """
        super().__init__()
        self.reduction = nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)
        self.norm = nn.LayerNorm(4 * embed_dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: 输入特征 [batch_size, H*W, embed_dim]
            H, W: 特征图高度和宽度
        Returns:
            output: 下采样后的特征 [batch_size, (H//2)*(W//2), 2*embed_dim]
            new_H, new_W: 新的特征图尺寸
        """
        B, L, C = x.shape
        assert L == H * W, "Input features length doesn't match H*W"

        # 重塑为2D格式
        x = x.view(B, H, W, C)

        # 确保H和W都是偶数
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H, W = H + H % 2, W + W % 2

        # 2x2相邻patch合并
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right

        # 拼接4个patch
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H//2, W//2, 4*C]

        # 展平并归一化
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)

        # 线性变换降维
        x = self.reduction(x)  # [B, (H//2)*(W//2), 2*C]

        return x, H // 2, W // 2


class VisualTransformer(nn.Module):
    """
    视觉Transformer主模型

    基于Hiera架构的视觉Transformer，用于图像分类任务
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: list = [96, 192, 384, 768],
        depths: list = [2, 3, 6, 3],
        num_heads: list = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 1000
    ):
        """
        Args:
            img_size: 输入图像尺寸
            patch_size: 初始patch尺寸
            in_channels: 输入通道数
            embed_dims: 各阶段的embedding维度
            depths: 各阶段的Transformer块数量
            num_heads: 各阶段的注意力头数
            window_size: 注意力窗口大小
            mlp_ratio: MLP扩展比例
            dropout: Dropout概率
            num_classes: 分类数量
        """
        super().__init__()

        self.num_stages = len(embed_dims)
        self.embed_dims = embed_dims
        self.num_classes = num_classes

        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dims[0], in_channels)

        # 各阶段的Transformer块和Patch Merging
        self.stages = nn.ModuleList()
        self.patch_mergings = nn.ModuleList()

        for i in range(self.num_stages):
            stage_blocks = nn.ModuleList([
                HieraBlock(
                    embed_dim=embed_dims[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(depths[i])
            ])
            self.stages.append(stage_blocks)

            # 除了最后阶段，其他阶段都需要Patch Merging
            if i < self.num_stages - 1:
                self.patch_mergings.append(PatchMerging(embed_dims[i]))

        # 分类头
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        """初始化分类头权重"""
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征提取前向传播

        Args:
            x: 输入图像 [batch_size, channels, height, width]
        Returns:
            特征向量 [batch_size, final_embed_dim]
        """
        # Patch Embedding
        x = self.patch_embed(x)

        # 初始特征图尺寸
        H, W = int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5)

        # 逐阶段处理
        for i, stage_blocks in enumerate(self.stages):
            # 通过当前阶段的所有Transformer块
            for block in stage_blocks:
                x = block(x, H, W)

            # Patch Merging (除了最后阶段)
            if i < len(self.patch_mergings):
                x, H, W = self.patch_mergings[i](x, H, W)

        # 最终归一化
        x = self.norm(x)

        # 全局平均池化
        x = x.mean(dim=1)  # [batch_size, embed_dim]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整前向传播

        Args:
            x: 输入图像 [batch_size, channels, height, width]
        Returns:
            分类logits [batch_size, num_classes]
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x


def test_visual_transformer():
    """
    测试Visual Transformer模型
    """
    print("测试Visual Transformer模型...")

    # 创建模型
    model = VisualTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dims=[96, 192, 384],
        depths=[2, 3, 6],
        num_heads=[3, 6, 12],
        window_size=7,
        num_classes=1000
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    print(f"输入形状: {x.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(x)

    print(f"输出形状: {output.shape}")
    print("✓ 模型测试通过!")

    return model


if __name__ == "__main__":
    # 运行测试
    model = test_visual_transformer()

    # 简单使用示例
    print("\n使用示例:")
    print("```python")
    print("# 创建模型")
    print("model = VisualTransformer(")
    print("    img_size=224,")
    print("    patch_size=16,")
    print("    embed_dims=[96, 192, 384],")
    print("    depths=[2, 3, 6],")
    print("    num_heads=[3, 6, 12],")
    print("    num_classes=1000")
    print(")")
    print("")
    print("# 预测")
    print("x = torch.randn(1, 3, 224, 224)  # 输入图像")
    print("with torch.no_grad():")
    print("    logits = model(x)")
    print("    probs = F.softmax(logits, dim=-1)")
    print("    pred_class = probs.argmax(dim=-1)")
    print("```")