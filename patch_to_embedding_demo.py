"""
详细演示：每个16x16 patch如何变成384维向量
解释卷积操作的内部机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class ConvolutionExplainer:
    """详细解释卷积如何将patch转换为嵌入向量"""

    def __init__(self, patch_size=16, in_channels=3, embed_dim=384):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 创建简单的卷积核来演示
        self.conv = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def explain_convolution_mechanism(self):
        """详细解释卷积机制"""
        print("=" * 80)
        print("🔬 卷积机制深度解析：patch → 向量")
        print("=" * 80)

        print(f"📋 基本参数:")
        print(f"  输入: {self.in_channels}通道的 {self.patch_size}x{self.patch_size} patch")
        print(f"  输出: {self.embed_dim}维向量")
        print(f"  卷积核形状: [{self.in_channels}, {self.patch_size}, {self.patch_size}]")
        print(f"  输出通道数: {self.embed_dim}")
        print()

    def demonstrate_single_patch_process(self):
        """演示单个patch的处理过程"""
        print("🎯 单个patch处理演示:")
        print("=" * 50)

        # 创建一个16x16的RGB patch
        patch = torch.randn(1, 3, 16, 16)
        print(f"📥 输入patch形状: {patch.shape}")
        print(f"   包含: 3 × 16 × 16 = {3*16*16} 个数值")
        print()

        # 模拟卷积过程
        print("🔄 卷积计算过程:")
        print("   每个输出通道 = 一个卷积核 × patch的加权和")
        print("   公式: output[i] = Σ(channel × kernel[i] × patch) + bias[i]")
        print()

        # 展示卷积核的结构
        print("🧮 卷积核结构:")
        print(f"   有 {self.embed_dim} 个独立的卷积核")
        print(f"   每个卷积核形状: [{self.in_channels}, {self.patch_size}, {self.patch_size}]")
        print(f"   每个卷积核参数数: {self.in_channels} × {self.patch_size} × {self.patch_size} = {self.in_channels * self.patch_size * self.patch_size}")
        print(f"   总参数数: {self.embed_dim} × {self.in_channels * self.patch_size * self.patch_size} = {self.embed_dim * self.in_channels * self.patch_size * self.patch_size:,}")
        print()

    def show_feature_maps_concept(self):
        """展示特征图的概念"""
        print("🗺️  特征图概念:")
        print("=" * 30)

        print("💡 每个输出通道学习不同的特征模式:")
        channels_examples = [
            "第1通道: 检测红色调模式",
            "第2通道: 检测垂直边缘",
            "第3通道: 检测纹理模式",
            "第4通道: 检测绿色调模式",
            "第5通道: 检测水平边缘",
            "...",
            f"第{self.embed_dim}通道: 检测某种复杂模式"
        ]

        for example in channels_examples:
            print(f"  {example}")
        print()

    def visualize_weight_matrix(self):
        """可视化权重矩阵的概念"""
        print("⚖️  权重矩阵可视化:")
        print("=" * 40)

        print("📊 卷积操作可以理解为矩阵乘法:")
        print("   输入: 768维向量 (3×16×16 展平的patch)")
        print("   权重: 384×768 矩阵 (embed_dim × input_dim)")
        print("   输出: 384维向量")
        print()

        # 创建简化的权重矩阵示例
        weight_matrix = torch.randn(384, 768)  # 假设的权重矩阵
        input_vector = torch.randn(768)       # 展平的patch

        print("🔄 计算过程:")
        print("   output[0] = w[0] · input_vector + b[0]")
        print("   output[1] = w[1] · input_vector + b[1]")
        print("   ...")
        print("   output[383] = w[383] · input_vector + b[383]")
        print()

        print("📈 维度关系:")
        print(f"   输入维度: {input_vector.shape[0]}")
        print(f"   权重矩阵: {weight_matrix.shape}")
        print(f"   输出维度: {weight_matrix.shape[0]}")
        print()

    def demonstrate_with_actual_numbers(self):
        """用实际数字演示计算"""
        print("🔢 实际计算演示:")
        print("=" * 30)

        # 创建简化的例子
        mini_patch = torch.ones(1, 3, 2, 2)  # 2x2 patch
        mini_embed_dim = 4
        mini_conv = nn.Conv2d(3, mini_embed_dim, 2, stride=2)

        # 设置可读的权重
        with torch.no_grad():
            mini_conv.weight[0] = torch.ones(3, 2, 2) * 0.1  # 第1个卷积核
            mini_conv.weight[1] = torch.ones(3, 2, 2) * 0.2  # 第2个卷积核
            mini_conv.weight[2] = torch.ones(3, 2, 2) * 0.3  # 第3个卷积核
            mini_conv.weight[3] = torch.ones(3, 2, 2) * 0.4  # 第4个卷积核
            mini_conv.bias.data.zero_()

        print(f"📥 简化输入patch: {mini_patch.shape} (所有值为1)")
        print(f"   展平后: {mini_patch.flatten().numpy()}")
        print()

        # 计算输出
        output = mini_conv(mini_patch)
        print(f"📤 输出嵌入: {output.shape}")
        print(f"   输出值: {output.flatten().numpy()}")
        print()

        # 手动计算验证
        input_sum = mini_patch.sum().item()  # 3×2×2 = 12
        print("✋ 手动计算验证:")
        print(f"   输入所有值相加: {input_sum}")
        print(f"   通道1: {input_sum} × 0.1 = {input_sum * 0.1}")
        print(f"   通道2: {input_sum} × 0.2 = {input_sum * 0.2}")
        print(f"   通道3: {input_sum} × 0.3 = {input_sum * 0.3}")
        print(f"   通道4: {input_sum} × 0.4 = {input_sum * 0.4}")
        print()

    def explain_feature_learning(self):
        """解释特征学习的过程"""
        print("🧠 特征学习机制:")
        print("=" * 40)

        print("🎯 训练过程中:")
        print("1. 初始: 卷积核权重随机初始化")
        print("2. 训练: 通过反向传播调整权重")
        print("3. 结果: 每个卷积核学会检测特定模式")
        print()

        print("🔍 学到的特征类型:")
        learned_features = [
            "低级特征 (早期卷积层):",
            "  - 边缘检测",
            "  - 颜色变化",
            "  - 简单纹理",
            "",
            "高级特征 (后期卷积层):",
            "  - 复杂模式",
            "  - 物体部件",
            "  - 语义信息"
        ]

        for feature in learned_features:
            print(f"  {feature}")
        print()

    def show_dimension_progression(self):
        """展示维度变化的全过程"""
        print("📊 完整维度变化:")
        print("=" * 30)

        steps = [
            ("原始图像", "[batch, 3, 224, 224]", "2张224×224的RGB图像"),
            ("单个patch", "[1, 3, 16, 16]", "一个16×16的RGB图像块"),
            ("展平patch", "[768]", "3×16×16 = 768个数值"),
            ("卷积权重", "[384, 768]", "384个768维的权重向量"),
            ("输出向量", "[384]", "384个加权和的结果"),
            ("完整输出", "[batch, 196, 384]", "196个patch，每个384维")
        ]

        for step, shape, description in steps:
            print(f"  {step:<12}: {shape:<20} - {description}")
        print()

    def demonstrate_cosine_similarity(self):
        """演示不同patch之间的相似性"""
        print("🔗 Patch相似性演示:")
        print("=" * 40)

        # 创建不同的patch
        red_patch = torch.ones(1, 3, 16, 16)
        red_patch[0, 0] *= 2   # 增强红色通道
        red_patch[0, 1:] *= 0.3 # 减弱绿色蓝色

        green_patch = torch.ones(1, 3, 16, 16)
        green_patch[0, 1] *= 2  # 增强绿色通道
        green_patch[0, [0, 2]] *= 0.3 # 减弱红色蓝色

        # 使用实际卷积处理
        conv = nn.Conv2d(3, 4, 16, stride=16)

        red_embedding = conv(red_patch).flatten()
        green_embedding = conv(green_patch).flatten()

        # 计算相似性
        similarity = F.cosine_similarity(red_embedding, green_embedding, dim=0)

        print(f"🟥 红色patch嵌入: {red_embedding.numpy()}")
        print(f"🟩 绿色patch嵌入: {green_embedding.numpy()}")
        print(f"📐 余弦相似度: {similarity.item():.4f}")
        print("   (值越小，表示差异越大)")
        print()

def main():
    """主函数：运行所有演示"""
    print("🎨 Patch → 嵌入向量 详解")
    print("每个16×16图像块如何变成384维特征向量")
    print()

    explainer = ConvolutionExplainer(patch_size=16, in_channels=3, embed_dim=384)

    # 运行所有演示
    explainer.explain_convolution_mechanism()
    explainer.demonstrate_single_patch_process()
    explainer.show_feature_maps_concept()
    explainer.visualize_weight_matrix()
    explainer.demonstrate_with_actual_numbers()
    explainer.explain_feature_learning()
    explainer.show_dimension_progression()
    explainer.demonstrate_cosine_similarity()

    print("=" * 80)
    print("💡 核心理解要点:")
    print("=" * 80)
    print("1. 每个patch → 384维向量是通过卷积实现的")
    print("2. 有384个独立的卷积核，每个负责一个维度")
    print("3. 每个卷积核: 3×16×16 = 768个参数")
    print("4. 总参数: 384 × 768 = 294,912")
    print("5. 训练过程中，每个卷积核学会检测特定模式")
    print("6. 不同patch会得到不同的384维向量表示")
    print("7. 相似的patch会产生相似的嵌入向量")
    print()
    print("🔑 关键公式:")
    print("   output_dim = ∑(input_dim × kernel_weight) + bias")
    print("   每个维度都是整个patch的加权和")

if __name__ == "__main__":
    main()