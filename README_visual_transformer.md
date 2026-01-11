# Visual Transformer - 精简版Hiera架构

基于SAM2的Hiera Transformer思想实现的精简版视觉Transformer，专为理解和学习计算机视觉中的注意力机制而设计。

## 🌟 特点

- **基于Hiera架构**：采用SAM2中的多尺度特征融合思想
- **窗口自注意力**：高效的局部注意力机制，降低计算复杂度
- **多尺度设计**：阶梯式下采样，捕获不同尺度的特征
- **模块化实现**：清晰的组件分离，易于理解和修改
- **详细注释**：每个组件都有清晰的中文注释说明

## 📋 核心组件

### 1. PatchEmbedding (图像分块嵌入)
- 将输入图像分割为固定大小的patch
- 通过卷积层将patch投影到embedding空间
- 添加可学习的位置编码

### 2. WindowAttention (窗口自注意力)
- 在局部窗口内计算注意力，避免全局注意力的平方复杂度
- 支持非重叠窗口的高效计算
- 多头注意力机制

### 3. HieraBlock (Hiera Transformer块)
- 包含窗口注意力和MLP前馈网络
- 采用Pre-Normalization结构
- 残差连接确保梯度流动

### 4. PatchMerging (特征合并)
- 实现2x2 patch合并，完成下采样
- 通道维度翻倍，空间维度减半
- 多尺度特征融合

## 🏗️ 模型架构

```
输入图像 [B, 3, 224, 224]
    ↓
Patch Embedding
    ↓ [B, 196, 96]  # 14x14 patches
Stage 1 (2层，96维，3头注意力)
    ↓
Patch Merging
    ↓ [B, 49, 192]  # 7x7 patches
Stage 2 (3层，192维，6头注意力)
    ↓
Patch Merging
    ↓ [B, 12, 384]  # 3x4 patches (padding后4x4)
Stage 3 (6层，384维，12头注意力)
    ↓
全局平均池化 + 分类头
    ↓ [B, num_classes]
```

## 🚀 快速开始

### 基本使用

```python
from visual_transformer import VisualTransformer
import torch

# 创建模型
model = VisualTransformer(
    img_size=224,           # 输入图像尺寸
    patch_size=16,          # patch大小
    embed_dims=[96, 192, 384],  # 各阶段维度
    depths=[2, 3, 6],       # 各阶段层数
    num_heads=[3, 6, 12],   # 各阶段注意力头数
    window_size=7,          # 注意力窗口大小
    num_classes=1000        # 分类数
)

# 前向传播
x = torch.randn(1, 3, 224, 224)  # 输入图像
with torch.no_grad():
    logits = model(x)       # [1, 1000]
    probs = torch.softmax(logits, dim=-1)
    pred_class = probs.argmax(dim=-1)
```

### 运行示例

```bash
# 运行基本测试
python3 visual_transformer.py

# 运行完整示例
python3 example_visual_transformer.py
```

## ⚙️ 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `img_size` | int | 224 | 输入图像尺寸 |
| `patch_size` | int | 16 | 初始patch大小 |
| `in_channels` | int | 3 | 输入通道数 |
| `embed_dims` | list | [96, 192, 384, 768] | 各阶段embedding维度 |
| `depths` | list | [2, 3, 6, 3] | 各阶段Transformer块数量 |
| `num_heads` | list | [3, 6, 12, 24] | 各阶段注意力头数 |
| `window_size` | int | 7 | 窗口注意力大小 |
| `mlp_ratio` | float | 4.0 | MLP扩展比例 |
| `dropout` | float | 0.1 | Dropout概率 |
| `num_classes` | int | 1000 | 分类数 |

## 📊 性能对比

| 配置 | 参数量 | 输入尺寸 | 输出尺寸 | 适用场景 |
|------|--------|----------|----------|----------|
| 轻量级 | 2.2M | 224x224 | [B, 10] | 移动设备、小数据集 |
| 标准 | 12.7M | 224x224 | [B, 1000] | 通用图像分类 |
| 大型 | 45M+ | 384x384 | [B, 1000] | 大规模数据集 |

## 🔧 核心创新

### 1. 窗口注意力机制
```python
# 传统注意力：O(N²) 复杂度，N = H×W
attn = query @ key.transpose(-2, -1)  # [N, N]

# 窗口注意力：O(W²) 复杂度，W = window_size²
attn = window_query @ window_key.transpose(-2, -1)  # [W², W²]
```

### 2. 多尺度特征融合
- 通过PatchMerging实现渐进式下采样
- 每个阶段捕获不同粒度的视觉信息
- 适合处理目标尺度变化较大的任务

### 3. 高效的内存使用
- 使用非重叠窗口减少内存占用
- Pre-Normalization有助于训练稳定性
- 支持任意输入尺寸（需要是patch_size的倍数）

## 🎯 应用场景

1. **图像分类**：ImageNet、CIFAR等数据集
2. **目标检测**：作为骨干网络提取特征
3. **语义分割**：多尺度特征适合像素级预测
4. **视觉问答**：结合文本处理实现多模态理解

## 🔍 代码结构

```
visual_transformer.py
├── PatchEmbedding      # 图像分块和位置编码
├── WindowAttention     # 窗口自注意力机制
├── MLP                # 前馈网络
├── HieraBlock         # Transformer块
├── PatchMerging       # 特征下采样
└── VisualTransformer  # 主模型类
```

## 📚 学习要点

1. **Patch Embedding**：理解如何将2D图像转换为1D序列
2. **位置编码**：学习Transformer如何处理空间位置信息
3. **窗口注意力**：掌握局部注意力的实现和优势
4. **多尺度设计**：理解特征金字塔在Transformer中的应用
5. **残差连接**：学习深度网络的梯度流动技巧

## 🛠️ 自定义扩展

### 修改窗口大小
```python
# 增大窗口获得更大感受野
model = VisualTransformer(window_size=14, ...)

# 减小窗口提高计算效率
model = VisualTransformer(window_size=4, ...)
```

### 调整模型容量
```python
# 更深的模型
model = VisualTransformer(depths=[4, 6, 12, 4], ...)

# 更宽的模型
model = VisualTransformer(embed_dims=[128, 256, 512, 1024], ...)
```

### 添加新功能
```python
# 添加DropPath (Stochastic Depth)
from timm.models.layers import DropPath

class HieraBlock(nn.Module):
    def __init__(self, ...):
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
```

## 📖 参考资源

- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2) - 原始SAM2实现
- [Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/abs/2306.00989) - Hiera论文
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) - 经典ViT论文

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目仅供学习和研究使用。