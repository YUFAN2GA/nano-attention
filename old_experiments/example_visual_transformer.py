"""
Visual Transformer 使用示例
演示如何使用Visual Transformer进行图像分类
"""

import torch
import torch.nn.functional as F
from visual_transformer import VisualTransformer

def create_sample_model():
    """创建一个示例模型"""
    print("创建Visual Transformer模型...")

    model = VisualTransformer(
        img_size=224,           # 输入图像尺寸
        patch_size=16,          # patch大小
        in_channels=3,          # RGB图像
        embed_dims=[96, 192, 384],  # 各阶段embedding维度
        depths=[2, 3, 6],       # 各阶段层数
        num_heads=[3, 6, 12],   # 各阶段注意力头数
        window_size=7,          # 窗口大小
        num_classes=10          # 分类数（例如CIFAR-10）
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model

def demo_image_classification():
    """演示图像分类"""
    print("\n" + "="*50)
    print("图像分类演示")
    print("="*50)

    # 创建模型
    model = create_sample_model()
    model.eval()

    # 模拟输入图像 (batch_size=2, 3通道, 224x224)
    batch_size = 2
    input_images = torch.randn(batch_size, 3, 224, 224)

    print(f"输入图像形状: {input_images.shape}")

    # 前向传播
    with torch.no_grad():
        logits = model(input_images)
        probabilities = F.softmax(logits, dim=-1)
        predicted_classes = probabilities.argmax(dim=-1)

    print(f"输出logits形状: {logits.shape}")
    print(f"预测类别: {predicted_classes.tolist()}")

    # 显示每个类别的概率
    print("\n预测概率:")
    for i, probs in enumerate(probabilities):
        top3_probs, top3_indices = torch.topk(probs, 3)
        print(f"样本 {i+1}:")
        for j, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            print(f"  Top{j+1}: 类别{idx.item()} (概率: {prob.item():.4f})")

def demo_feature_extraction():
    """演示特征提取"""
    print("\n" + "="*50)
    print("特征提取演示")
    print("="*50)

    model = create_sample_model()

    # 输入图像
    input_image = torch.randn(1, 3, 224, 224)

    # 提取特征
    with torch.no_grad():
        features = model.forward_features(input_image)

    print(f"输入图像形状: {input_image.shape}")
    print(f"提取的特征形状: {features.shape}")
    print(f"特征维度: {features.shape[1]}")

def demo_different_configurations():
    """演示不同配置的模型"""
    print("\n" + "="*50)
    print("不同配置演示")
    print("="*50)

    configurations = [
        {
            "name": "轻量级配置",
            "config": {
                "img_size": 224,
                "patch_size": 16,
                "embed_dims": [48, 96, 192],
                "depths": [1, 2, 4],
                "num_heads": [2, 4, 8],
                "num_classes": 10
            }
        },
        {
            "name": "标准配置",
            "config": {
                "img_size": 224,
                "patch_size": 16,
                "embed_dims": [96, 192, 384],
                "depths": [2, 3, 6],
                "num_heads": [3, 6, 12],
                "num_classes": 100
            }
        }
    ]

    for config_info in configurations:
        name = config_info["name"]
        config = config_info["config"]

        print(f"\n{name}:")
        model = VisualTransformer(**config)
        total_params = sum(p.numel() for p in model.parameters())

        # 测试前向传播
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config["img_size"], config["img_size"])
            output = model(dummy_input)

        print(f"  参数量: {total_params:,}")
        print(f"  输出形状: {output.shape}")

def demo_training_step():
    """演示训练步骤"""
    print("\n" + "="*50)
    print("训练步骤演示")
    print("="*50)

    model = create_sample_model()

    # 模拟训练数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 10, (batch_size,))  # 假设10个类别

    # 设置为训练模式
    model.train()

    # 前向传播
    logits = model(images)

    # 计算损失
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)

    # 反向传播
    loss.backward()

    print(f"输入形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"Logits形状: {logits.shape}")
    print(f"损失值: {loss.item():.4f}")

if __name__ == "__main__":
    print("Visual Transformer 使用示例")
    print("基于SAM2 Hiera架构的精简版视觉Transformer")

    # 运行所有演示
    demo_image_classification()
    demo_feature_extraction()
    demo_different_configurations()
    demo_training_step()

    print("\n" + "="*50)
    print("所有演示完成!")
    print("="*50)

    print("\n使用说明:")
    print("1. 根据任务需求调整模型配置 (img_size, embed_dims, depths等)")
    print("2. 对于较大的数据集，建议使用更大的embedding维度")
    print("3. 窗口大小window_size影响计算效率和感受野")
    print("4. 可以通过增加depths来提高模型容量")
    print("5. 适合用于图像分类、目标检测、语义分割等视觉任务")