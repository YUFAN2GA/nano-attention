"""
可视化Transformer模型的注意力权重
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import NanoTransformer
from data import SimpleTextDataset
from logger import get_logger, reset_logger


def load_model(model_path='nano_transformer.pth', device='cpu'):
    """加载训练好的模型"""
    logger = get_logger()
    logger.subsection("加载训练好的模型")
    logger.info(f"模型路径: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    logger.debug(f"成功加载checkpoint")

    # 创建数据集以获取vocab_size
    dataset = SimpleTextDataset()

    # 创建模型
    model = NanoTransformer(
        vocab_size=dataset.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_len=50,
        dropout=0.1
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("模型权重加载完成，设置为评估模式")

    return model, dataset


def visualize_attention(model, dataset, sentence_idx=0, device='cpu'):
    """
    可视化给定句子的注意力权重

    Args:
        model: 训练好的模型
        dataset: 数据集
        sentence_idx: 要可视化的句子索引
        device: 设备
    """
    logger = get_logger()
    logger.subsection(f"可视化句子 {sentence_idx+1} 的注意力")

    model.eval()

    # 获取句子
    sentence = dataset.sentences[sentence_idx]
    input_tokens, _ = dataset[sentence_idx]
    input_tokens = input_tokens.unsqueeze(0).to(device)  # [1, seq_len]

    # 获取token对应的词
    words = ['<START>'] + sentence.split() + ['<END>']
    # 截断到实际长度
    seq_len = input_tokens.size(1)
    words = words[:seq_len]

    logger.info(f"句子: {sentence}")
    logger.info(f"Tokens: {words}")
    logger.debug(f"序列长度: {seq_len}")

    # 前向传播
    with torch.no_grad():
        output = model(input_tokens)

        # 获取所有层的注意力权重
        attention_weights_all_layers = model.get_attention_weights()
        logger.debug(f"提取到 {len(attention_weights_all_layers)} 层的注意力权重")

    # 为每一层创建可视化
    num_layers = len(attention_weights_all_layers)

    for layer_idx, attention_weights in enumerate(attention_weights_all_layers):
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        attention_weights = attention_weights[0].cpu().numpy()  # [num_heads, seq_len, seq_len]
        num_heads = attention_weights.shape[0]

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Layer {layer_idx + 1} - Attention Weights', fontsize=16, fontweight='bold')

        # 为每个注意力头创建热力图
        for head_idx in range(num_heads):
            ax = axes[head_idx // 2, head_idx % 2]

            # 获取该头的注意力权重
            attn = attention_weights[head_idx]  # [seq_len, seq_len]

            # 只显示有效的tokens（非padding）
            valid_len = len(words)
            attn_valid = attn[:valid_len, :valid_len]

            # 创建热力图
            sns.heatmap(
                attn_valid,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                xticklabels=words[:valid_len],
                yticklabels=words[:valid_len],
                ax=ax,
                cbar_kws={'label': 'Attention Weight'},
                vmin=0,
                vmax=1
            )

            ax.set_title(f'Head {head_idx + 1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Key (被关注的词)', fontsize=10)
            ax.set_ylabel('Query (当前词)', fontsize=10)

            # 旋转x轴标签
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        filename = f'attention_layer{layer_idx + 1}_sentence{sentence_idx}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"已保存: {filename}")
        plt.show()

    # 创建平均注意力可视化
    visualize_average_attention(attention_weights_all_layers, words, sentence_idx)


def visualize_average_attention(attention_weights_all_layers, words, sentence_idx):
    """可视化所有头的平均注意力"""
    logger = get_logger()

    for layer_idx, attention_weights in enumerate(attention_weights_all_layers):
        # 计算所有头的平均注意力
        attention_weights = attention_weights[0].cpu().numpy()  # [num_heads, seq_len, seq_len]
        avg_attention = attention_weights.mean(axis=0)  # [seq_len, seq_len]
        logger.debug(f"Layer {layer_idx+1}: 计算平均注意力，shape={avg_attention.shape}")

        valid_len = len(words)
        avg_attention_valid = avg_attention[:valid_len, :valid_len]

        # 创建图
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            avg_attention_valid,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            xticklabels=words[:valid_len],
            yticklabels=words[:valid_len],
            cbar_kws={'label': 'Average Attention Weight'},
            vmin=0,
            vmax=1
        )

        plt.title(f'Layer {layer_idx + 1} - Average Attention (所有头平均)', fontsize=14, fontweight='bold')
        plt.xlabel('Key (被关注的词)', fontsize=11)
        plt.ylabel('Query (当前词)', fontsize=11)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        filename = f'attention_avg_layer{layer_idx + 1}_sentence{sentence_idx}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"已保存: {filename}")
        plt.show()


def visualize_all_sentences(model, dataset, device='cpu'):
    """可视化所有句子的注意力"""
    print("=" * 60)
    print("可视化所有句子的注意力权重")
    print("=" * 60)

    for i in range(len(dataset.sentences)):
        print(f"\n{'='*60}")
        print(f"句子 {i + 1}/{len(dataset.sentences)}")
        print(f"{'='*60}")
        visualize_attention(model, dataset, sentence_idx=i, device=device)


def compare_attention_patterns(model, dataset, sentence_indices=[0, 1], device='cpu'):
    """比较不同句子的注意力模式"""

    fig, axes = plt.subplots(1, len(sentence_indices), figsize=(7 * len(sentence_indices), 6))

    if len(sentence_indices) == 1:
        axes = [axes]

    for idx, sent_idx in enumerate(sentence_indices):
        sentence = dataset.sentences[sent_idx]
        input_tokens, _ = dataset[sent_idx]
        input_tokens = input_tokens.unsqueeze(0).to(device)

        words = ['<START>'] + sentence.split() + ['<END>']
        seq_len = input_tokens.size(1)
        words = words[:seq_len]

        with torch.no_grad():
            output = model(input_tokens)
            attention_weights_all_layers = model.get_attention_weights()

        # 使用第一层的平均注意力
        attention_weights = attention_weights_all_layers[0][0].cpu().numpy()
        avg_attention = attention_weights.mean(axis=0)

        valid_len = len(words)
        avg_attention_valid = avg_attention[:valid_len, :valid_len]

        ax = axes[idx]
        sns.heatmap(
            avg_attention_valid,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            xticklabels=words[:valid_len],
            yticklabels=words[:valid_len],
            ax=ax,
            cbar_kws={'label': 'Attention'},
            vmin=0,
            vmax=1
        )

        ax.set_title(f'"{sentence}"', fontsize=11, fontweight='bold')
        ax.set_xlabel('Key', fontsize=10)
        ax.set_ylabel('Query', fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    filename = 'attention_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    logger = get_logger()
    logger.info(f"已保存: {filename}")
    plt.show()


if __name__ == "__main__":
    # 重置日志
    logger = reset_logger()
    logger.section("注意力可视化程序启动")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载模型
    logger.section("加载模型")
    model, dataset = load_model('nano_transformer.pth', device=device)

    # 可视化第一个句子
    logger.section("可视化示例句子的注意力")
    visualize_attention(model, dataset, sentence_idx=0, device=device)

    # 比较不同句子的注意力模式
    logger.section("比较不同句子的注意力模式")
    compare_attention_patterns(model, dataset, sentence_indices=[0, 1, 2], device=device)

    # 可选：可视化所有句子（会生成很多图片）
    visualize_all = input("\n是否可视化所有句子? (y/n): ")
    if visualize_all.lower() == 'y':
        logger.section("可视化所有句子")
        visualize_all_sentences(model, dataset, device=device)

    logger.section("可视化完成!")
    logger.info("日志已保存到 llm-log.txt")
