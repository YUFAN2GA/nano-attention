"""
训练脚本 - 详细日志版本
仅训练5个epoch，用于查看详细的注意力机制计算过程
"""
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloader
from model import NanoTransformer
from logger import get_logger, reset_logger


if __name__ == "__main__":
    # 重置日志（清空之前的日志）
    logger = reset_logger()
    logger.section("详细日志训练模式启动")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.section("加载数据")
    dataloader, dataset = get_dataloader(batch_size=4, max_len=10)

    # 创建模型
    logger.section("创建模型")
    model = NanoTransformer(
        vocab_size=dataset.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_len=50,
        dropout=0.1
    ).to(device)

    logger.info(f"模型已移动到设备: {device}")

    # 定义损失函数和优化器
    logger.subsection("配置训练组件")
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logger.info(f"损失函数: CrossEntropyLoss (忽略<PAD>)")
    logger.info(f"优化器: Adam, lr=0.001")

    # 训练循环（仅5个epoch用于演示）
    logger.section("开始训练循环（详细模式）")
    num_epochs = 5
    logger.info(f"总epoch数: {num_epochs}")

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_tokens, target_tokens) in enumerate(dataloader):
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            # 仅第1个epoch的第1个batch记录详细日志
            log_details = (epoch == 0 and batch_idx == 0)

            if log_details:
                logger.section("═════ 第1个Epoch 第1个Batch 的详细计算过程 ═════")
                logger.info(f"Batch大小: {input_tokens.shape[0]}, 序列长度: {input_tokens.shape[1]}")
                logger.info(f"输入token IDs:\n{input_tokens.cpu().numpy()}")
                logger.info(f"目标token IDs:\n{target_tokens.cpu().numpy()}")

                # 解码显示实际词汇
                logger.info("输入句子:")
                for i in range(input_tokens.shape[0]):
                    sentence = dataset.decode(input_tokens[i])
                    logger.info(f"  样本{i+1}: {sentence}")

            # 前向传播
            optimizer.zero_grad()
            output = model(input_tokens, log_details=log_details)

            # 计算损失
            loss = criterion(output.view(-1, dataset.vocab_size), target_tokens.view(-1))

            if log_details:
                logger.subsection("损失计算")
                logger.debug(f"损失值: {loss.item():.6f}")
                logger.debug(f"Output logits (第1个样本，第1个位置的前5个类别): {output[0, 0, :5].detach().cpu().numpy()}")

                # 显示第一个位置的预测概率
                probs = torch.softmax(output[0, 0], dim=-1)
                top5_probs, top5_indices = torch.topk(probs, k=5)
                logger.debug(f"第1个样本，第1个位置的预测概率 Top5:")
                for i in range(5):
                    word = dataset.idx_to_word[top5_indices[i].item()]
                    prob = top5_probs[i].item()
                    logger.debug(f"  {i+1}. '{word}' (ID={top5_indices[i].item()}): {prob:.4f}")

            # 反向传播
            loss.backward()

            if log_details:
                logger.subsection("梯度统计")
                # 记录关键层的梯度
                key_params = ['embedding.weight', 'transformer_blocks.0.attention.W_q.weight',
                              'transformer_blocks.0.attention.W_k.weight', 'fc_out.weight']
                for name, param in model.named_parameters():
                    if param.grad is not None and name in key_params:
                        grad_mean = param.grad.mean().item()
                        grad_std = param.grad.std().item()
                        grad_max = param.grad.max().item()
                        grad_min = param.grad.min().item()
                        logger.debug(f"{name}:")
                        logger.debug(f"  形状: {list(param.grad.shape)}")
                        logger.debug(f"  梯度: 均值={grad_mean:.6f}, 标准差={grad_std:.6f}, 范围=[{grad_min:.6f}, {grad_max:.6f}]")

            optimizer.step()

            if log_details:
                logger.subsection("参数更新后统计")
                logger.debug(f"Embedding权重统计: 范围=[{model.embedding.weight.min():.4f}, {model.embedding.weight.max():.4f}], 均值={model.embedding.weight.mean():.4f}")
                logger.section("═════ 详细计算过程结束 ═════")

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], 平均Loss: {avg_loss:.4f}")

    logger.section("训练完成!")
    logger.info(f"最终平均Loss: {avg_loss:.4f}")
    logger.info("详细日志已保存到 llm-log.txt")
