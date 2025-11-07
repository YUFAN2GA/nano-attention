"""
训练Transformer模型进行下一个词预测
"""
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloader
from model import NanoTransformer
from logger import get_logger, reset_logger


def train_model(num_epochs=200, batch_size=4, learning_rate=0.001, device='cpu'):
    """训练模型"""
    logger = get_logger()
    logger.section("开始训练流程")

    logger.info(f"训练参数:")
    logger.info(f"  num_epochs: {num_epochs}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  device: {device}")

    # 加载数据
    logger.section("加载数据")
    dataloader, dataset = get_dataloader(batch_size=batch_size, max_len=10)

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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"损失函数: CrossEntropyLoss (忽略<PAD>)")
    logger.info(f"优化器: Adam, lr={learning_rate}")

    # 训练循环
    logger.section("开始训练循环")
    logger.info(f"总epoch数: {num_epochs}")

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_tokens, target_tokens) in enumerate(dataloader):
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            # 是否记录详细日志（仅第1个epoch的第1个batch）
            log_details = (epoch == 0 and batch_idx == 0)

            if log_details:
                logger.section("═════ 第1个Epoch 第1个Batch 的详细计算过程 ═════")
                logger.info(f"Batch大小: {input_tokens.shape[0]}, 序列长度: {input_tokens.shape[1]}")
                logger.info(f"输入token IDs:\n{input_tokens.cpu().numpy()}")
                logger.info(f"目标token IDs:\n{target_tokens.cpu().numpy()}")

            # 前向传播
            optimizer.zero_grad()
            output = model(input_tokens, log_details=log_details)  # [batch_size, seq_len, vocab_size]

            # 计算损失
            # 将output展平: [batch_size * seq_len, vocab_size]
            # 将target展平: [batch_size * seq_len]
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
                # 记录各层的梯度
                for name, param in model.named_parameters():
                    if param.grad is not None:
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

            # 记录第一个epoch的每个batch
            if epoch == 0:
                logger.debug(f"  Epoch 1, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches

        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], 平均Loss: {avg_loss:.4f}")

        # 详细记录前5个和最后5个epoch
        if epoch < 5 or epoch >= num_epochs - 5:
            logger.debug(f"Epoch [{epoch+1}/{num_epochs}] 详细信息: 平均Loss={avg_loss:.4f}, 总Loss={total_loss:.4f}, Batches={num_batches}")

    logger.section("训练完成")
    logger.info(f"最终平均Loss: {avg_loss:.4f}")

    # 保存模型
    logger.subsection("保存模型")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': dataset.vocab,
        'idx_to_word': dataset.idx_to_word,
    }
    torch.save(checkpoint, 'nano_transformer.pth')
    logger.info("模型已保存到 nano_transformer.pth")
    logger.debug(f"保存内容: model_state_dict, vocab ({len(dataset.vocab)} 个词), idx_to_word")

    # 测试模型
    logger.section("测试模型 - 预测下一个词")
    test_model(model, dataset, device)

    logger.section("训练流程全部完成!")
    return model, dataset


def test_model(model, dataset, device='cpu'):
    """测试模型的预测能力"""
    logger = get_logger()
    logger.subsection("测试所有训练句子")

    model.eval()

    with torch.no_grad():
        # 测试所有训练句子
        for i, sent in enumerate(dataset.sentences):
            logger.info(f"\n句子 {i+1}/{len(dataset.sentences)}: {sent}")

            # 获取tokenized的句子
            input_tokens, target_tokens = dataset[i]
            input_tokens = input_tokens.unsqueeze(0).to(device)  # [1, seq_len]

            # 预测
            output = model(input_tokens)  # [1, seq_len, vocab_size]

            # 获取每个位置预测概率最高的词
            predicted_indices = output.argmax(dim=-1).squeeze(0)  # [seq_len]

            # 解码预测结果
            logger.info(f"  输入: {dataset.decode(input_tokens.squeeze(0))}")
            logger.info(f"  真实下一个词序列: {dataset.decode(target_tokens)}")
            logger.info(f"  预测下一个词序列: {dataset.decode(predicted_indices)}")

            # 显示每个位置的预测概率
            logger.debug(f"  各位置预测详情:")
            for j in range(min(5, len(predicted_indices))):  # 只显示前5个位置
                # 获取该位置的概率分布
                probs = torch.softmax(output[0, j], dim=-1)
                top3_probs, top3_indices = torch.topk(probs, k=3)

                input_word = dataset.idx_to_word[input_tokens[0, j].item()]
                target_word = dataset.idx_to_word[target_tokens[j].item()]

                logger.debug(f"    位置{j} (输入词: '{input_word}') 预测下一个词:")
                logger.debug(f"      真实: '{target_word}'")
                for k in range(3):
                    pred_word = dataset.idx_to_word[top3_indices[k].item()]
                    prob = top3_probs[k].item()
                    logger.debug(f"      Top{k+1}: '{pred_word}' (概率: {prob:.3f})")


def generate_text(model, dataset, start_text, max_len=10, device='cpu'):
    """使用模型生成文本"""
    logger = get_logger()
    logger.debug(f"生成文本，起始文本: '{start_text}'")

    model.eval()

    # Tokenize起始文本
    words = start_text.split()
    tokens = [dataset.vocab['<START>']] + [dataset.vocab.get(word, dataset.vocab['<PAD>']) for word in words]
    logger.debug(f"  初始tokens: {tokens}")

    with torch.no_grad():
        for step in range(max_len - len(tokens)):
            # 准备输入
            input_tokens = torch.tensor([tokens], dtype=torch.long).to(device)

            # 预测
            output = model(input_tokens)

            # 获取最后一个位置的预测
            next_token_logits = output[0, -1, :]
            next_token = next_token_logits.argmax().item()

            logger.debug(f"  步骤{step+1}: 预测下一个token={next_token} ('{dataset.idx_to_word[next_token]}')")

            # 如果预测到END，停止生成
            if next_token == dataset.vocab['<END>']:
                logger.debug(f"  生成结束 (遇到<END>)")
                break

            tokens.append(next_token)

    # 解码
    generated_text = dataset.decode(torch.tensor(tokens))
    logger.debug(f"  最终生成文本: '{generated_text}'")
    return generated_text


if __name__ == "__main__":
    # 重置日志（清空之前的日志）
    logger = reset_logger()
    logger.section("程序启动")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 训练模型
    model, dataset = train_model(
        num_epochs=200,
        batch_size=4,
        learning_rate=0.001,
        device=device
    )

    # 尝试生成文本
    logger.section("文本生成测试")

    test_starts = ["我 爱", "我 要", "我 爱 吃"]
    for start in test_starts:
        logger.subsection(f"生成测试 - 起始: '{start}'")
        generated = generate_text(model, dataset, start, max_len=10, device=device)
        logger.info(f"起始: '{start}'")
        logger.info(f"生成: '{generated}'")

    logger.section("所有任务完成!")
    logger.info("日志已保存到 llm-log.txt")
