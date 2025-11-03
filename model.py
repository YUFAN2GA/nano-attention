"""
简单的Transformer模型实现，用于学习注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from logger import get_logger


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出的线性变换
        self.W_o = nn.Linear(d_model, d_model)

        # 保存最后一次的注意力权重用于可视化
        self.last_attention_weights = None

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or [seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)

        # 线性变换并分成多个头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用mask（用于防止看到未来的词）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 保存注意力权重用于可视化
        self.last_attention_weights = attention_weights.detach()

        # 应用注意力权重到value
        # [batch_size, num_heads, seq_len, d_k]
        output = torch.matmul(attention_weights, V)

        # 合并多头
        # [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最后的线性变换
        output = self.W_o(output)

        return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class NanoTransformer(nn.Module):
    """简单的Transformer语言模型"""
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, d_ff=128, max_len=50, dropout=0.1):
        super().__init__()
        logger = get_logger()
        logger.subsection("初始化Transformer模型")

        self.d_model = d_model

        logger.info(f"模型超参数:")
        logger.info(f"  vocab_size: {vocab_size}")
        logger.info(f"  d_model (嵌入维度): {d_model}")
        logger.info(f"  num_heads (注意力头数): {num_heads}")
        logger.info(f"  num_layers (Transformer层数): {num_layers}")
        logger.info(f"  d_ff (前馈网络维度): {d_ff}")
        logger.info(f"  max_len (最大序列长度): {max_len}")
        logger.info(f"  dropout: {dropout}")

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        logger.debug(f"创建词嵌入层: {vocab_size} x {d_model}")

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        logger.debug(f"创建位置编码: max_len={max_len}, d_model={d_model}")

        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        logger.debug(f"创建 {num_layers} 个Transformer块")

        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
        logger.debug(f"创建输出层: {d_model} -> {vocab_size}")

        self.dropout = nn.Dropout(dropout)

        # 计算总参数量
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"模型总参数量: {total_params:,}")

    def create_causal_mask(self, seq_len):
        """创建因果mask，防止看到未来的词"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] - 输入的token索引
        Returns:
            output: [batch_size, seq_len, vocab_size] - 每个位置的词概率分布
        """
        seq_len = x.size(1)

        # 创建因果mask
        mask = self.create_causal_mask(seq_len).to(x.device)

        # 词嵌入 * sqrt(d_model) （按照原始论文）
        x = self.embedding(x) * math.sqrt(self.d_model)

        # 添加位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过Transformer块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # 输出层
        output = self.fc_out(x)

        return output

    def get_attention_weights(self):
        """获取所有层的注意力权重，用于可视化"""
        attention_weights = []
        for block in self.transformer_blocks:
            if block.attention.last_attention_weights is not None:
                attention_weights.append(block.attention.last_attention_weights)
        return attention_weights
