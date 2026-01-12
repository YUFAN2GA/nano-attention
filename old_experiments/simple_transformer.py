import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    位置编码 (Positional Encoding)
    
    Transformer模型没有循环结构或卷积结构，为了让模型利用序列的顺序信息，
    我们需要注入关于token在序列中相对或绝对位置的信息。
    这里使用正弦和余弦函数进行位置编码。
    """
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: 模型的维度 (embedding dimension)
            max_len: 预计算的最大序列长度
        """
        super().__init__()
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母中的项: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加batch维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为buffer，这样它会成为state_dict的一部分，但不会被视为模型参数（不需要梯度更新）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            加上位置编码后的张量
        """
        # 截取与输入序列长度对应的位置编码并相加
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    允许模型同时关注来自不同表示子空间的不同位置的信息。
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义Q, K, V的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出的线性变换层
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量，用于屏蔽某些位置 (如padding或未来的token)
        """
        batch_size = query.size(0)
        
        # 1. 线性变换并分头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算缩放点积注意力 (Scaled Dot-Product Attention)
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask (如果提供)
        if mask is not None:
            # 将mask为0的位置填充为极小的负数，使其在softmax后接近0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 3. 加权求和
        # [batch_size, num_heads, seq_len, d_k]
        output = torch.matmul(attn_weights, V)
        
        # 4. 合并多头并进行最终线性变换
        # [batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, num_heads, d_k] -> [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)


class FeedForward(nn.Module):
    """
    前馈神经网络 (Position-wise Feed-Forward Networks)
    
    包含两个线性变换，中间有一个ReLU激活函数。
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    """
    Transformer编码器/解码器块
    
    包含多头注意力和前馈网络，以及残差连接和层归一化。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层1: 多头注意力 + 残差连接 + 层归一化
        # 注意: 这里的实现采用 Post-LN (Norm在残差连接之后)，也可以采用 Pre-LN
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 子层2: 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class SimpleTransformer(nn.Module):
    """
    简化的Transformer模型 (类似GPT的Decoder-only架构)
    """
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, d_ff=128, max_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 堆叠多个Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终输出层 (将隐藏状态映射回词表大小)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)

    def create_causal_mask(self, seq_len):
        """
        创建因果掩码 (Causal Mask) / 下三角掩码
        用于在训练时防止模型看到未来的token。
        """
        # 创建下三角矩阵，对角线及以下为1，上方为0
        mask = torch.tril(torch.ones(seq_len, seq_len))
        # 扩展维度以匹配注意力分数的形状 [batch, heads, seq, seq]
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: 输入token索引 [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        seq_len = x.size(1)
        device = x.device
        
        # 1. 嵌入层 + 缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 2. 加上位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 3. 创建因果mask
        mask = self.create_causal_mask(seq_len).to(device)
        
        # 4. 通过所有Transformer块
        for block in self.transformer_blocks:
            x = block(x, mask)
            
        # 5. 最终线性层
        output = self.fc_out(x)
        
        return output
