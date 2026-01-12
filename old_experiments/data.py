"""
简单的数据集定义，用于训练Transformer模型
"""
import torch
from torch.utils.data import Dataset
from logger import get_logger


class SimpleTextDataset(Dataset):
    """简单的文本数据集"""

    def __init__(self, max_len=10):
        logger = get_logger()
        logger.subsection("初始化数据集")

        # 定义中文句子（空格分隔词汇）
        # "我爱吃苹果" 3次，"我爱吃香蕉" 2次，"我要去北京" 2次，"我要去上海" 1次
        self.sentences = [
            "我 爱 吃 苹果",
            "我 爱 吃 苹果",
            "我 爱 吃 苹果",
            "我 爱 吃 香蕉",
            "我 爱 吃 香蕉",
            "我 要 去 北京",
            "我 要 去 北京",
            "我 要 去 上海"
        ]

        logger.info(f"语料库包含 {len(self.sentences)} 个句子")
        logger.debug("句子列表:")
        for i, sent in enumerate(self.sentences, 1):
            logger.debug(f"  {i}. {sent}")

        self.max_len = max_len
        logger.info(f"最大序列长度: {max_len}")

        # 构建词汇表
        logger.info("开始构建词汇表...")
        self.build_vocab()

        # 将句子转换为token索引
        logger.info("开始tokenize句子...")
        self.tokenized_sentences = [self.tokenize(sent) for sent in self.sentences]
        logger.info(f"成功tokenize {len(self.tokenized_sentences)} 个句子")

    def build_vocab(self):
        """构建词汇表"""
        logger = get_logger()

        # 收集所有词汇
        all_words = set()
        for sent in self.sentences:
            words = sent.split()
            all_words.update(words)

        logger.debug(f"从语料中提取到 {len(all_words)} 个不同的词汇")
        logger.debug(f"词汇: {sorted(all_words)}")

        # 添加特殊token
        self.special_tokens = ['<PAD>', '<START>', '<END>']
        logger.debug(f"特殊tokens: {self.special_tokens}")

        # 创建词汇到索引的映射
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        for word in sorted(all_words):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        # 创建索引到词汇的映射
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}

        self.vocab_size = len(self.vocab)

        logger.info(f"词汇表大小: {self.vocab_size}")
        logger.info(f"完整词汇表: {sorted(self.vocab.keys())}")

        # 详细的词汇到索引映射
        logger.debug("词汇到索引的映射:")
        for word, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
            logger.debug(f"  '{word}' -> {idx}")

    def tokenize(self, sentence):
        """将句子转换为token索引"""
        logger = get_logger()

        words = sentence.split()
        # 添加START和END标记
        tokens = [self.vocab['<START>']] + [self.vocab[word] for word in words] + [self.vocab['<END>']]

        logger.debug(f"Tokenize '{sentence}' -> {tokens}")

        # 填充到max_len
        if len(tokens) < self.max_len:
            original_len = len(tokens)
            tokens += [self.vocab['<PAD>']] * (self.max_len - len(tokens))
            logger.debug(f"  填充: {original_len} -> {self.max_len} (添加了 {self.max_len - original_len} 个PAD)")
        else:
            tokens = tokens[:self.max_len]
            logger.debug(f"  截断到max_len: {self.max_len}")

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens):
        """将token索引转换回句子"""
        words = []
        for idx in tokens:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            word = self.idx_to_word[idx]
            if word == '<END>':
                break
            if word not in ['<PAD>', '<START>']:
                words.append(word)
        return ' '.join(words)

    def __len__(self):
        return len(self.tokenized_sentences)

    def __getitem__(self, idx):
        """
        返回输入和目标
        输入: token[:-1] (除了最后一个token)
        目标: token[1:] (除了第一个token)
        这样模型学习预测下一个词
        """
        tokens = self.tokenized_sentences[idx]

        # 输入是前n-1个token
        input_tokens = tokens[:-1]

        # 目标是后n-1个token（即每个位置预测下一个词）
        target_tokens = tokens[1:]

        return input_tokens, target_tokens


def get_dataloader(batch_size=4, max_len=10):
    """创建数据加载器"""
    logger = get_logger()
    logger.subsection("创建数据加载器")

    dataset = SimpleTextDataset(max_len=max_len)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    logger.info(f"数据加载器配置:")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  shuffle: True")
    logger.info(f"  数据集大小: {len(dataset)}")
    logger.info(f"  批次数量: {len(dataloader)}")

    return dataloader, dataset


if __name__ == "__main__":
    # 测试数据集
    dataset = SimpleTextDataset()

    print(f"\n数据集大小: {len(dataset)}")
    print(f"\n示例句子:")
    for i in range(min(3, len(dataset))):
        input_tokens, target_tokens = dataset[i]
        print(f"\n句子 {i+1}:")
        print(f"  原始: {dataset.sentences[i]}")
        print(f"  输入tokens: {input_tokens}")
        print(f"  目标tokens: {target_tokens}")
        print(f"  解码输入: {dataset.decode(input_tokens)}")
