# Nano Attention - 学习Transformer注意力机制

一个用于学习和理解Transformer注意力机制的简单项目。

## 项目特点

- 简单的Transformer模型实现
- 小型数据集（8个句子，约10个词汇）
- 下一个词预测任务
- 注意力权重可视化

## 环境设置

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

```bash
# 训练模型
python train.py

# 可视化注意力
python visualize_attention.py
```

## 项目结构

- `model.py` - Transformer模型实现
- `data.py` - 数据集定义
- `train.py` - 训练脚本
- `visualize_attention.py` - 注意力可视化
