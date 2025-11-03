# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a learning project for understanding Transformer attention mechanisms. It implements a minimal Transformer model for next-word prediction on a tiny dataset (8 sentences, ~10 unique vocabulary words) with visualization of attention weights.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Core Commands

**Train the model:**
```bash
python train.py
```
- Trains for 200 epochs by default
- Saves model to `nano_transformer.pth`
- Displays training loss, predictions, and text generation examples

**Visualize attention:**
```bash
python visualize_attention.py
```
- Requires `nano_transformer.pth` from training
- Generates heatmap visualizations of attention weights
- Creates per-head and average attention plots
- Saves PNG files for each layer and sentence

**Test dataset:**
```bash
python data.py
```
- Displays vocabulary and tokenized examples

## Architecture

### Model Components (model.py)

- **NanoTransformer**: Main model class
  - 2 transformer layers
  - 4 attention heads
  - 64-dimensional embeddings
  - Causal masking for autoregressive generation

- **MultiHeadAttention**: Core attention mechanism
  - Stores `last_attention_weights` for visualization
  - Implements scaled dot-product attention with multiple heads

- **TransformerBlock**: Single transformer layer
  - Self-attention + residual connection + layer norm
  - Feed-forward network + residual connection + layer norm

- **PositionalEncoding**: Sinusoidal position embeddings

### Data (data.py)

- **SimpleTextDataset**: 8 hardcoded sentences
  - Vocabulary: ~13 tokens (including special tokens)
  - Special tokens: `<PAD>`, `<START>`, `<END>`
  - Returns (input_tokens[:-1], target_tokens[1:]) for next-word prediction

### Training (train.py)

- Task: Predict next token given previous tokens
- Loss: Cross-entropy with padding ignored
- Optimizer: Adam
- Includes `generate_text()` function for autoregressive generation

### Visualization (visualize_attention.py)

- Extracts attention weights via `model.get_attention_weights()`
- Creates heatmaps showing Query-Key attention patterns
- Visualizes both individual heads and averaged attention across heads
- Demonstrates how attention focuses on different tokens for prediction

## Key Implementation Details

- **Causal Masking**: `create_causal_mask()` prevents attending to future tokens
- **Attention Extraction**: Model saves attention weights during forward pass for visualization
- **Token Decoding**: `dataset.decode()` converts token indices back to readable text
- **Residual Connections**: Critical for gradient flow in transformer blocks
