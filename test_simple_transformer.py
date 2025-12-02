import torch
from simple_transformer import SimpleTransformer

def test_transformer():
    print("Testing SimpleTransformer...")
    
    # Hyperparameters
    vocab_size = 1000
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 128
    max_len = 50
    dropout = 0.1
    
    # Initialize model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    print("Model initialized successfully.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Check output shape
    expected_shape = (batch_size, seq_len, vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("Test passed!")

if __name__ == "__main__":
    test_transformer()
