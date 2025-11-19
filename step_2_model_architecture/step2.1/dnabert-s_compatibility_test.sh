#!/bin/bash
#SBATCH --job-name=model_test_complete
#SBATCH --account=lu2025-7-54
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --output=model_test_%j.log
#SBATCH --error=model_test_%j.err

# Load your modules
source /home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/LMOD.sourceme

python3 << 'EOF'
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
import sys

print("="*60)
print("Complete Model Architecture Test")
print("="*60)

try:
    # Configuration matching DNABERT-S architecture
    config = BertConfig(
        vocab_size=69,              # 64 3-mers + 5 special tokens
        hidden_size=512,            
        num_hidden_layers=6,        
        num_attention_heads=8,      
        intermediate_size=2048,     
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    print(f"\n1. Model Configuration:")
    print(f"   ✓ Layers: {config.num_hidden_layers}")
    print(f"   ✓ Hidden size: {config.hidden_size}")
    print(f"   ✓ Attention heads: {config.num_attention_heads}")

    # Define full model with classification head
    class MethylationDeconvolutionModel(nn.Module):
        def __init__(self, config, n_tissues=39):
            super().__init__()
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(config.hidden_size, n_tissues)
            
        def forward(self, input_ids):
            outputs = self.bert(input_ids)
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)
            return logits

    # Create and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MethylationDeconvolutionModel(config, n_tissues=39)
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n2. Model Created:")
    print(f"   ✓ Total parameters: {param_count:.1f}M")
    print(f"   ✓ Device: {device}")

    # Test forward pass
    print(f"\n3. Forward Pass Test:")
    batch_size = 8
    seq_length = 150
    dummy_input = torch.randint(0, 69, (batch_size, seq_length)).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=-1)
    
    print(f"   ✓ Input: {dummy_input.shape}")
    print(f"   ✓ Output: {logits.shape}")
    print(f"   ✓ Probabilities sum: {probs[0].sum().item():.6f}")

    # Test backward pass
    print(f"\n4. Backward Pass Test:")
    model.train()
    dummy_labels = torch.randint(0, 39, (batch_size,)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Forward
    logits = model(dummy_input)
    loss = criterion(logits, dummy_labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Loss computed: {loss.item():.4f}")
    print(f"   ✓ Gradients computed successfully")
    print(f"   ✓ Optimizer step completed")

    # Test multiple batches (mini training loop)
    print(f"\n5. Mini Training Loop (10 steps):")
    model.train()
    losses = []
    
    for step in range(10):
        dummy_input = torch.randint(0, 69, (batch_size, seq_length)).to(device)
        dummy_labels = torch.randint(0, 39, (batch_size,)).to(device)
        
        optimizer.zero_grad()
        logits = model(dummy_input)
        loss = criterion(logits, dummy_labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    print(f"   ✓ 10 training steps completed")
    print(f"   ✓ Loss range: {min(losses):.4f} - {max(losses):.4f}")
    print(f"   ✓ Final loss: {losses[-1]:.4f}")

    # Memory usage
    print(f"\n6. GPU Memory Usage:")
    memory_allocated = torch.cuda.memory_allocated() / 1e9
    memory_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"   Allocated: {memory_allocated:.2f} GB")
    print(f"   Reserved: {memory_reserved:.2f} GB")
    print(f"   Available: ~{80 - memory_reserved:.1f} GB")

    # Test saving/loading
    print(f"\n7. Save/Load Test:")
    save_path = "/tmp/test_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, save_path)
    print(f"   ✓ Model saved to {save_path}")
    
    # Load back
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✓ Model loaded successfully")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour setup is ready for:")
    print("  • Model training")
    print("  • Gradient computation")
    print("  • Checkpointing")
    print("  • Full training pipeline")
    
    sys.exit(0)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Test completed successfully!"
else
    echo ""
    echo "Test failed with exit code: $exit_code"
fi
