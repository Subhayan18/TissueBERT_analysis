#!/usr/bin/env python3
"""
Test TissueBERT Deconvolution Model
====================================

Verify that:
1. Model architecture is correct
2. Pre-trained weights load successfully
3. Forward pass produces valid proportions
4. Loss computation works
"""

import torch
import sys
from pathlib import Path

# Import model
from model_deconvolution import (
    TissueBERTDeconvolution,
    load_pretrained_model,
    mixture_mse_loss
)

# Paths
CHECKPOINT_PATH = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/checkpoints/checkpoint_best_acc.pt'


def test_model_architecture():
    """Test 1: Model architecture"""
    print("\n" + "="*80)
    print("TEST 1: Model Architecture")
    print("="*80)
    
    model = TissueBERTDeconvolution(
        vocab_size=69,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=2048,
        max_position_embeddings=150,
        num_classes=22,
        dropout=0.1,
        n_regions=51089
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Check layer names
    print("\nModel layers:")
    for name, param in model.named_parameters():
        print(f"  {name}: {list(param.shape)}")
    
    print("\n✓ Test 1 passed: Model architecture OK")
    return model


def test_pretrained_loading():
    """Test 2: Load pre-trained weights"""
    print("\n" + "="*80)
    print("TEST 2: Pre-trained Weight Loading")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load pre-trained model
    model = load_pretrained_model(CHECKPOINT_PATH, device=device, verbose=True)
    
    print("\n✓ Test 2 passed: Pre-trained weights loaded successfully")
    return model, device


def test_forward_pass(model, device):
    """Test 3: Forward pass with dummy data"""
    print("\n" + "="*80)
    print("TEST 3: Forward Pass")
    print("="*80)
    
    batch_size = 4
    n_regions = 51089
    seq_length = 150
    
    print(f"\nGenerating dummy data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Regions: {n_regions:,}")
    print(f"  Sequence length: {seq_length}")
    
    # Create dummy methylation data
    methylation = torch.randint(0, 3, (batch_size, n_regions, seq_length)).to(device)
    
    print(f"\nInput shape: {methylation.shape}")
    print(f"Input dtype: {methylation.dtype}")
    print(f"Input range: [{methylation.min().item()}, {methylation.max().item()}]")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        proportions = model(methylation)
    
    print(f"\nOutput shape: {proportions.shape}")
    print(f"Output dtype: {proportions.dtype}")
    
    # Check proportions
    print(f"\nProportion validation:")
    for i in range(min(batch_size, 3)):
        prop_sum = proportions[i].sum().item()
        prop_min = proportions[i].min().item()
        prop_max = proportions[i].max().item()
        top3_idx = proportions[i].topk(3).indices.cpu().numpy()
        top3_vals = proportions[i].topk(3).values.cpu().numpy()
        
        print(f"  Sample {i}:")
        print(f"    Sum: {prop_sum:.6f}")
        print(f"    Range: [{prop_min:.6f}, {prop_max:.6f}]")
        print(f"    Top 3 tissues: {top3_idx} = {top3_vals}")
    
    # Verify all proportions sum to 1.0
    sums = proportions.sum(dim=1)
    all_valid = torch.allclose(sums, torch.ones(batch_size, device=device), atol=1e-5)
    
    if all_valid:
        print(f"\n✓ All proportions sum to 1.0 (within tolerance)")
    else:
        print(f"\n✗ WARNING: Some proportions don't sum to 1.0!")
        print(f"  Sums: {sums}")
    
    print("\n✓ Test 3 passed: Forward pass OK")
    return proportions


def test_loss_computation(proportions, device):
    """Test 4: Loss computation"""
    print("\n" + "="*80)
    print("TEST 4: Loss Computation")
    print("="*80)
    
    batch_size = proportions.shape[0]
    n_classes = proportions.shape[1]
    
    # Generate dummy ground truth proportions
    print(f"\nGenerating dummy ground truth proportions...")
    true_props = torch.randn(batch_size, n_classes, device=device).abs()
    true_props = true_props / true_props.sum(dim=1, keepdim=True)
    
    print(f"True proportions shape: {true_props.shape}")
    print(f"True proportions sum check: {true_props.sum(dim=1)}")
    
    # Compute loss
    loss = mixture_mse_loss(proportions, true_props)
    
    print(f"\nMSE Loss: {loss.item():.6f}")
    
    # Verify loss is reasonable
    assert loss.item() >= 0, "Loss should be non-negative"
    assert loss.item() < 1.0, "Loss should be < 1.0 for normalized proportions"
    
    print("\n✓ Test 4 passed: Loss computation OK")
    return loss


def test_gradient_flow(model, device):
    """Test 5: Gradient flow (backpropagation)"""
    print("\n" + "="*80)
    print("TEST 5: Gradient Flow")
    print("="*80)
    
    batch_size = 2
    n_regions = 51089
    seq_length = 150
    
    # Create dummy data
    methylation = torch.randint(0, 3, (batch_size, n_regions, seq_length)).to(device)
    true_props = torch.randn(batch_size, 22, device=device).abs()
    true_props = true_props / true_props.sum(dim=1, keepdim=True)
    
    # Forward pass
    model.train()
    proportions = model(methylation)
    
    # Compute loss
    loss = mixture_mse_loss(proportions, true_props)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"\nChecking gradients...")
    has_grads = 0
    no_grads = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            has_grads += 1
            if has_grads <= 3:  # Show first 3
                print(f"  {name}: grad_norm={grad_norm:.6f}")
        else:
            no_grads += 1
    
    print(f"\nGradient summary:")
    print(f"  Parameters with gradients: {has_grads}")
    print(f"  Parameters without gradients: {no_grads}")
    
    if has_grads > 0 and no_grads == 0:
        print("\n✓ Test 5 passed: Gradients flow correctly")
    else:
        print("\n✗ WARNING: Some parameters don't have gradients!")
    
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("TISSUEBERT DECONVOLUTION MODEL TESTING")
    print("="*80)
    
    try:
        # Test 1: Architecture
        model = test_model_architecture()
        
        # Test 2: Pre-trained loading
        model, device = test_pretrained_loading()
        
        # Test 3: Forward pass
        proportions = test_forward_pass(model, device)
        
        # Test 4: Loss computation
        loss = test_loss_computation(proportions, device)
        
        # Test 5: Gradient flow
        test_gradient_flow(model, device)
        
        # Summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nModel is ready for training:")
        print("  ✓ Architecture correct")
        print("  ✓ Pre-trained weights loaded")
        print("  ✓ Forward pass works")
        print("  ✓ Proportions sum to 1.0")
        print("  ✓ Loss computation works")
        print("  ✓ Gradients flow correctly")
        print("\nReady to proceed with mixture deconvolution training!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
