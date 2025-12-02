#!/usr/bin/env python3
"""
TissueBERT Deconvolution Model
===============================

Modified version of TissueBERT for mixture deconvolution.

Key changes from single-tissue classification:
1. Output: Sigmoid + L1 normalization (instead of softmax)
2. Loss: MSE on proportions (instead of CrossEntropy)
3. Task: Predict tissue proportions [22] (instead of single class)

Architecture identical to original up to classifier layer, enabling
transfer learning from pre-trained single-tissue model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TissueBERTDeconvolution(nn.Module):
    """
    File-Level MLP for Mixture Deconvolution
    
    Input: [batch, 51089, 150] methylation (mixed samples)
    Process:
        1. Compute mean methylation per region → [batch, 51089]
        2. Project to hidden dimension → [batch, 512]
        3. MLP feature extraction → [batch, 1024]
        4. Classifier → [batch, 22]
        5. Sigmoid + L1 normalization → [batch, 22] proportions (sum=1.0)
    """

    def __init__(self,
                 vocab_size=69,
                 hidden_size=512,
                 num_hidden_layers=3,
                 num_attention_heads=4,
                 intermediate_size=2048,
                 max_position_embeddings=150,
                 num_classes=22,
                 dropout=0.1,
                 n_regions=51089):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.n_regions = n_regions

        # Input projection: 51089 → hidden_size
        self.input_projection = nn.Linear(n_regions, hidden_size)
        
        # MLP network (identical to original)
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(hidden_size, intermediate_size),
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output (classifier)
            nn.Linear(intermediate_size, num_classes)
        )
        
        # NEW: Sigmoid activation for independent tissue probabilities
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self._init_weights)
        
        # Calculate parameter count
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 70)
        print("TISSUEBERT DECONVOLUTION MODEL")
        print("=" * 70)
        print(f"Input: [batch, {n_regions:,} regions, 150 bp] methylation")
        print(f"Process: Region aggregation → MLP → Deconvolution")
        print(f"Output: [batch, {num_classes}] tissue proportions (sum=1.0)")
        print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"Memory: ~{n_params*4/1e6:.0f} MB (float32)")
        print("=" * 70)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, methylation_status):
        """
        Forward pass with proportion output
        
        Args:
            methylation_status: [batch, n_regions, seq_length]
                               Values: 0 (unmeth), 1 (meth), 2 (missing)
            
        Returns:
            proportions: [batch, num_classes] tissue proportions (sum=1.0)
        """
        batch_size = methylation_status.shape[0]
        n_regions = methylation_status.shape[1]
        seq_length = methylation_status.shape[2]
        
        # Verify shape
        assert n_regions == self.n_regions, \
            f"Expected {self.n_regions} regions, got {n_regions}"
        
        # ===================================================================
        # STEP 1: Compute mean methylation per region (same as original)
        # ===================================================================
        
        # Mask missing values (2) and convert to float
        valid_mask = (methylation_status != 2).float()
        meth_float = methylation_status.float()
        meth_float = torch.where(methylation_status == 2, 
                                torch.zeros_like(meth_float), 
                                meth_float)
        
        # Compute mean per region
        region_sums = (meth_float * valid_mask).sum(dim=2)  # [batch, n_regions]
        region_counts = valid_mask.sum(dim=2).clamp(min=1.0)  # [batch, n_regions]
        region_means = region_sums / region_counts  # [batch, n_regions]
        
        # ===================================================================
        # STEP 2: Project to hidden dimension (same as original)
        # ===================================================================
        
        features = self.input_projection(region_means)  # [batch, hidden_size]
        
        # ===================================================================
        # STEP 3: MLP feature extraction (same as original)
        # ===================================================================
        
        logits = self.network(features)  # [batch, num_classes]
        
        # ===================================================================
        # STEP 4: NEW - Sigmoid + L1 normalization for proportions
        # ===================================================================
        
        # Apply sigmoid to get independent probabilities
        sigmoid_outputs = self.sigmoid(logits)  # [batch, num_classes]
        
        # L1 normalization to sum to 1.0
        proportions = sigmoid_outputs / sigmoid_outputs.sum(dim=1, keepdim=True)
        
        return proportions


def load_pretrained_model(checkpoint_path, device='cuda', verbose=True):
    """
    Load pre-trained single-tissue model and convert to deconvolution model.
    
    The architecture is identical, so all weights can be loaded directly.
    Only the forward pass changes (sigmoid+normalize instead of softmax).
    
    Args:
        checkpoint_path: Path to checkpoint_best_acc.pt from single-tissue training
        device: 'cuda' or 'cpu'
        verbose: Print loading information
        
    Returns:
        model: TissueBERTDeconvolution with loaded weights
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING PRE-TRAINED MODEL")
        print("="*80)
        print(f"Checkpoint: {checkpoint_path}")
    
    # Initialize deconvolution model
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if verbose:
        print(f"\nCheckpoint info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
        print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    # Load state dict
    # The model architecture is identical, so strict loading should work
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    if verbose:
        print(f"\n✓ Successfully loaded pre-trained weights")
        print(f"  All layers loaded successfully")
        print("="*80)
    
    return model.to(device)


def mixture_mse_loss(predicted_proportions, true_proportions):
    """
    Mean Squared Error loss on tissue proportions.
    
    Args:
        predicted_proportions: [batch, 22] - model predictions (sum to 1.0)
        true_proportions: [batch, 22] - ground truth mixing weights (sum to 1.0)
        
    Returns:
        loss: scalar MSE value
    """
    # MSE across all tissues
    loss = F.mse_loss(predicted_proportions, true_proportions)
    
    return loss


def test_model():
    """Test deconvolution model with dummy data"""
    print("\nTesting TissueBERT Deconvolution Model...")
    
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
    
    batch_size = 4
    n_regions = 51089
    seq_length = 150
    
    print(f"\nTest input shapes:")
    print(f"  Batch size: {batch_size}")
    print(f"  Regions: {n_regions:,}")
    print(f"  Sequence length: {seq_length}")
    
    # Create dummy data
    methylation = torch.randint(0, 3, (batch_size, n_regions, seq_length))
    
    # Test forward pass
    with torch.no_grad():
        proportions = model(methylation)
    
    print(f"\nOutput shape: {proportions.shape}")
    assert proportions.shape == (batch_size, 22)
    
    print(f"\nProportion checks:")
    for i in range(batch_size):
        prop_sum = proportions[i].sum().item()
        prop_max = proportions[i].max().item()
        prop_min = proportions[i].min().item()
        print(f"  Sample {i}: sum={prop_sum:.6f}, min={prop_min:.6f}, max={prop_max:.6f}")
    
    # Verify all sum to 1.0
    sums = proportions.sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), \
        f"Proportions don't sum to 1.0: {sums}"
    
    # Test loss function
    true_props = torch.randn(batch_size, 22).abs()
    true_props = true_props / true_props.sum(dim=1, keepdim=True)
    
    loss = mixture_mse_loss(proportions, true_props)
    print(f"\nTest loss: {loss.item():.6f}")
    
    print("\n✓ Deconvolution model test passed!")


if __name__ == '__main__':
    test_model()
