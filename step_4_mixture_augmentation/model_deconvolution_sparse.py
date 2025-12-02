#!/usr/bin/env python3
"""
TissueBERT Deconvolution Model with Sparsity Regularization
=============================================================

Two-Stage Architecture for Sparse Mixture Deconvolution:
1. Presence Detection: Binary classification (tissue present/absent)
2. Proportion Estimation: Quantify proportions only for present tissues

This addresses the "baseline noise" problem where models predict
small non-zero proportions for absent tissues, stealing probability
mass from true tissues.

Key changes from original:
- Two output heads (presence + proportion)
- Sparse output via presence masking
- Combined loss (presence BCE + proportion MSE + sparsity L1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TissueBERTDeconvolution(nn.Module):
    """
    Two-Stage File-Level MLP for Mixture Deconvolution with Sparsity
    
    Input: [batch, 51089, 150] methylation (mixed samples)
    Process:
        1. Compute mean methylation per region → [batch, 51089]
        2. Project to hidden dimension → [batch, 512]
        3. MLP feature extraction → [batch, intermediate_size]
        
        Stage 1 (Presence Detection):
        4a. Presence head → [batch, 22] binary presence logits
        4b. Sigmoid → [batch, 22] presence probabilities
        
        Stage 2 (Proportion Estimation):
        5a. Proportion head → [batch, 22] proportion logits
        5b. Mask by presence → zero out absent tissues
        5c. Softmax + normalization → [batch, 22] proportions (sum=1.0)
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
                 n_regions=51089,
                 use_two_stage=True,
                 presence_threshold=0.5,
                 sparsity_regularization=True):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.n_regions = n_regions
        self.use_two_stage = use_two_stage
        self.presence_threshold = presence_threshold
        self.sparsity_regularization = sparsity_regularization

        # Input projection: 51089 → hidden_size
        self.input_projection = nn.Linear(n_regions, hidden_size)
        
        # Shared MLP backbone
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
        )
        
        if use_two_stage:
            # Stage 1: Presence detection head
            self.presence_head = nn.Linear(intermediate_size, num_classes)
            
            # Stage 2: Proportion estimation head
            self.proportion_head = nn.Linear(intermediate_size, num_classes)
        else:
            # Single-stage: original architecture
            self.classifier = nn.Linear(intermediate_size, num_classes)
            self.sigmoid = nn.Sigmoid()
        
        self.apply(self._init_weights)
        
        # Calculate parameter count
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        mode_str = "TWO-STAGE" if use_two_stage else "SINGLE-STAGE"
        sparse_str = "WITH SPARSITY" if sparsity_regularization else "NO SPARSITY"
        
        print("=" * 70)
        print(f"TISSUEBERT DECONVOLUTION MODEL ({mode_str}, {sparse_str})")
        print("=" * 70)
        print(f"Input: [batch, {n_regions:,} regions, 150 bp] methylation")
        if use_two_stage:
            print("Stage 1: Presence detection (binary classification)")
            print("Stage 2: Proportion estimation (masked by presence)")
        else:
            print("Process: Region aggregation → MLP → Sigmoid + L1 norm")
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

    def forward(self, methylation_status, return_presence=False):
        """
        Forward pass with two-stage sparse deconvolution
        
        Args:
            methylation_status: [batch, n_regions, seq_length]
                               Values: 0 (unmeth), 1 (meth), 2 (missing)
            return_presence: If True, return (proportions, presence_probs)
            
        Returns:
            proportions: [batch, num_classes] tissue proportions (sum=1.0)
            presence_probs: [batch, num_classes] presence probabilities (if requested)
        """
        batch_size = methylation_status.shape[0]
        n_regions = methylation_status.shape[1]
        seq_length = methylation_status.shape[2]
        
        # Verify shape
        assert n_regions == self.n_regions, \
            f"Expected {self.n_regions} regions, got {n_regions}"
        
        # ===================================================================
        # STEP 1: Compute mean methylation per region
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
        # STEP 2: Project to hidden dimension
        # ===================================================================
        
        features = self.input_projection(region_means)  # [batch, hidden_size]
        
        # ===================================================================
        # STEP 3: MLP feature extraction (shared backbone)
        # ===================================================================
        
        shared_features = self.network(features)  # [batch, intermediate_size]
        
        # ===================================================================
        # STEP 4 & 5: Two-Stage Output
        # ===================================================================
        
        if self.use_two_stage:
            # Stage 1: Presence detection
            presence_logits = self.presence_head(shared_features)  # [batch, num_classes]
            presence_probs = torch.sigmoid(presence_logits)  # [0, 1]
            
            # Stage 2: Proportion estimation
            prop_logits = self.proportion_head(shared_features)  # [batch, num_classes]
            
            # Mask absent tissues (during inference, use hard threshold)
            if self.training:
                # During training: use soft masking (differentiable)
                presence_mask = presence_probs
            else:
                # During inference: use hard masking
                presence_mask = (presence_probs > self.presence_threshold).float()
            
            # Apply mask and normalize
            masked_logits = prop_logits * presence_mask
            
            # Softmax over masked logits
            proportions = F.softmax(masked_logits, dim=-1)
            
            # Ensure normalization (handle edge case where all masked)
            prop_sum = proportions.sum(dim=1, keepdim=True).clamp(min=1e-8)
            proportions = proportions / prop_sum
            
            if return_presence:
                return proportions, presence_probs
            else:
                return proportions
        
        else:
            # Single-stage: original sigmoid + L1 normalization
            logits = self.classifier(shared_features)
            sigmoid_outputs = self.sigmoid(logits)
            proportions = sigmoid_outputs / sigmoid_outputs.sum(dim=1, keepdim=True)
            
            if return_presence:
                # No presence prediction in single-stage mode
                return proportions, None
            else:
                return proportions


def mixture_loss_with_sparsity(predicted_proportions, true_proportions, 
                                 presence_probs=None, true_presence=None,
                                 mse_weight=1.0, presence_weight=1.0, 
                                 sparsity_weight=0.01):
    """
    Combined loss for two-stage sparse deconvolution
    
    Args:
        predicted_proportions: [batch, 22] - predicted proportions
        true_proportions: [batch, 22] - ground truth proportions
        presence_probs: [batch, 22] - predicted presence probabilities (optional)
        true_presence: [batch, 22] - ground truth presence (binary) (optional)
        mse_weight: Weight for proportion MSE loss
        presence_weight: Weight for presence BCE loss
        sparsity_weight: Weight for L1 sparsity regularization
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    # Proportion MSE loss
    mse_loss = F.mse_loss(predicted_proportions, true_proportions)
    
    loss_dict = {'mse_loss': mse_loss.item()}
    total_loss = mse_weight * mse_loss
    
    # Presence BCE loss (if two-stage)
    if presence_probs is not None and true_presence is not None:
        presence_loss = F.binary_cross_entropy(presence_probs, true_presence)
        loss_dict['presence_loss'] = presence_loss.item()
        total_loss = total_loss + presence_weight * presence_loss
    
    # Sparsity regularization (L1 penalty on proportions)
    if sparsity_weight > 0:
        sparsity_loss = predicted_proportions.abs().sum(dim=1).mean()
        loss_dict['sparsity_loss'] = sparsity_loss.item()
        total_loss = total_loss + sparsity_weight * sparsity_loss
    
    loss_dict['total_loss'] = total_loss.item()
    
    return total_loss, loss_dict


def compute_true_presence(true_proportions, threshold=0.01):
    """
    Compute binary presence labels from ground truth proportions
    
    Args:
        true_proportions: [batch, num_classes] - ground truth proportions
        threshold: Minimum proportion to consider tissue as "present"
        
    Returns:
        true_presence: [batch, num_classes] - binary presence (0 or 1)
    """
    return (true_proportions >= threshold).float()


def load_pretrained_model(checkpoint_path, device='cuda', use_two_stage=True, verbose=True):
    """
    Load pre-trained model and optionally convert to two-stage
    
    Args:
        checkpoint_path: Path to checkpoint
        device: 'cuda' or 'cpu'
        use_two_stage: Whether to use two-stage architecture
        verbose: Print loading information
        
    Returns:
        model: TissueBERTDeconvolution with loaded weights
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING PRE-TRAINED MODEL")
        print("="*80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Mode: {'TWO-STAGE' if use_two_stage else 'SINGLE-STAGE'}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config if available
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Initialize model
    model = TissueBERTDeconvolution(
        vocab_size=69,
        hidden_size=model_config.get('hidden_size', 512),
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=model_config.get('intermediate_size', 2048),
        max_position_embeddings=150,
        num_classes=model_config.get('num_classes', 22),
        dropout=model_config.get('dropout', 0.1),
        n_regions=model_config.get('n_regions', 51089),
        use_two_stage=use_two_stage
    )
    
    # Load state dict (strict=False for two-stage, as we have new heads)
    if use_two_stage:
        # Load shared layers, initialize new heads randomly
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                          if k in model_dict and 'presence_head' not in k and 'proportion_head' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        if verbose:
            print(f"\n✓ Loaded {len(pretrained_dict)}/{len(model_dict)} layers from checkpoint")
            print(f"  New heads initialized randomly: presence_head, proportion_head")
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if verbose:
            print(f"\n✓ Successfully loaded all weights")
    
    if verbose:
        print("="*80)
    
    return model.to(device)


def test_model():
    """Test two-stage deconvolution model"""
    print("\nTesting Two-Stage TissueBERT Deconvolution Model...")
    
    model = TissueBERTDeconvolution(
        vocab_size=69,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=2048,
        max_position_embeddings=150,
        num_classes=22,
        dropout=0.1,
        n_regions=51089,
        use_two_stage=True,
        presence_threshold=0.5
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
    model.eval()
    with torch.no_grad():
        proportions, presence_probs = model(methylation, return_presence=True)
    
    print(f"\nOutput shapes:")
    print(f"  Proportions: {proportions.shape}")
    print(f"  Presence probs: {presence_probs.shape}")
    
    print(f"\nProportion checks:")
    for i in range(batch_size):
        prop_sum = proportions[i].sum().item()
        n_present = (presence_probs[i] > 0.5).sum().item()
        print(f"  Sample {i}: sum={prop_sum:.6f}, tissues_present={n_present}")
    
    # Verify all sum to 1.0
    sums = proportions.sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), \
        f"Proportions don't sum to 1.0: {sums}"
    
    # Test loss function
    true_props = torch.randn(batch_size, 22).abs()
    true_props = true_props / true_props.sum(dim=1, keepdim=True)
    true_presence = compute_true_presence(true_props, threshold=0.01)
    
    loss, loss_dict = mixture_loss_with_sparsity(
        proportions, true_props,
        presence_probs, true_presence
    )
    
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n✓ Two-stage deconvolution model test passed!")


if __name__ == '__main__':
    test_model()
