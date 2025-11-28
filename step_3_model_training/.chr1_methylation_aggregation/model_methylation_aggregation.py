#!/usr/bin/env python3
"""
File-Level MLP for Tissue Classification
=========================================

Processes ALL regions from a file, exactly like logistic regression.

Input: [batch, n_regions, seq_length] methylation data
1. Compute mean methylation per region
2. Feed all region means through MLP
3. Classify tissue

This should match logistic regression's 91% accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TissueBERT(nn.Module):
    """
    File-Level MLP - Processes all regions from a file
    
    ===================================================================
    MATCHES LOGISTIC REGRESSION EXACTLY
    ===================================================================
    
    Input: [batch, n_regions, seq_length] methylation
    Process:
        1. Compute mean methylation per region → [batch, n_regions]
        2. Pass through MLP
        3. Classify tissue
    
    This is what logistic regression did (1000 features = mean per region)
    ===================================================================
    """

    def __init__(self,
                 vocab_size=69,  # Kept for compatibility
                 hidden_size=512,
                 num_hidden_layers=3,
                 num_attention_heads=4,  # Not used
                 intermediate_size=1024,
                 max_position_embeddings=150,  # Not used
                 num_classes=22,
                 dropout=0.1):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Network processes n_regions features (one per region)
        # We don't know n_regions ahead of time, so we'll handle it dynamically
        
        # Input projection: n_regions → hidden_size
        # We'll use a linear layer that takes variable-length input
        # by reshaping in forward pass
        
        self.network = nn.Sequential(
            # NOTE: We'll project from n_regions to hidden in forward()
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
            
            # Output
            nn.Linear(intermediate_size, num_classes)
        )
        
        # Learnable projection from n_regions to hidden_size
        # Will be created dynamically based on n_regions
        self.input_projection = None
        
        self.apply(self._init_weights)
        
        print("=" * 70)
        print("FILE-LEVEL MLP FOR TISSUE CLASSIFICATION")
        print("=" * 70)
        print("Input: [batch, n_regions, seq_length] methylation")
        print("Process: Mean per region → MLP → Classification")
        print("Matches logistic regression approach")
        print("=" * 70)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, dna_tokens, methylation_status):
        """
        Forward pass
        
        Args:
            dna_tokens: [batch, n_regions, seq_length] - IGNORED
            methylation_status: [batch, n_regions, seq_length]
            
        Returns:
            logits: [batch, num_classes]
        """
        batch_size = methylation_status.shape[0]
        n_regions = methylation_status.shape[1]
        seq_length = methylation_status.shape[2]
        
        # ===================================================================
        # STEP 1: Compute mean methylation per region
        # ===================================================================
        
        # Reshape: [batch, n_regions, seq_length] → [batch*n_regions, seq_length]
        meth_flat = methylation_status.reshape(-1, seq_length)
        
        # Mask missing values (2)
        valid_mask = (meth_flat != 2).float()
        meth_float = meth_flat.float()
        meth_float[meth_flat == 2] = 0
        
        # Mean per region
        region_means = (meth_float * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        # Shape: [batch*n_regions]
        
        # Reshape back: [batch*n_regions] → [batch, n_regions]
        region_means = region_means.reshape(batch_size, n_regions)
        
        # ===================================================================
        # STEP 2: Project to hidden dimension
        # ===================================================================
        
        # Create input projection if needed
        if self.input_projection is None or self.input_projection.in_features != n_regions:
            self.input_projection = nn.Linear(n_regions, self.hidden_size).to(region_means.device)
            nn.init.xavier_uniform_(self.input_projection.weight)
            if self.input_projection.bias is not None:
                nn.init.constant_(self.input_projection.bias, 0)
        
        features = self.input_projection(region_means)  # [batch, hidden_size]
        
        # ===================================================================
        # STEP 3: MLP classification
        # ===================================================================
        
        logits = self.network(features)  # [batch, num_classes]
        
        return logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model"""
    print("\nTesting File-Level MLP...")

    model = TissueBERT(
        vocab_size=69,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=150,
        num_classes=22,
        dropout=0.1
    )

    print(f"\nModel: File-Level MLP")
    print(f"Parameters: {count_parameters(model):,}")

    batch_size = 4
    n_regions = 1000
    seq_length = 150

    dna_tokens = torch.randint(0, 65, (batch_size, n_regions, seq_length))
    methylation = torch.randint(0, 3, (batch_size, n_regions, seq_length))

    print(f"\nInput shapes:")
    print(f"  DNA tokens: {dna_tokens.shape} (ignored)")
    print(f"  Methylation: {methylation.shape}")

    with torch.no_grad():
        logits = model(dna_tokens, methylation)

    print(f"\nOutput shape: {logits.shape}")
    assert logits.shape == (batch_size, model.num_classes)

    probs = F.softmax(logits, dim=1)
    print(f"\nProbability distribution (sample 0):")
    print(f"  Sum: {probs[0].sum():.6f}")
    print(f"  Max: {probs[0].max():.6f}")
    print(f"  Predicted class: {probs[0].argmax().item()}")

    print("\n✓ Model test passed!")


if __name__ == '__main__':
    test_model()
