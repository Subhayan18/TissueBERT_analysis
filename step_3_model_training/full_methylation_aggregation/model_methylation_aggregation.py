#!/usr/bin/env python3
"""
Memory-Efficient File-Level MLP for Full Genome (51,089 regions)
=================================================================

Changes from chr1 version:
1. Fixed n_regions=51089 (no dynamic projection needed)
2. More aggressive dimensionality reduction
3. Memory-optimized aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TissueBERT(nn.Module):
    """
    File-Level MLP optimized for full genome (51,089 regions)
    
    Input: [batch, 51089, 150] methylation
    Process:
        1. Compute mean methylation per region â†' [batch, 51089]
        2. Project to hidden dimension â†' [batch, hidden_size]
        3. MLP classification â†' [batch, num_classes]
    """

    def __init__(self,
                 vocab_size=69,
                 hidden_size=512,
                 num_hidden_layers=3,
                 num_attention_heads=4,
                 intermediate_size=1024,
                 max_position_embeddings=150,
                 num_classes=22,
                 dropout=0.1,
                 n_regions=51089):  # Fixed for full genome
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.n_regions = n_regions

        # Fixed input projection: 51089 â†' hidden_size
        # This is the memory bottleneck: 51089 x 512 = 26M parameters
        self.input_projection = nn.Linear(n_regions, hidden_size)
        
        # MLP network
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
            
            # Output
            nn.Linear(intermediate_size, num_classes)
        )
        
        self.apply(self._init_weights)
        
        # Calculate parameter count
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 70)
        print("FULL GENOME FILE-LEVEL MLP")
        print("=" * 70)
        print(f"Input: [batch, {n_regions:,} regions, 150 bp] methylation")
        print(f"Process: Region aggregation â†' MLP â†' Classification")
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

    def forward(self, dna_tokens, methylation_status):
        """
        Forward pass with memory-efficient aggregation
        
        Args:
            dna_tokens: [batch, n_regions, seq_length] - IGNORED
            methylation_status: [batch, n_regions, seq_length]
            
        Returns:
            logits: [batch, num_classes]
        """
        batch_size = methylation_status.shape[0]
        n_regions = methylation_status.shape[1]
        seq_length = methylation_status.shape[2]
        
        # Verify shape
        assert n_regions == self.n_regions, \
            f"Expected {self.n_regions} regions, got {n_regions}"
        
        # ===================================================================
        # STEP 1: Compute mean methylation per region (memory-efficient)
        # ===================================================================
        
        # Process in-place to save memory
        # Shape: [batch, n_regions, seq_length]
        
        # Mask missing values (2) and convert to float
        valid_mask = (methylation_status != 2).float()  # [batch, n_regions, seq_length]
        meth_float = methylation_status.float()
        meth_float = torch.where(methylation_status == 2, torch.zeros_like(meth_float), meth_float)
        
        # Compute mean per region
        # Sum over sequence length, divide by valid count
        region_sums = (meth_float * valid_mask).sum(dim=2)  # [batch, n_regions]
        region_counts = valid_mask.sum(dim=2).clamp(min=1.0)  # [batch, n_regions]
        region_means = region_sums / region_counts  # [batch, n_regions]
        
        # ===================================================================
        # STEP 2: Project to hidden dimension
        # ===================================================================
        
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
    """Test model with full genome dimensions"""
    print("\nTesting Full Genome File-Level MLP...")

    model = TissueBERT(
        vocab_size=69,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=1024,
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
    dna_tokens = torch.randint(0, 65, (batch_size, n_regions, seq_length))
    methylation = torch.randint(0, 3, (batch_size, n_regions, seq_length))

    # Test memory usage
    import torch.cuda as cuda
    if cuda.is_available():
        model = model.cuda()
        dna_tokens = dna_tokens.cuda()
        methylation = methylation.cuda()
        
        cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            logits = model(dna_tokens, methylation)
        
        peak_mem = cuda.max_memory_allocated() / 1e9
        print(f"\nPeak GPU memory: {peak_mem:.2f} GB")
    else:
        with torch.no_grad():
            logits = model(dna_tokens, methylation)
        print("\nCPU mode (no GPU memory measurement)")

    print(f"\nOutput shape: {logits.shape}")
    assert logits.shape == (batch_size, model.num_classes)

    probs = F.softmax(logits, dim=1)
    print(f"\nProbability check (sample 0):")
    print(f"  Sum: {probs[0].sum():.6f}")
    print(f"  Max: {probs[0].max():.6f}")
    print(f"  Predicted class: {probs[0].argmax().item()}")

    print("\n[OK] Full genome model test passed!")


if __name__ == '__main__':
    test_model()
