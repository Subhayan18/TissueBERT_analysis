#!/usr/bin/env python3
"""
CNN + Attention Model for Tissue Classification
================================================

Input: [batch, n_regions, seq_length] methylation data
1. CNN encodes each region (uses all 150 positions)
2. Attention aggregates across regions
3. Classify tissue

Per-base modeling with file-level prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNAttentionTissueBERT(nn.Module):
    """
    CNN + Attention for File-Level Tissue Classification
    
    Architecture:
        1. Region Encoder (1D CNN): [150] → [region_embed_dim]
        2. Attention Aggregator: [n_regions, region_embed_dim] → [region_embed_dim]
        3. Classifier: [region_embed_dim] → [num_classes]
    """

    def __init__(self,
                 vocab_size=69,  # Kept for compatibility
                 hidden_size=512,
                 num_hidden_layers=3,  # Not used
                 num_attention_heads=8,
                 intermediate_size=1024,
                 max_position_embeddings=150,  # seq_length
                 num_classes=22,
                 dropout=0.1,
                 region_embed_dim=256,
                 region_chunk_size=512):  # Process regions in chunks to save memory
        super().__init__()

        self.num_classes = num_classes
        self.region_embed_dim = region_embed_dim
        self.seq_length = max_position_embeddings
        self.region_chunk_size = region_chunk_size  # Process this many regions at once
        
        # ===================================================================
        # Region Encoder: 1D CNN
        # ===================================================================
        # Input: [batch*n_regions, 2, 150] (2 channels: methylation value + valid mask)
        # Output: [batch*n_regions, region_embed_dim]
        
        self.region_encoder = nn.Sequential(
            # Conv1: [2, 150] → [64, 150] (2 channels: value + mask)
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),  # [64, 75]
            
            # Conv2: [64, 75] → [128, 75]
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),  # [128, 37]
            
            # Conv3: [128, 37] → [256, 37]
            nn.Conv1d(128, region_embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(region_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Global pooling: [256, 37] → [256, 1]
            nn.AdaptiveAvgPool1d(1)
        )
        
        # ===================================================================
        # Attention Aggregator across regions
        # ===================================================================
        
        self.attention = nn.MultiheadAttention(
            embed_dim=region_embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True  # Expect [batch, seq, dim]
        )
        
        self.attention_norm = nn.LayerNorm(region_embed_dim)
        
        # ===================================================================
        # Classifier
        # ===================================================================
        
        self.classifier = nn.Sequential(
            nn.Linear(region_embed_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, intermediate_size),
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(intermediate_size, num_classes)
        )
        
        self.apply(self._init_weights)
        
        print("=" * 70)
        print("CNN + ATTENTION MODEL FOR TISSUE CLASSIFICATION")
        print("=" * 70)
        print("Architecture:")
        print("  1. CNN Region Encoder: [2, 150] → [256] (2ch: value + mask)")
        print(f"  2. {num_attention_heads}-head Attention: Aggregate regions")
        print("  3. MLP Classifier: [256] → [num_classes]")
        print("=" * 70)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
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
        # STEP 1: Preprocess methylation with MISSING VALUE MASK
        # ===================================================================
        
        # Reshape: [batch, n_regions, seq_length] → [batch*n_regions, seq_length]
        meth_flat = methylation_status.reshape(-1, seq_length)
        
        # Convert to float
        meth_float = meth_flat.float()
        
        # Create TWO channels:
        # Channel 0: methylation value (0 or 1, missing becomes 0)
        meth_value = meth_float.clone()
        meth_value[meth_flat == 2] = 0
        
        # Channel 1: valid mask (1 if real data, 0 if missing)
        meth_mask = (meth_flat != 2).float()
        
        # Stack into 2-channel input: [batch*n_regions, 2, seq_length]
        meth_input = torch.stack([meth_value, meth_mask], dim=1)
        
        # ===================================================================
        # STEP 2: Encode each region with CNN (MEMORY-EFFICIENT CHUNKED PROCESSING)
        # ===================================================================
        
        # Process regions in chunks to avoid OOM
        # Instead of processing all 35k regions at once, do 512 at a time
        region_features_list = []
        
        for chunk_start in range(0, batch_size * n_regions, self.region_chunk_size):
            chunk_end = min(chunk_start + self.region_chunk_size, batch_size * n_regions)
            chunk = meth_input[chunk_start:chunk_end]
            
            # Encode this chunk
            chunk_features = self.region_encoder(chunk)  # [chunk_size, region_embed_dim, 1]
            chunk_features = chunk_features.squeeze(-1)  # [chunk_size, region_embed_dim]
            
            region_features_list.append(chunk_features)
        
        # Concatenate all chunks
        region_features = torch.cat(region_features_list, dim=0)  # [batch*n_regions, region_embed_dim]
        
        # Reshape back: [batch*n_regions, region_embed_dim] → [batch, n_regions, region_embed_dim]
        region_features = region_features.view(batch_size, n_regions, self.region_embed_dim)
        
        # ===================================================================
        # STEP 3: Attention aggregation across regions
        # ===================================================================
        
        # Self-attention
        attn_output, attn_weights = self.attention(
            region_features,  # query
            region_features,  # key
            region_features   # value
        )
        # attn_output: [batch, n_regions, region_embed_dim]
        # attn_weights: [batch, n_regions, n_regions]
        
        # Residual connection + norm
        region_features = self.attention_norm(region_features + attn_output)
        
        # Pool across regions (mean pooling)
        file_features = region_features.mean(dim=1)  # [batch, region_embed_dim]
        
        # ===================================================================
        # STEP 4: Classification
        # ===================================================================
        
        logits = self.classifier(file_features)  # [batch, num_classes]
        
        return logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model"""
    print("\nTesting CNN + Attention Model...")

    model = CNNAttentionTissueBERT(
        vocab_size=69,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=150,
        num_classes=22,
        dropout=0.1,
        region_embed_dim=256
    )

    print(f"\nModel: CNN + Attention")
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
