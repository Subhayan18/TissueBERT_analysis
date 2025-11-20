#!/usr/bin/env python3
"""
TissueBERT Model Architecture
DNABERT-S based transformer for tissue classification from methylation patterns

Based on DNABERT (Ji et al., 2021) and MethylBERT (Titus et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        
        assert hidden_size % num_attention_heads == 0
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Q, K, V projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x):
        """Reshape for multi-head attention"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # Project Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        return context_layer


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Feed-forward
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + ffn_output)
        
        return hidden_states


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers"""
    
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads, 
                 intermediate_size, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class TissueBERT(nn.Module):
    """
    TissueBERT: DNABERT-S based model for tissue classification
    
    Architecture:
    1. DNA sequence embedding (3-mer tokens)
    2. Methylation status embedding
    3. Position embedding
    4. Transformer encoder (6 layers)
    5. Pooling + classification head
    
    Args:
        vocab_size (int): Size of DNA token vocabulary (default: 69)
        hidden_size (int): Hidden dimension size (default: 512)
        num_hidden_layers (int): Number of transformer layers (default: 6)
        num_attention_heads (int): Number of attention heads (default: 8)
        intermediate_size (int): FFN intermediate dimension (default: 2048)
        max_position_embeddings (int): Maximum sequence length (default: 150)
        num_classes (int): Number of tissue classes (default: 22)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, 
                 vocab_size=69,
                 hidden_size=512,
                 num_hidden_layers=6,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 max_position_embeddings=150,
                 num_classes=22,
                 dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Embeddings
        # DNA token embedding (0-64 = 3-mers, 64-68 = special tokens)
        self.dna_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=64)
        
        # Methylation status embedding (0=unmethylated, 1=methylated, 2=missing/no_CpG)
        self.meth_embedding = nn.Embedding(3, hidden_size)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Embedding normalization and dropout
        self.embedding_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout
        )
        
        # Pooling
        self.pooling = nn.Linear(hidden_size, hidden_size)
        self.pooling_activation = nn.Tanh()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights (Xavier uniform for linear layers)"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, dna_tokens, methylation_status):
        """
        Forward pass
        
        Args:
            dna_tokens: [batch_size, seq_length] - DNA token IDs (0-68)
            methylation_status: [batch_size, seq_length] - Methylation status (0, 1, 2)
            
        Returns:
            logits: [batch_size, num_classes] - Classification logits
        """
        batch_size, seq_length = dna_tokens.shape
        
        # Create position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=dna_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        dna_embeds = self.dna_embedding(dna_tokens)
        meth_embeds = self.meth_embedding(methylation_status)
        pos_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings (sum)
        embeddings = dna_embeds + meth_embeds + pos_embeds
        
        # Normalize and dropout
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Create attention mask (mask padding tokens = 64)
        attention_mask = (dna_tokens != 64).unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=embeddings.dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Pass through transformer encoder
        hidden_states = self.encoder(embeddings, attention_mask)
        
        # Pool: use [CLS] token (first token) or mean pooling
        # Using mean pooling over non-padding tokens
        mask = (dna_tokens != 64).unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        # Apply pooling layer
        pooled = self.pooling(pooled)
        pooled = self.pooling_activation(pooled)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def get_attention_weights(self, dna_tokens, methylation_status):
        """
        Get attention weights for visualization
        
        Returns attention weights from all layers
        """
        # TODO: Implement attention weight extraction for visualization
        # This would be useful for interpreting which genomic regions
        # the model focuses on for each tissue type
        pass


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model with random inputs"""
    print("Testing TissueBERT model...")
    
    # Create model
    model = TissueBERT(
        vocab_size=69,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=150,
        num_classes=22,
        dropout=0.1
    )
    
    # Print model info
    print(f"Model: TissueBERT")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model size: {count_parameters(model) * 4 / 1e6:.2f} MB (fp32)")
    
    # Test forward pass
    batch_size = 4
    seq_length = 150
    
    dna_tokens = torch.randint(0, 65, (batch_size, seq_length))  # 0-64
    methylation = torch.randint(0, 3, (batch_size, seq_length))  # 0, 1, 2
    
    print(f"\nInput shapes:")
    print(f"  DNA tokens: {dna_tokens.shape}")
    print(f"  Methylation: {methylation.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(dna_tokens, methylation)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {model.num_classes})")
    
    # Check output
    assert logits.shape == (batch_size, model.num_classes)
    
    # Test with softmax
    probs = F.softmax(logits, dim=1)
    print(f"\nProbability distribution (sample 0):")
    print(f"  Sum: {probs[0].sum():.6f} (should be ~1.0)")
    print(f"  Max: {probs[0].max():.6f}")
    print(f"  Predicted class: {probs[0].argmax().item()}")
    
    print("\n✓ Model test passed!")


if __name__ == '__main__':
    test_model()
