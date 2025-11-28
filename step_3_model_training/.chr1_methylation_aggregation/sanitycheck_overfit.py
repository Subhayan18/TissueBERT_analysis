#!/usr/bin/env python3
"""
SANITY CHECK: Can TissueBERT overfit 10 samples?
=================================================

If the model can't memorize 10 samples, there's a fundamental issue with:
- Gradient flow
- Model architecture
- Initialization
- Loss function

This script:
1. Takes 10 samples (5 from each of 2 classes)
2. Trains with high LR until 100% accuracy
3. Reports if it works or not

Run: python sanity_check_overfit.py
"""

import sys
sys.path.append('/home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/step2.2')

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from model_methylation_only import TissueBERT

print("=" * 80)
print("SANITY CHECK: Can TissueBERT Overfit 10 Samples?")
print("=" * 80)

# ============================================================================
# LOAD 10 SAMPLES
# ============================================================================
HDF5_PATH = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset_chr1.h5"

print("\nLoading 10 samples (5 from each of 2 classes)...")

with h5py.File(HDF5_PATH, 'r') as f:
    tissue_labels = f['tissue_labels'][:]
    
    # Find 2 classes with enough samples
    unique_tissues = np.unique(tissue_labels)
    class_0 = unique_tissues[0]
    class_1 = unique_tissues[1]
    
    # Get indices for each class - using specific file indices
    idx_class_0 = np.array([0, 1, 2, 3, 4])  # First 5 files
    idx_class_1 = np.array([10, 11, 12, 13, 14])  # Files 10-14
    
    # Get actual classes from these indices
    class_0 = tissue_labels[idx_class_0[0]]
    class_1 = tissue_labels[idx_class_1[0]]
    
    all_indices = np.concatenate([idx_class_0, idx_class_1])
    
    print(f"\nSelected 10 samples:")
    print(f"  Class {class_0}: indices {idx_class_0}")
    print(f"  Class {class_1}: indices {idx_class_1}")
    
    # Load data
    dna_tokens = f['dna_tokens'][all_indices]  # [10, n_regions, 150]
    methylation = f['methylation'][all_indices]  # [10, n_regions, 150]
    labels = tissue_labels[all_indices]  # [10]
    
    n_regions = dna_tokens.shape[1]
    
    # Take only first 100 regions for speed
    dna_tokens = dna_tokens[:, :100, :]
    methylation = methylation[:, :100, :]
    
    print(f"\nData shapes:")
    print(f"  DNA tokens: {dna_tokens.shape}")
    print(f"  Methylation: {methylation.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Labels: {labels}")

# Flatten to regions (10 files × 100 regions = 1000 samples)
dna_flat = dna_tokens.reshape(-1, 150)  # [1000, 150]
meth_flat = methylation.reshape(-1, 150)  # [1000, 150]
labels_flat = np.repeat(labels, 100)  # [1000]

print(f"\nFlattened to regions:")
print(f"  Total regions: {len(labels_flat)}")
print(f"  Class {class_0}: {(labels_flat == class_0).sum()} regions")
print(f"  Class {class_1}: {(labels_flat == class_1).sum()} regions")

# Convert to tensors
X_dna = torch.tensor(dna_flat, dtype=torch.long)
X_meth = torch.tensor(meth_flat, dtype=torch.long)
y = torch.tensor(labels_flat, dtype=torch.long)

# ============================================================================
# CREATE TINY MODEL
# ============================================================================
print("\n" + "=" * 80)
print("Creating Model")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = TissueBERT(
    vocab_size=69,
    hidden_size=128,  # Smaller for speed
    num_hidden_layers=2,  # Fewer layers
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=150,
    num_classes=22,  # Keep all classes
    dropout=0.0  # NO DROPOUT for overfitting test
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ============================================================================
# TRAIN
# ============================================================================
print("\n" + "=" * 80)
print("Training to Overfit")
print("=" * 80)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

batch_size = 100
max_epochs = 200
target_acc = 0.99

print(f"\nSettings:")
print(f"  Learning rate: 1e-3")
print(f"  Batch size: {batch_size}")
print(f"  Max epochs: {max_epochs}")
print(f"  Target accuracy: {target_acc*100}%")

X_dna = X_dna.to(device)
X_meth = X_meth.to(device)
y = y.to(device)

best_acc = 0
converged = False

for epoch in range(max_epochs):
    model.train()
    
    # Shuffle
    perm = torch.randperm(len(y))
    X_dna_shuffled = X_dna[perm]
    X_meth_shuffled = X_meth[perm]
    y_shuffled = y[perm]
    
    total_loss = 0
    correct = 0
    n_batches = 0
    
    for i in range(0, len(y), batch_size):
        batch_dna = X_dna_shuffled[i:i+batch_size]
        batch_meth = X_meth_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        
        optimizer.zero_grad()
        
        logits = model(batch_dna, batch_meth)
        loss = criterion(logits, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    acc = correct / len(y)
    
    if acc > best_acc:
        best_acc = acc
    
    if epoch % 10 == 0 or acc > 0.9:
        print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Acc={acc:.4f} ({acc*100:.1f}%)")
    
    if acc >= target_acc:
        print(f"\n✓ CONVERGED at epoch {epoch}!")
        print(f"  Final accuracy: {acc*100:.2f}%")
        converged = True
        break

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

if converged:
    print(f"\n✓ SUCCESS: Model can overfit!")
    print(f"  Reached {target_acc*100}% accuracy in {epoch} epochs")
    print(f"\n  INTERPRETATION:")
    print(f"  - Gradients are flowing correctly")
    print(f"  - Model architecture is sound")
    print(f"  - Issue is likely with:")
    print(f"    * Training dynamics on full dataset")
    print(f"    * Class imbalance handling")
    print(f"    * Learning rate schedule")
    print(f"    * Batch sampling strategy")
else:
    print(f"\n✗ FAILURE: Model CANNOT overfit {len(y)} samples!")
    print(f"  Best accuracy: {best_acc*100:.2f}% (after {max_epochs} epochs)")
    print(f"\n  INTERPRETATION:")
    print(f"  - Fundamental issue with model or training")
    print(f"  - Possible causes:")
    print(f"    * Gradients not reaching embeddings")
    print(f"    * Bad initialization")
    print(f"    * Model output stuck at constant values")
    
    # Debug: Check if model outputs are diverse
    model.eval()
    with torch.no_grad():
        all_logits = model(X_dna[:batch_size], X_meth[:batch_size])
        all_probs = F.softmax(all_logits, dim=1)
        
        print(f"\n  Debug - Model outputs (first batch):")
        print(f"    Logits range: [{all_logits.min().item():.4f}, {all_logits.max().item():.4f}]")
        print(f"    Logits std: {all_logits.std().item():.4f}")
        print(f"    Predictions: {all_logits.argmax(dim=1).cpu().numpy()}")
        print(f"    True labels: {y[:batch_size].cpu().numpy()}")
        
        # Check if model predicts same class for everything
        preds = all_logits.argmax(dim=1).cpu().numpy()
        unique_preds = len(set(preds))
        print(f"    Unique predictions: {unique_preds}")
        
        if unique_preds == 1:
            print(f"\n    ✗ Model predicts ONLY class {preds[0]} for all samples!")
            print(f"       This is mode collapse - model output is stuck")

print("\n" + "=" * 80)
