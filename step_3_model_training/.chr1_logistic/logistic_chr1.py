#!/usr/bin/env python3
"""
BASELINE: Logistic Regression on Methylation Data
==================================================

Tests if the methylation data is learnable with a simple model.
If logistic regression works but TissueBERT doesn't, the problem is in the model.
If logistic regression also fails, there's a deeper data issue.

This script:
1. Extracts features from the HDF5 (mean methylation per region)
2. Trains logistic regression on train split
3. Evaluates on val and test splits
4. Reports accuracy, confusion matrix, and per-class metrics

Run: python baseline_logistic_regression.py
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    top_k_accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
HDF5_PATH = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset_chr1.h5"
TRAIN_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/train_files.csv"
VAL_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/val_files.csv"
TEST_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/test_files.csv"

# Feature extraction settings
# We'll compute mean methylation per region as features
# This gives us n_regions features per file
MAX_REGIONS = 1000  # Limit regions for speed (set to None for all)

print("=" * 80)
print("BASELINE: LOGISTIC REGRESSION ON METHYLATION DATA")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

# Load CSVs
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)} files")
print(f"  Val:   {len(val_df)} files")
print(f"  Test:  {len(test_df)} files")

# Load HDF5
with h5py.File(HDF5_PATH, 'r') as f:
    hdf5_filenames = [fn.decode() for fn in f['metadata/filenames'][:]]
    hdf5_tissue_labels = f['tissue_labels'][:]
    n_files = f['methylation'].shape[0]
    n_regions = f['methylation'].shape[1]
    seq_len = f['methylation'].shape[2]
    
    print(f"\nHDF5 info:")
    print(f"  Files: {n_files}")
    print(f"  Regions: {n_regions}")
    print(f"  Seq length: {seq_len}")
    print(f"  Unique labels: {len(np.unique(hdf5_tissue_labels))}")

# Create filename to index mapping
filename_to_idx = {fn: idx for idx, fn in enumerate(hdf5_filenames)}

# Get indices for each split
def get_indices(df):
    indices = []
    labels = []
    for _, row in df.iterrows():
        fn = row['filename']
        if fn in filename_to_idx:
            idx = filename_to_idx[fn]
            indices.append(idx)
            labels.append(hdf5_tissue_labels[idx])
    return np.array(indices), np.array(labels)

train_indices, train_labels = get_indices(train_df)
val_indices, val_labels = get_indices(val_df)
test_indices, test_labels = get_indices(test_df)

print(f"\nMatched files:")
print(f"  Train: {len(train_indices)}")
print(f"  Val:   {len(val_indices)}")
print(f"  Test:  {len(test_indices)}")

# ============================================================================
# EXTRACT FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXTRACTING FEATURES")
print("=" * 80)

# Determine number of regions to use
if MAX_REGIONS is not None and MAX_REGIONS < n_regions:
    np.random.seed(42)
    region_indices = np.sort(np.random.choice(n_regions, MAX_REGIONS, replace=False))
    print(f"\nUsing {MAX_REGIONS} randomly sampled regions (out of {n_regions})")
else:
    region_indices = np.arange(n_regions)
    print(f"\nUsing all {n_regions} regions")

n_features = len(region_indices)

def extract_features(file_indices, desc=""):
    """Extract mean methylation per region for each file."""
    n_files = len(file_indices)
    features = np.zeros((n_files, n_features))
    
    with h5py.File(HDF5_PATH, 'r') as f:
        for i, file_idx in enumerate(file_indices):
            # Get methylation for this file, selected regions
            # Shape: (n_regions, seq_len)
            meth_data = f['methylation'][file_idx, region_indices, :]
            
            # Compute mean methylation per region (ignoring missing=2)
            for j in range(n_features):
                region_meth = meth_data[j, :]
                valid_mask = region_meth < 2  # 0=unmeth, 1=meth, 2=missing
                if valid_mask.sum() > 0:
                    features[i, j] = region_meth[valid_mask].mean()
                else:
                    features[i, j] = 0.5  # Default if all missing
            
            if (i + 1) % 100 == 0:
                print(f"  {desc}: Processed {i + 1}/{n_files} files...")
    
    return features

print("\nExtracting features...")
print("  (Each feature = mean methylation rate for one genomic region)")

X_train = extract_features(train_indices, "Train")
print(f"  Train features shape: {X_train.shape}")

X_val = extract_features(val_indices, "Val")
print(f"  Val features shape: {X_val.shape}")

X_test = extract_features(test_indices, "Test")
print(f"  Test features shape: {X_test.shape}")

# ============================================================================
# PREPROCESS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: PREPROCESSING")
print("=" * 80)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeature statistics (after scaling):")
print(f"  Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print(f"  Val mean: {X_val_scaled.mean():.4f}, std: {X_val_scaled.std():.4f}")
print(f"  Test mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")

# Check label distribution
print(f"\nLabel distribution:")
print(f"  Train: {dict(sorted(Counter(train_labels).items()))}")
print(f"  Val: {dict(sorted(Counter(val_labels).items()))}")
print(f"  Test: {dict(sorted(Counter(test_labels).items()))}")

# ============================================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAINING LOGISTIC REGRESSION")
print("=" * 80)

# Try different regularization strengths
C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
best_val_acc = 0
best_model = None
best_C = None

print(f"\nTrying different regularization strengths (C)...")

for C in C_values:
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, train_labels)
    
    train_acc = accuracy_score(train_labels, model.predict(X_train_scaled))
    val_acc = accuracy_score(val_labels, model.predict(X_val_scaled))
    
    print(f"  C={C:6.3f}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_C = C

print(f"\nBest model: C={best_C} with Val Acc={best_val_acc:.4f}")

# ============================================================================
# EVALUATE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: EVALUATION")
print("=" * 80)

# Predictions
train_pred = best_model.predict(X_train_scaled)
val_pred = best_model.predict(X_val_scaled)
test_pred = best_model.predict(X_test_scaled)

# Probabilities for top-k accuracy
train_prob = best_model.predict_proba(X_train_scaled)
val_prob = best_model.predict_proba(X_val_scaled)
test_prob = best_model.predict_proba(X_test_scaled)

# Metrics
def evaluate_split(y_true, y_pred, y_prob, split_name):
    print(f"\n{'-' * 80}")
    print(f"{split_name.upper()} RESULTS")
    print(f"{'-' * 80}")
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Top-k accuracy
    n_classes = y_prob.shape[1]
    if n_classes >= 3:
        top3_acc = top_k_accuracy_score(y_true, y_prob, k=3, labels=range(n_classes))
        print(f"Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    if n_classes >= 5:
        top5_acc = top_k_accuracy_score(y_true, y_prob, k=5, labels=range(n_classes))
        print(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    
    # Random chance baseline
    random_chance = 1.0 / n_classes
    print(f"Random Chance: {random_chance:.4f} ({random_chance*100:.2f}%)")
    print(f"Improvement over random: {(acc - random_chance) / random_chance * 100:.1f}%")
    
    # Per-class report
    print(f"\nClassification Report:")
    unique_labels = sorted(set(y_true) | set(y_pred))
    print(classification_report(y_true, y_pred, labels=unique_labels, zero_division=0))
    
    # Confusion matrix summary
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    print(f"\nConfusion Matrix (top 5 most confused pairs):")
    
    # Find most confused pairs
    confused_pairs = []
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((label_i, label_j, cm[i, j]))
    
    confused_pairs.sort(key=lambda x: -x[2])
    for true_label, pred_label, count in confused_pairs[:5]:
        print(f"  True={true_label} -> Pred={pred_label}: {count} times")
    
    return acc

train_acc = evaluate_split(train_labels, train_pred, train_prob, "Train")
val_acc = evaluate_split(val_labels, val_pred, val_prob, "Validation")
test_acc = evaluate_split(test_labels, test_pred, test_prob, "Test")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

n_classes = len(np.unique(train_labels))
random_chance = 1.0 / n_classes

print(f"\n{'Metric':<25} {'Train':<12} {'Val':<12} {'Test':<12}")
print("-" * 60)
print(f"{'Accuracy':<25} {train_acc:<12.4f} {val_acc:<12.4f} {test_acc:<12.4f}")
print(f"{'Random Chance':<25} {random_chance:<12.4f} {random_chance:<12.4f} {random_chance:<12.4f}")

print(f"\n" + "-" * 80)
print("INTERPRETATION")
print("-" * 80)

if test_acc > random_chance * 2:
    print(f"\n✓ LOGISTIC REGRESSION WORKS!")
    print(f"  Test accuracy ({test_acc:.2%}) is significantly above random chance ({random_chance:.2%})")
    print(f"\n  CONCLUSION: The data IS learnable.")
    print(f"  If TissueBERT fails but logistic regression works,")
    print(f"  the problem is in the transformer model or training setup.")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Check if TissueBERT can overfit a tiny subset (10 samples)")
    print(f"  2. Increase learning rate (try 1e-4 or 1e-3)")
    print(f"  3. Verify input encoding is correct")
    print(f"  4. Check gradient flow through the model")
    
elif test_acc > random_chance * 1.2:
    print(f"\n⚠️  LOGISTIC REGRESSION SHOWS WEAK SIGNAL")
    print(f"  Test accuracy ({test_acc:.2%}) is slightly above random chance ({random_chance:.2%})")
    print(f"\n  CONCLUSION: The data has some signal but it's weak.")
    print(f"  A more powerful model might help, but also check:")
    print(f"  1. Feature engineering (use more regions, different aggregations)")
    print(f"  2. Data quality issues")
    print(f"  3. Whether the task is inherently very hard")
    
else:
    print(f"\n✗ LOGISTIC REGRESSION ALSO FAILS")
    print(f"  Test accuracy ({test_acc:.2%}) is near random chance ({random_chance:.2%})")
    print(f"\n  CONCLUSION: Either:")
    print(f"  1. The features extracted don't capture tissue-specific patterns")
    print(f"  2. There's a data issue we haven't found yet")
    print(f"  3. The methylation patterns in chr1 alone aren't tissue-specific")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Try on full dataset (all chromosomes)")
    print(f"  2. Visualize methylation patterns with PCA/t-SNE")
    print(f"  3. Check if the broad tissue labels are too heterogeneous")

print("\n" + "=" * 80)
print("BASELINE COMPLETE")
print("=" * 80)
