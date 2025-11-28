#!/usr/bin/env python3
"""
DIAGNOSTIC 2: Augmentation Leakage & Fine-grained Tissue Separability
======================================================================

This script checks:
1. Whether augmented versions of the same sample leak across train/val/test splits
2. Whether fine-grained tissue types are more separable than broad categories
3. Computes a "separability score" comparing the two classification schemes

Run: python diagnostic_2_augmentation_and_separability.py
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
HDF5_PATH = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset_chr1.h5"
TRAIN_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/train_files.csv"
VAL_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/val_files.csv"
TEST_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/test_files.csv"

# How many regions to sample for separability analysis
N_REGIONS_SAMPLE = 200

print("=" * 80)
print("DIAGNOSTIC 2: AUGMENTATION LEAKAGE & TISSUE SEPARABILITY")
print("=" * 80)

# ============================================================================
# PART 1: AUGMENTATION LEAKAGE CHECK
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: AUGMENTATION LEAKAGE CHECK")
print("=" * 80)
print("\nChecking if augmented versions of the same sample leak across splits...")

# Load all CSVs
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"\nLoaded {len(all_df)} files")
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Extract original sample name (without aug version)
# original_sample column should have this, but let's verify
if 'original_sample' in all_df.columns:
    all_df['base_sample'] = all_df['original_sample']
else:
    # Extract from sample_name by removing _augN suffix
    all_df['base_sample'] = all_df['sample_name'].str.replace(r'_aug\d+$', '', regex=True)

# Group by base sample and check splits
sample_splits = all_df.groupby('base_sample')['split'].apply(set).reset_index()
sample_splits['n_splits'] = sample_splits['split'].apply(len)
sample_splits['splits_str'] = sample_splits['split'].apply(lambda x: ','.join(sorted(x)))

# Find leakers
leakers = sample_splits[sample_splits['n_splits'] > 1]

print(f"\n" + "-" * 80)
print("AUGMENTATION LEAKAGE RESULTS")
print("-" * 80)

if len(leakers) == 0:
    print(f"\n✓ NO LEAKAGE: All augmented versions of each sample are in the same split")
    print(f"  {len(sample_splits)} unique base samples, all contained within single splits")
else:
    print(f"\n✗ LEAKAGE DETECTED!")
    print(f"  {len(leakers)} samples have augmentations across multiple splits")
    print(f"  This causes data leakage and invalidates your validation metrics!")
    
    print(f"\nLeakage breakdown:")
    leak_summary = leakers.groupby('splits_str').size().sort_values(ascending=False)
    for splits, count in leak_summary.items():
        print(f"  {splits}: {count} samples")
    
    print(f"\nFirst 10 leaking samples:")
    for idx, row in leakers.head(10).iterrows():
        print(f"  {row['base_sample']}: in {row['splits_str']}")
        
        # Show which files are in which split
        sample_files = all_df[all_df['base_sample'] == row['base_sample']]
        for _, file_row in sample_files.iterrows():
            print(f"    - {file_row['filename']} -> {file_row['split']}")

# Show split statistics by original sample
print(f"\n" + "-" * 80)
print("SPLIT COMPOSITION")
print("-" * 80)

for split_name in ['train', 'val', 'test']:
    split_df = all_df[all_df['split'] == split_name]
    n_unique_samples = split_df['base_sample'].nunique()
    n_files = len(split_df)
    augs_per_sample = n_files / n_unique_samples if n_unique_samples > 0 else 0
    
    print(f"\n{split_name.upper()}:")
    print(f"  Files: {n_files}")
    print(f"  Unique base samples: {n_unique_samples}")
    print(f"  Avg augmentations per sample: {augs_per_sample:.1f}")

# ============================================================================
# PART 2: FINE-GRAINED VS BROAD TISSUE SEPARABILITY
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: FINE-GRAINED VS BROAD TISSUE SEPARABILITY")
print("=" * 80)
print("\nComparing classification difficulty between broad and fine-grained labels...")

# Create fine-grained labels (tissue_type without trailing number)
# Handle NaN values first
all_df['tissue_type'] = all_df['tissue_type'].fillna(all_df['tissue'])
all_df['tissue_fine'] = all_df['tissue_type'].astype(str).str.replace(r'_\d+$', '', regex=True)

# Get unique labels
broad_tissues = sorted(all_df['tissue'].dropna().unique())
fine_tissues = sorted(all_df['tissue_fine'].dropna().unique())

print(f"\nLabel counts:")
print(f"  Broad categories (tissue): {len(broad_tissues)}")
print(f"  Fine-grained types (tissue_fine): {len(fine_tissues)}")

# Show the mapping
print(f"\n" + "-" * 80)
print("BROAD TO FINE-GRAINED MAPPING")
print("-" * 80)

broad_to_fine_map = all_df.groupby('tissue')['tissue_fine'].apply(lambda x: sorted(set(x))).to_dict()

print(f"\n{'Broad Category':<25} {'# Subtypes':<12} {'Subtypes'}")
print("-" * 80)

high_diversity_tissues = []
for broad in sorted(broad_to_fine_map.keys()):
    fine_list = broad_to_fine_map[broad]
    n_subtypes = len(fine_list)
    
    subtypes_str = ', '.join(fine_list[:3])
    if len(fine_list) > 3:
        subtypes_str += f" (+{len(fine_list)-3} more)"
    
    marker = "⚠️" if n_subtypes > 3 else "  "
    print(f"{marker} {broad:<23} {n_subtypes:<12} {subtypes_str}")
    
    if n_subtypes > 3:
        high_diversity_tissues.append((broad, n_subtypes))

if high_diversity_tissues:
    print(f"\n⚠️  HIGH DIVERSITY WARNING:")
    print(f"   The following broad categories contain many distinct subtypes:")
    for tissue, n in high_diversity_tissues:
        print(f"   - {tissue}: {n} subtypes")
    print(f"   This diversity may make the broad category too heterogeneous to learn!")

# ============================================================================
# PART 3: METHYLATION-BASED SEPARABILITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: METHYLATION-BASED SEPARABILITY ANALYSIS")
print("=" * 80)
print(f"\nComputing separability scores using methylation patterns...")
print(f"(Sampling {N_REGIONS_SAMPLE} regions)")

# Load methylation data
with h5py.File(HDF5_PATH, 'r') as f:
    hdf5_filenames = [fn.decode() for fn in f['metadata/filenames'][:]]
    n_files = f['methylation'].shape[0]
    n_regions = f['methylation'].shape[1]
    seq_len = f['methylation'].shape[2]
    
    print(f"\nHDF5 shape: ({n_files} files, {n_regions} regions, {seq_len} bp)")
    
    # Sample random regions
    np.random.seed(42)
    region_indices = np.random.choice(n_regions, min(N_REGIONS_SAMPLE, n_regions), replace=False)
    region_indices = np.sort(region_indices)
    
    # Create file-level feature vectors (mean methylation per region)
    print(f"\nExtracting features from {len(region_indices)} regions...")
    
    file_features = np.zeros((n_files, len(region_indices)))
    
    for i, region_idx in enumerate(region_indices):
        region_data = f['methylation'][:, region_idx, :]
        
        for file_idx in range(n_files):
            meth_data = region_data[file_idx, :]
            valid_mask = meth_data < 2
            if valid_mask.sum() > 0:
                file_features[file_idx, i] = meth_data[valid_mask].mean()
            else:
                file_features[file_idx, i] = 0.5  # Default if all missing
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(region_indices)} regions...")

# Create filename to index mapping
filename_to_hdf5_idx = {fn: idx for idx, fn in enumerate(hdf5_filenames)}

# Get labels for each file (in HDF5 order)
file_broad_labels = []
file_fine_labels = []
file_valid = []

for filename in hdf5_filenames:
    if filename in all_df['filename'].values:
        row = all_df[all_df['filename'] == filename].iloc[0]
        file_broad_labels.append(row['tissue'])
        file_fine_labels.append(row['tissue_fine'])
        file_valid.append(True)
    else:
        file_broad_labels.append('UNKNOWN')
        file_fine_labels.append('UNKNOWN')
        file_valid.append(False)

file_valid = np.array(file_valid)
valid_features = file_features[file_valid]
valid_broad = [l for l, v in zip(file_broad_labels, file_valid) if v]
valid_fine = [l for l, v in zip(file_fine_labels, file_valid) if v]

print(f"\nFiles with labels: {sum(file_valid)}/{n_files}")

# Convert string labels to integers
broad_label_map = {l: i for i, l in enumerate(sorted(set(valid_broad)))}
fine_label_map = {l: i for i, l in enumerate(sorted(set(valid_fine)))}

broad_labels_int = np.array([broad_label_map[l] for l in valid_broad])
fine_labels_int = np.array([fine_label_map[l] for l in valid_fine])

print(f"Broad categories: {len(broad_label_map)}")
print(f"Fine-grained types: {len(fine_label_map)}")

# Compute silhouette scores (measure of cluster quality)
print(f"\n" + "-" * 80)
print("SILHOUETTE SCORE ANALYSIS")
print("-" * 80)
print("\nSilhouette score measures how similar samples are to their own cluster")
print("vs. other clusters. Range: [-1, 1], higher is better.")
print("Score > 0.5: Strong separation")
print("Score 0.25-0.5: Moderate separation")
print("Score < 0.25: Poor separation (overlapping clusters)")

# PCA for dimensionality reduction (silhouette is expensive with high dims)
print(f"\nReducing dimensions with PCA...")
pca = PCA(n_components=min(50, valid_features.shape[1]))
features_pca = pca.fit_transform(valid_features)
var_explained = pca.explained_variance_ratio_.sum()
print(f"  PCA: {features_pca.shape[1]} components, {var_explained*100:.1f}% variance explained")

# Compute silhouette scores
try:
    sil_broad = silhouette_score(features_pca, broad_labels_int)
    print(f"\nSilhouette Score (BROAD categories): {sil_broad:.4f}")
except Exception as e:
    sil_broad = None
    print(f"\nCould not compute broad silhouette: {e}")

try:
    sil_fine = silhouette_score(features_pca, fine_labels_int)
    print(f"Silhouette Score (FINE-GRAINED types): {sil_fine:.4f}")
except Exception as e:
    sil_fine = None
    print(f"Could not compute fine silhouette: {e}")

if sil_broad is not None and sil_fine is not None:
    print(f"\n" + "-" * 80)
    print("SEPARABILITY COMPARISON")
    print("-" * 80)
    
    if sil_fine > sil_broad:
        improvement = (sil_fine - sil_broad) / abs(sil_broad) * 100 if sil_broad != 0 else float('inf')
        print(f"\n✓ Fine-grained types are MORE SEPARABLE than broad categories")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"\n  RECOMMENDATION: Train on fine-grained tissue types!")
        print(f"  The model will learn better with cleaner class boundaries.")
    else:
        print(f"\n  Broad categories are as separable or more separable than fine-grained")
        print(f"  The issue may lie elsewhere (learning rate, architecture, etc.)")

# ============================================================================
# PART 4: PER-TISSUE SEPARABILITY
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: PER-BROAD-CATEGORY INTERNAL COHERENCE")
print("=" * 80)
print("\nChecking which broad categories have the most internal diversity...")

broad_coherence = {}

for broad_tissue in sorted(set(valid_broad)):
    # Get indices for this broad tissue
    mask = np.array([l == broad_tissue for l in valid_broad])
    tissue_features = features_pca[mask]
    tissue_fine_labels = [l for l, m in zip(valid_fine, mask) if m]
    
    if len(tissue_features) < 2:
        continue
    
    # Compute internal variance (lower = more coherent)
    internal_var = np.var(tissue_features, axis=0).mean()
    
    # Count fine-grained subtypes
    n_subtypes = len(set(tissue_fine_labels))
    
    # If multiple subtypes, compute silhouette within this broad category
    if n_subtypes > 1:
        fine_labels_local = [fine_label_map[l] for l in tissue_fine_labels]
        try:
            sil_internal = silhouette_score(tissue_features, fine_labels_local)
        except:
            sil_internal = 0.0
    else:
        sil_internal = 1.0  # Perfect if only one subtype
    
    broad_coherence[broad_tissue] = {
        'n_samples': len(tissue_features),
        'n_subtypes': n_subtypes,
        'internal_var': internal_var,
        'subtype_separability': sil_internal
    }

print(f"\n{'Broad Tissue':<25} {'Samples':<10} {'Subtypes':<10} {'Internal Var':<15} {'Subtype Sep':<15}")
print("-" * 80)

# Sort by internal variance (highest = most problematic)
sorted_tissues = sorted(broad_coherence.items(), key=lambda x: -x[1]['internal_var'])

for tissue, stats in sorted_tissues:
    var_marker = "⚠️" if stats['internal_var'] > np.median([s['internal_var'] for s in broad_coherence.values()]) else "  "
    sep_marker = "⚠️" if stats['subtype_separability'] > 0.3 and stats['n_subtypes'] > 1 else "  "
    
    print(f"{var_marker}{tissue:<23} {stats['n_samples']:<10} {stats['n_subtypes']:<10} "
          f"{stats['internal_var']:<15.4f} {sep_marker}{stats['subtype_separability']:<15.4f}")

# Identify most problematic broad categories
problematic = [(t, s) for t, s in broad_coherence.items() 
               if s['n_subtypes'] > 2 and s['subtype_separability'] > 0.25]

if problematic:
    print(f"\n⚠️  PROBLEMATIC BROAD CATEGORIES:")
    print(f"   These have multiple distinct subtypes that the model is forced to treat as one:")
    for tissue, stats in problematic:
        print(f"   - {tissue}: {stats['n_subtypes']} subtypes, internal separability={stats['subtype_separability']:.3f}")
    print(f"\n   When subtypes within a broad category are themselves separable,")
    print(f"   forcing them into one class creates contradictory training signal!")

# ============================================================================
# FINAL DIAGNOSIS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL DIAGNOSIS")
print("=" * 80)

issues_found = []

if len(leakers) > 0:
    issues_found.append(f"AUGMENTATION_LEAKAGE: {len(leakers)} samples have augs across splits")

if sil_broad is not None and sil_broad < 0.25:
    issues_found.append(f"POOR_BROAD_SEPARATION: Silhouette={sil_broad:.3f} (< 0.25 threshold)")

if sil_fine is not None and sil_broad is not None and sil_fine > sil_broad + 0.05:
    issues_found.append(f"FINE_MORE_SEPARABLE: Fine-grained ({sil_fine:.3f}) > Broad ({sil_broad:.3f})")

if len(problematic) > 0:
    issues_found.append(f"INTERNAL_HETEROGENEITY: {len(problematic)} broad categories have distinct subtypes")

if len(issues_found) == 0:
    print("\n✓ No major issues found in augmentation or separability")
else:
    print(f"\n✗ ISSUES FOUND ({len(issues_found)}):")
    for issue in issues_found:
        print(f"  - {issue}")

print("\n" + "-" * 80)
print("RECOMMENDATIONS")
print("-" * 80)

if len(leakers) > 0:
    print("\n1. FIX AUGMENTATION LEAKAGE:")
    print("   - Re-split your data ensuring all augmentations of a sample stay together")
    print("   - Group by 'original_sample' before splitting into train/val/test")

if sil_fine is not None and sil_broad is not None and sil_fine > sil_broad:
    print("\n2. USE FINE-GRAINED LABELS:")
    print("   - Modify your HDF5 or dataloader to use tissue_type instead of tissue")
    print("   - This gives the model cleaner, more separable targets")
    print("   - You can always map predictions back to broad categories post-hoc")

if len(problematic) > 0:
    print("\n3. CONSIDER TWO-STAGE CLASSIFICATION:")
    print("   - First classify into broad categories")
    print("   - Then classify into fine-grained types within each category")
    print("   - Or just use fine-grained labels directly (recommended)")

print("\n" + "=" * 80)
print("DIAGNOSTIC 2 COMPLETE")
print("=" * 80)
