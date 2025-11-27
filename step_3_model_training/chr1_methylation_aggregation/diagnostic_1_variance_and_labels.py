#!/usr/bin/env python3
"""
DIAGNOSTIC 1: Label Consistency & Within-Class Variance Analysis
=================================================================

This script checks:
1. Whether CSV tissue_index matches HDF5 tissue_labels for each file
2. Whether broad tissue categories have coherent methylation patterns
   or if within-class variance is too high for learning

Run: python diagnostic_1_labels_and_variance.py
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
HDF5_PATH = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset_chr1.h5"
TRAIN_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/train_files.csv"
VAL_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/val_files.csv"
TEST_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/test_files.csv"

# How many regions to sample for variance analysis (for speed)
N_REGIONS_SAMPLE = 500

print("=" * 80)
print("DIAGNOSTIC 1: LABEL CONSISTENCY & WITHIN-CLASS VARIANCE")
print("=" * 80)

# ============================================================================
# PART 1: LABEL CONSISTENCY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: LABEL CONSISTENCY CHECK")
print("=" * 80)
print("\nChecking if CSV tissue_index matches HDF5 tissue_labels...")

# Load all CSVs
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"\nLoaded {len(all_df)} files from CSVs")
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Load HDF5 metadata
with h5py.File(HDF5_PATH, 'r') as f:
    hdf5_filenames = [fn.decode() for fn in f['metadata/filenames'][:]]
    hdf5_tissue_labels = f['tissue_labels'][:]
    hdf5_tissue_names = [tn.decode() for tn in f['metadata/tissue_names'][:]]

filename_to_hdf5_idx = {fn: idx for idx, fn in enumerate(hdf5_filenames)}

print(f"Loaded {len(hdf5_filenames)} files from HDF5")
print(f"HDF5 tissue_labels range: [{hdf5_tissue_labels.min()}, {hdf5_tissue_labels.max()}]")
print(f"Unique HDF5 tissue labels: {len(np.unique(hdf5_tissue_labels))}")

# Check consistency
mismatches = []
matches = 0

for idx, row in all_df.iterrows():
    filename = row['filename']
    csv_tissue_index = row['tissue_index']
    csv_tissue_name = row['tissue']
    
    if filename not in filename_to_hdf5_idx:
        mismatches.append({
            'filename': filename,
            'issue': 'FILE_NOT_IN_HDF5',
            'csv_index': csv_tissue_index,
            'hdf5_index': None
        })
        continue
    
    hdf5_idx = filename_to_hdf5_idx[filename]
    hdf5_label = hdf5_tissue_labels[hdf5_idx]
    hdf5_name = hdf5_tissue_names[hdf5_idx]
    
    if csv_tissue_index != hdf5_label:
        mismatches.append({
            'filename': filename,
            'issue': 'LABEL_MISMATCH',
            'csv_index': csv_tissue_index,
            'csv_tissue': csv_tissue_name,
            'hdf5_index': hdf5_label,
            'hdf5_tissue': hdf5_name
        })
    else:
        matches += 1

print(f"\n" + "-" * 80)
print("LABEL CONSISTENCY RESULTS")
print("-" * 80)

if len(mismatches) == 0:
    print(f"\n✓ PERFECT MATCH: All {matches} files have consistent labels")
    print("  CSV tissue_index == HDF5 tissue_labels for all files")
else:
    print(f"\n✗ MISMATCH DETECTED!")
    print(f"  Matching: {matches}")
    print(f"  Mismatched: {len(mismatches)}")
    
    print(f"\nFirst 20 mismatches:")
    for m in mismatches[:20]:
        if m['issue'] == 'FILE_NOT_IN_HDF5':
            print(f"  {m['filename']}: NOT FOUND IN HDF5")
        else:
            print(f"  {m['filename']}:")
            print(f"    CSV:  tissue_index={m['csv_index']} ({m['csv_tissue']})")
            print(f"    HDF5: tissue_label={m['hdf5_index']} ({m['hdf5_tissue']})")

# Show mapping between broad and fine-grained
print(f"\n" + "-" * 80)
print("TISSUE MAPPING ANALYSIS")
print("-" * 80)

# Group by broad tissue category
broad_to_fine = defaultdict(set)
broad_to_hdf5_labels = defaultdict(set)

for idx, row in all_df.iterrows():
    broad = row['tissue']  # e.g., "Blood"
    fine = row['tissue_type']  # e.g., "Blood-Granulocytes_2"
    csv_index = row['tissue_index']
    
    # Skip if fine is NaN or not a string
    if pd.isna(fine) or not isinstance(fine, str):
        fine_base = str(broad)  # Fallback to broad category
    else:
        # Remove trailing number from tissue_type
        fine_base = '_'.join(fine.rsplit('_', 1)[:-1]) if fine[-1].isdigit() else fine
    
    broad_to_fine[broad].add(fine_base)
    
    filename = row['filename']
    if filename in filename_to_hdf5_idx:
        hdf5_idx = filename_to_hdf5_idx[filename]
        hdf5_label = hdf5_tissue_labels[hdf5_idx]
        broad_to_hdf5_labels[broad].add(hdf5_label)

print(f"\nBroad tissue categories and their fine-grained subtypes:")
print(f"{'Broad Category':<25} {'HDF5 Labels':<15} {'Fine-grained Subtypes'}")
print("-" * 80)

for broad in sorted(broad_to_fine.keys()):
    fine_types = sorted(broad_to_fine[broad])
    hdf5_labels = sorted(broad_to_hdf5_labels[broad])
    
    hdf5_str = ','.join(map(str, hdf5_labels))
    fine_str = ', '.join(fine_types[:3])
    if len(fine_types) > 3:
        fine_str += f" (+{len(fine_types)-3} more)"
    
    print(f"{broad:<25} {hdf5_str:<15} {fine_str}")

print(f"\nTotal broad categories: {len(broad_to_fine)}")
print(f"Total fine-grained types: {sum(len(v) for v in broad_to_fine.values())}")

# ============================================================================
# PART 2: WITHIN-CLASS VARIANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: WITHIN-CLASS VARIANCE ANALYSIS")
print("=" * 80)
print("\nAnalyzing methylation pattern coherence within broad tissue categories...")
print(f"(Sampling {N_REGIONS_SAMPLE} regions for speed)")

with h5py.File(HDF5_PATH, 'r') as f:
    n_files = f['methylation'].shape[0]
    n_regions = f['methylation'].shape[1]
    seq_len = f['methylation'].shape[2]
    
    print(f"\nHDF5 shape: ({n_files} files, {n_regions} regions, {seq_len} bp)")
    
    # Sample random regions
    np.random.seed(42)
    region_indices = np.random.choice(n_regions, min(N_REGIONS_SAMPLE, n_regions), replace=False)
    region_indices = np.sort(region_indices)
    
    # For each file, compute mean methylation across sampled regions
    # methylation: 0=unmeth, 1=meth, 2=missing
    
    print(f"\nComputing per-file methylation profiles...")
    
    file_profiles = []  # Mean methylation rate per file
    file_labels = hdf5_tissue_labels
    
    for file_idx in range(n_files):
        # Load methylation for sampled regions
        meth_data = f['methylation'][file_idx, region_indices, :]
        
        # Compute methylation rate (ignoring missing=2)
        valid_mask = meth_data < 2
        if valid_mask.sum() > 0:
            meth_rate = meth_data[valid_mask].mean()  # 0 or 1, so mean = methylation rate
        else:
            meth_rate = np.nan
        
        file_profiles.append(meth_rate)
        
        if (file_idx + 1) % 100 == 0:
            print(f"  Processed {file_idx + 1}/{n_files} files...")
    
    file_profiles = np.array(file_profiles)

print(f"\nMethylation profiles computed for {n_files} files")

# Analyze variance within each broad tissue category
print(f"\n" + "-" * 80)
print("WITHIN-CLASS VS BETWEEN-CLASS VARIANCE")
print("-" * 80)

# Group files by broad tissue (using HDF5 labels)
tissue_to_files = defaultdict(list)
for file_idx, label in enumerate(file_labels):
    tissue_to_files[label].append(file_idx)

# Compute within-class and between-class statistics
within_class_vars = []
class_means = []

print(f"\n{'Tissue':<10} {'N Files':<10} {'Mean Meth':<12} {'Std Dev':<12} {'CV%':<10}")
print("-" * 60)

for tissue_id in sorted(tissue_to_files.keys()):
    file_indices = tissue_to_files[tissue_id]
    profiles = file_profiles[file_indices]
    profiles = profiles[~np.isnan(profiles)]  # Remove NaN
    
    if len(profiles) < 2:
        continue
    
    mean_val = np.mean(profiles)
    std_val = np.std(profiles)
    cv = (std_val / mean_val * 100) if mean_val > 0 else 0
    
    class_means.append(mean_val)
    within_class_vars.append(std_val ** 2)
    
    print(f"{tissue_id:<10} {len(profiles):<10} {mean_val:<12.4f} {std_val:<12.4f} {cv:<10.1f}")

# Overall statistics
mean_within_var = np.mean(within_class_vars)
between_var = np.var(class_means)

print(f"\n" + "-" * 80)
print("VARIANCE SUMMARY")
print("-" * 80)
print(f"\nMean within-class variance: {mean_within_var:.6f}")
print(f"Between-class variance:     {between_var:.6f}")
print(f"Ratio (between/within):     {between_var/mean_within_var:.4f}")

if between_var / mean_within_var < 1.0:
    print(f"\n⚠️  WARNING: Between-class variance is LOWER than within-class variance!")
    print(f"   This means tissues overlap more than they separate.")
    print(f"   The model cannot distinguish tissues based on methylation patterns.")
    print(f"\n   LIKELY CAUSE: Grouping diverse subtypes into broad categories")
    print(f"   SOLUTION: Train on fine-grained tissue types instead")
elif between_var / mean_within_var < 2.0:
    print(f"\n⚠️  WARNING: Between/within ratio is low (<2.0)")
    print(f"   Classes are only weakly separable.")
    print(f"   Consider using fine-grained tissue types for better separation.")
else:
    print(f"\n✓ Between/within ratio looks reasonable (>{between_var/mean_within_var:.1f})")

# ============================================================================
# PART 3: REGION-LEVEL METHYLATION PATTERN ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: REGION-LEVEL METHYLATION PATTERN ANALYSIS")
print("=" * 80)
print("\nChecking if methylation patterns vary meaningfully across tissues...")

with h5py.File(HDF5_PATH, 'r') as f:
    # Pick 10 random regions and see if different tissues have different patterns
    test_regions = np.random.choice(n_regions, 10, replace=False)
    
    print(f"\nSampling 10 random regions to check tissue-specific patterns...")
    
    for region_idx in test_regions[:5]:  # Show first 5
        print(f"\n  Region {region_idx}:")
        
        # Get methylation for this region across all files
        region_meth = f['methylation'][:, region_idx, :]
        
        # Compute mean methylation per tissue
        tissue_meth = {}
        for tissue_id in sorted(tissue_to_files.keys()):
            file_indices = tissue_to_files[tissue_id]
            meth_data = region_meth[file_indices, :]
            
            valid_mask = meth_data < 2
            if valid_mask.sum() > 0:
                meth_rate = meth_data[valid_mask].mean()
            else:
                meth_rate = np.nan
            
            tissue_meth[tissue_id] = meth_rate
        
        # Show range across tissues
        valid_rates = [v for v in tissue_meth.values() if not np.isnan(v)]
        if valid_rates:
            min_rate = min(valid_rates)
            max_rate = max(valid_rates)
            range_rate = max_rate - min_rate
            
            print(f"    Methylation range across tissues: {min_rate:.3f} - {max_rate:.3f} (range: {range_rate:.3f})")
            
            if range_rate < 0.1:
                print(f"    ⚠️  Low variation - this region looks similar across all tissues")
            else:
                print(f"    ✓ Good variation - this region differs between tissues")

# ============================================================================
# FINAL DIAGNOSIS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL DIAGNOSIS")
print("=" * 80)

issues_found = []

if len(mismatches) > 0:
    issues_found.append("LABEL_MISMATCH: CSV and HDF5 labels don't match")

if between_var / mean_within_var < 1.0:
    issues_found.append("HIGH_WITHIN_VARIANCE: Broad categories have too much internal diversity")

if len(issues_found) == 0:
    print("\n✓ No major issues found in labels or variance")
    print("  The problem may lie elsewhere (learning rate, model architecture, etc.)")
else:
    print(f"\n✗ ISSUES FOUND ({len(issues_found)}):")
    for issue in issues_found:
        print(f"  - {issue}")
    
    print("\nRECOMMENDED ACTIONS:")
    if "LABEL_MISMATCH" in str(issues_found):
        print("  1. Fix the label mapping between CSV and HDF5")
        print("     Either regenerate HDF5 with correct labels, or update CSVs")
    
    if "HIGH_WITHIN_VARIANCE" in str(issues_found):
        print("  2. Train on fine-grained tissue types instead of broad categories")
        print("     This will give the model cleaner, more separable targets")
        print("     You can group predictions post-hoc if needed")

print("\n" + "=" * 80)
print("DIAGNOSTIC 1 COMPLETE")
print("=" * 80)
