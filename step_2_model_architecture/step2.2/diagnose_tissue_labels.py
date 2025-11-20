#!/usr/bin/env python3
"""
Diagnostic script to check tissue labels in HDF5 file
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Paths
HDF5_FILE = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5")
TRAIN_CSV = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/data_splits/train_files.csv")
VAL_CSV = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/data_splits/val_files.csv")
TEST_CSV = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/data_splits/test_files.csv")

def main():
    print("="*80)
    print("HDF5 Tissue Label Diagnostic")
    print("="*80)
    
    # Open HDF5 file
    with h5py.File(HDF5_FILE, 'r') as f:
        print("\n1. HDF5 File Contents:")
        print(f"   - Total files: {f.attrs['n_files']}")
        print(f"   - n_tissues attribute: {f.attrs['n_tissues']}")
        
        tissue_labels = f['tissue_labels'][:]
        filenames = [fn.decode() for fn in f['metadata/filenames'][:]]
        
        print(f"\n2. Tissue Label Statistics:")
        print(f"   - Shape: {tissue_labels.shape}")
        print(f"   - Min: {tissue_labels.min()}")
        print(f"   - Max: {tissue_labels.max()}")
        print(f"   - Unique values: {len(np.unique(tissue_labels))}")
        print(f"   - Unique tissue indices: {sorted(np.unique(tissue_labels))}")
        
        # Count distribution
        tissue_counts = Counter(tissue_labels)
        print(f"\n3. Tissue Distribution (all files):")
        for tissue_idx in sorted(tissue_counts.keys())[:20]:  # First 20
            count = tissue_counts[tissue_idx]
            print(f"   Tissue {tissue_idx}: {count} files")
        if len(tissue_counts) > 20:
            print(f"   ... and {len(tissue_counts) - 20} more tissues")
    
    # Load split CSVs
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Map filenames to indices
    filename_to_idx = {fn: idx for idx, fn in enumerate(filenames)}
    
    # Get indices for each split
    train_indices = np.sort([filename_to_idx[fn] for fn in train_df['filename']])
    val_indices = np.sort([filename_to_idx[fn] for fn in val_df['filename']])
    test_indices = np.sort([filename_to_idx[fn] for fn in test_df['filename']])
    
    print(f"\n4. Split Indices:")
    print(f"   Train: {len(train_indices)} files")
    print(f"     First 10 indices: {train_indices[:10]}")
    print(f"     Last 10 indices: {train_indices[-10:]}")
    print(f"   Val: {len(val_indices)} files")
    print(f"     First 10 indices: {val_indices[:10]}")
    print(f"     Last 10 indices: {val_indices[-10:]}")
    print(f"   Test: {len(test_indices)} files")
    print(f"     First 10 indices: {test_indices[:10]}")
    print(f"     Last 10 indices: {test_indices[-10:]}")
    
    # Check tissue labels for each split
    with h5py.File(HDF5_FILE, 'r') as f:
        train_tissues = f['tissue_labels'][train_indices]
        val_tissues = f['tissue_labels'][val_indices]
        test_tissues = f['tissue_labels'][test_indices]
        
        print(f"\n5. Tissue Labels by Split:")
        print(f"\n   TRAIN SET:")
        print(f"     Unique tissues: {len(np.unique(train_tissues))}")
        print(f"     Tissue range: [{train_tissues.min()}, {train_tissues.max()}]")
        train_counts = Counter(train_tissues)
        print(f"     Distribution (top 10):")
        for tissue_idx, count in train_counts.most_common(10):
            print(f"       Tissue {tissue_idx}: {count} files")
        
        print(f"\n   VALIDATION SET:")
        print(f"     Unique tissues: {len(np.unique(val_tissues))}")
        print(f"     Tissue range: [{val_tissues.min()}, {val_tissues.max()}]")
        val_counts = Counter(val_tissues)
        print(f"     Distribution (all):")
        for tissue_idx, count in sorted(val_counts.items()):
            print(f"       Tissue {tissue_idx}: {count} files")
        
        print(f"\n   TEST SET:")
        print(f"     Unique tissues: {len(np.unique(test_tissues))}")
        print(f"     Tissue range: [{test_tissues.min()}, {test_tissues.max()}]")
        test_counts = Counter(test_tissues)
        print(f"     Distribution (top 10):")
        for tissue_idx, count in test_counts.most_common(10):
            print(f"       Tissue {tissue_idx}: {count} files")
    
    # Check for the specific issue: why validation shows only tissue 0
    print(f"\n6. Investigating Validation Issue:")
    print(f"   First 20 validation file indices: {val_indices[:20]}")
    with h5py.File(HDF5_FILE, 'r') as f:
        for i, idx in enumerate(val_indices[:20]):
            tissue = f['tissue_labels'][idx]
            filename = f['metadata/filenames'][idx].decode()
            print(f"     File {i}: index={idx}, tissue={tissue}, filename={filename[:50]}")
    
    print("\n" + "="*80)
    print("Diagnostic Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
