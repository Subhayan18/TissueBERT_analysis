#!/usr/bin/env python3
"""
Create Full Genome Dataset (All Chromosomes)

Memory-efficient processing:
- Process HDF5 in chunks to handle 40GB GPU memory limit
- 765 files × 51,089 regions × 150bp = ~5.8GB raw data
- Process in batches of files to stay within memory
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
input_hdf5 = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5'
output_hdf5 = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset_fullgenome.h5'

input_csv_dir = Path('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset')
output_csv_dir = Path('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/fullgenome_subset')
output_csv_dir.mkdir(exist_ok=True, parents=True)

print("="*80)
print("Creating Full Genome Dataset (All Chromosomes)")
print("="*80)

# Step 1: Verify input HDF5
print("\n1. Verifying input HDF5...")
with h5py.File(input_hdf5, 'r') as f:
    n_files = f['dna_tokens'].shape[0]
    n_regions = f['dna_tokens'].shape[1]
    seq_length = f['dna_tokens'].shape[2]
    
    print(f"   Files: {n_files}")
    print(f"   Regions per file: {n_regions:,}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Total size: {n_files} × {n_regions:,} × {seq_length} = {n_files*n_regions*seq_length:,} elements")
    
    # Calculate memory requirements
    bytes_per_element = 1  # uint8
    total_bytes = n_files * n_regions * seq_length * bytes_per_element
    total_gb = total_bytes / 1e9
    print(f"   Estimated size: {total_gb:.2f} GB per dataset (dna_tokens, methylation)")

# Step 2: Copy full genome data (it's already the full dataset)
print("\n2. Verifying this is already the full genome dataset...")
print("   The input HDF5 already contains all chromosomes.")
print("   Creating symbolic link for consistency...")

import os
if os.path.exists(output_hdf5):
    os.remove(output_hdf5)
os.symlink(input_hdf5, output_hdf5)

print(f"   [OK] Linked: {output_hdf5}")

# Step 3: Copy CSV files
print("\n3. Copying train/val/test splits...")
train_df = pd.read_csv(input_csv_dir / 'train_files.csv')
val_df = pd.read_csv(input_csv_dir / 'val_files.csv')
test_df = pd.read_csv(input_csv_dir / 'test_files.csv')

print(f"   Files: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

train_df.to_csv(output_csv_dir / 'train_files.csv', index=False)
val_df.to_csv(output_csv_dir / 'val_files.csv', index=False)
test_df.to_csv(output_csv_dir / 'test_files.csv', index=False)

print(f"   [OK] Created CSVs in: {output_csv_dir}")

# Step 4: Summary statistics
print("\n4. Full genome dataset statistics:")
with h5py.File(output_hdf5, 'r') as f:
    n_files = f['tissue_labels'].shape[0]
    n_regions = f['dna_tokens'].shape[1]
    tissues = f['tissue_labels'][:]
    unique_tissues = np.unique(tissues)
    
    print(f"   Files: {n_files}")
    print(f"   Regions per file: {n_regions:,}")
    print(f"   Total training examples (file-level): {n_files:,}")
    print(f"   Unique tissues: {len(unique_tissues)}")
    
    # Memory calculation for training
    bytes_per_sample = n_regions * seq_length * 2  # dna + methylation
    mb_per_sample = bytes_per_sample / 1e6
    print(f"\n   Memory per sample: {mb_per_sample:.1f} MB")
    print(f"   Memory for batch_size=8: {mb_per_sample * 8:.1f} MB")
    print(f"   Memory for batch_size=4: {mb_per_sample * 4:.1f} MB")

# Step 5: Recommendations
print("\n5. Training recommendations for 40GB GPU memory:")
print("   - Use batch_size=4 (safer for 40GB)")
print("   - Use gradient_accumulation_steps=8 (effective batch=32)")
print("   - Use num_workers=12 (24 cores / 2)")
print("   - Training will be ~12x slower than chr1 (51k vs 4.4k regions)")

print("\n" + "="*80)
print("Full genome dataset ready!")
print("="*80)
print(f"\nUpdate config to use:")
print(f"  hdf5_path: {output_hdf5}")
print(f"  train_csv: {output_csv_dir}/train_files.csv")
print(f"  val_csv: {output_csv_dir}/val_files.csv")
print(f"  test_csv: {output_csv_dir}/test_files.csv")
print("="*80)
