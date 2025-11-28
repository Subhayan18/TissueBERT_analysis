"""
Extract chr1-only data from HDF5 using region_ids from original NPZ files

FINAL CORRECT VERSION:
- Load region_ids from any original NPZ file (all files have same 51,089 regions)
- Find chr1 indices
- Subset HDF5 to only those regions
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

# Paths
npz_dir = Path('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/all_data')
input_hdf5 = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5'
output_hdf5 = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset_chr1.h5'

input_csv_dir = Path('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset')
output_csv_dir = Path('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset')
output_csv_dir.mkdir(exist_ok=True)

print("="*80)
print("Creating chr1-only subset for debugging")
print("="*80)

# Step 1: Load region_ids from any NPZ file
print("\n1. Loading region_ids from original NPZ files...")

# Find any NPZ file
npz_files = list(npz_dir.glob('*.npz'))
if len(npz_files) == 0:
    print(f"   ERROR: No NPZ files found in {npz_dir}")
    print(f"   Please provide the correct path to NPZ files")
    exit(1)

# Load first file to get region_ids
first_npz = npz_files[0]
print(f"   Loading: {first_npz.name}")

with np.load(first_npz) as data:
    if 'region_ids' not in data.files:
        print(f"   ERROR: 'region_ids' not found in NPZ file!")
        print(f"   Available keys: {data.files}")
        exit(1)
    
    region_ids = data['region_ids']
    print(f"   Found {len(region_ids):,} region_ids")
    print(f"   First 5: {region_ids[:5]}")

# Step 2: Find chr1 regions
print("\n2. Identifying chr1 regions...")

chr1_mask = np.array([str(rid).startswith('chr1_') for rid in region_ids])
chr1_indices = np.where(chr1_mask)[0]

print(f"   ✓ Found {len(chr1_indices):,} regions on chr1")
print(f"   That's {len(chr1_indices)/len(region_ids)*100:.1f}% of all regions")

# Show some examples
chr1_regions = region_ids[chr1_indices]
print(f"   First 5 chr1 regions: {chr1_regions[:5]}")
print(f"   Last 5 chr1 regions: {chr1_regions[-5:]}")

# Step 3: Extract chr1 data from HDF5
print("\n3. Extracting chr1 data to new HDF5...")
with h5py.File(input_hdf5, 'r') as f_in:
    with h5py.File(output_hdf5, 'w') as f_out:
        
        print(f"   Extracting dna_tokens...")
        # Shape: (765 files, 51089 regions, 150 bp) → (765 files, N_chr1 regions, 150 bp)
        dna_chr1 = f_in['dna_tokens'][:, chr1_indices, :]
        f_out.create_dataset('dna_tokens', data=dna_chr1, compression='gzip')
        print(f"      New shape: {dna_chr1.shape}")
        
        print(f"   Extracting methylation...")
        meth_chr1 = f_in['methylation'][:, chr1_indices, :]
        f_out.create_dataset('methylation', data=meth_chr1, compression='gzip')
        print(f"      New shape: {meth_chr1.shape}")
        
        print(f"   Extracting n_reads...")
        reads_chr1 = f_in['n_reads'][:, chr1_indices]
        f_out.create_dataset('n_reads', data=reads_chr1, compression='gzip')
        print(f"      New shape: {reads_chr1.shape}")
        
        print(f"   Copying tissue_labels...")
        f_out.create_dataset('tissue_labels', data=f_in['tissue_labels'][:])
        
        # Copy file-level metadata
        if 'metadata' in f_in.keys():
            print(f"   Copying metadata...")
            meta_out = f_out.create_group('metadata')
            for key in f_in['metadata'].keys():
                meta_out.create_dataset(key, data=f_in['metadata'][key][:])
        
        # CRITICAL: Copy ALL attributes from original HDF5
        print(f"   Copying HDF5 attributes...")
        for key in f_in.attrs.keys():
            if key == 'n_regions_per_file':
                # Adjust for chr1
                f_out.attrs[key] = dna_chr1.shape[1]
                print(f"      {key}: {dna_chr1.shape[1]} (adjusted for chr1)")
            else:
                # Copy as-is
                f_out.attrs[key] = f_in.attrs[key]
                print(f"      {key}: {f_in.attrs[key]} (copied)")
        
        # Add chr1 marker
        f_out.attrs['chromosome'] = 'chr1'
        print(f"      chromosome: chr1 (added)")

print(f"\n4. Created chr1 HDF5: {output_hdf5}")
file_size = Path(output_hdf5).stat().st_size / 1e6
print(f"   Size: {file_size:.1f} MB")
print(f"   Shape: ({dna_chr1.shape[0]} files, {dna_chr1.shape[1]} regions, {dna_chr1.shape[2]} bp)")

# Step 4: Create corresponding CSV files
print("\n5. Creating chr1 train/val/test splits...")
train_df = pd.read_csv(input_csv_dir / 'train_files.csv')
val_df = pd.read_csv(input_csv_dir / 'val_files.csv')
test_df = pd.read_csv(input_csv_dir / 'test_files.csv')

print(f"   Files: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

# CSVs stay the same - they reference the same 765 files
train_df.to_csv(output_csv_dir / 'train_files.csv', index=False)
val_df.to_csv(output_csv_dir / 'val_files.csv', index=False)
test_df.to_csv(output_csv_dir / 'test_files.csv', index=False)

print(f"   Created chr1 CSVs in: {output_csv_dir}")

# Step 5: Summary statistics
print("\n6. Chr1 subset statistics:")
with h5py.File(output_hdf5, 'r') as f:
    n_files = f['tissue_labels'].shape[0]
    n_regions_per_file = f['dna_tokens'].shape[1]
    tissues = f['tissue_labels'][:]
    unique_tissues = np.unique(tissues)
    
    print(f"   Files: {n_files}")
    print(f"   Chr1 regions per file: {n_regions_per_file:,}")
    print(f"   Total training examples: {n_files * n_regions_per_file:,}")
    print(f"   Unique tissues: {len(unique_tissues)}")
    print(f"\n   Files per tissue:")
    
    for tissue_id in sorted(unique_tissues):
        count = np.sum(tissues == tissue_id)
        pct = count / n_files * 100
        print(f"      Tissue {tissue_id:2d}: {count:3d} files ({pct:4.1f}%)")

# Calculate expected training time
batches_per_epoch = (n_files * n_regions_per_file) // 128  # batch_size=128
steps_per_epoch_limited = min(1000, batches_per_epoch)  # limited to 1000 in config
minutes_per_epoch = steps_per_epoch_limited / 200  # ~200 steps/min estimated

print(f"\n7. Expected training performance:")
print(f"   Theoretical batches per epoch: {batches_per_epoch:,}")
print(f"   Limited to (config): 1,000 steps")
print(f"   Estimated time per epoch: ~{minutes_per_epoch:.1f} minutes")
print(f"   10 epochs: ~{minutes_per_epoch * 10:.0f} minutes")

print("\n" + "="*80)
print("Chr1 subset created successfully!")
print("="*80)
print(f"\nTo use chr1 subset:")
print(f"1. Update config hdf5_path to:")
print(f"   {output_hdf5}")
print(f"2. Update CSV paths to:")
print(f"   {output_csv_dir}/train_files.csv")
print(f"   {output_csv_dir}/val_files.csv")
print(f"   {output_csv_dir}/test_files.csv")
print(f"3. Launch training with config_chr1_debug.yaml")
print(f"4. Expected: ~{minutes_per_epoch * 10:.0f} minutes for 10 epochs")
print("="*80)
