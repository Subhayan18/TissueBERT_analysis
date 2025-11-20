#!/usr/bin/env python3
"""
Quick check: What HDF5 file is the dataloader using?
"""

import h5py
from pathlib import Path

HDF5_PATH = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5")

print("="*80)
print("HDF5 File Check")
print("="*80)

if not HDF5_PATH.exists():
    print(f"\n❌ HDF5 file not found: {HDF5_PATH}")
    print("\nYou need to run the conversion script first:")
    print("  python convert_to_hdf5_v2.py")
    exit(1)

print(f"\n📂 Checking: {HDF5_PATH}")
print(f"   File size: {HDF5_PATH.stat().st_size / 1e9:.2f} GB")
print(f"   Modified: {Path(HDF5_PATH).stat().st_mtime}")

with h5py.File(HDF5_PATH, 'r') as f:
    print(f"\n📊 HDF5 Contents:")
    print(f"   n_tissues attribute: {f.attrs.get('n_tissues', 'NOT SET')}")
    
    tissue_labels = f['tissue_labels'][:]
    print(f"\n🏷️  Tissue Labels:")
    print(f"   Shape: {tissue_labels.shape}")
    print(f"   Min: {tissue_labels.min()}")
    print(f"   Max: {tissue_labels.max()}")
    print(f"   Unique: {len(set(tissue_labels))}")
    print(f"   Range: [{tissue_labels.min()}, {tissue_labels.max()}]")
    
    if 'tissue_source' in f.attrs:
        print(f"\n   Source: {f.attrs['tissue_source']}")
    
    print(f"\n✅ Verdict:")
    if tissue_labels.max() == 21:
        print(f"   ✅ CORRECT: Using 22-class system (0-21)")
        print(f"   This is the NEW, fixed HDF5 file")
    elif tissue_labels.max() == 118:
        print(f"   ❌ WRONG: Using 119-class system (0-118)")
        print(f"   This is the OLD HDF5 file - you need to regenerate it!")
        print(f"\n   Run: python convert_to_hdf5_v2.py")
    else:
        print(f"   ⚠️  UNEXPECTED: Max tissue label is {tissue_labels.max()}")

print("\n" + "="*80)
