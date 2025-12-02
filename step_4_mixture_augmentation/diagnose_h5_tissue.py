#!/usr/bin/env python3
"""
Diagnostic script to check H5 file structure and extract tissue names
"""

import h5py
import numpy as np

h5_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/phase2_test_mixtures.h5'

print("="*60)
print("DIAGNOSTIC: H5 File Structure")
print("="*60)

with h5py.File(h5_path, 'r') as f:
    print("\nTop-level keys:")
    for key in f.keys():
        print(f"  - {key}: {f[key].shape if hasattr(f[key], 'shape') else 'group'}")
    
    print("\n" + "="*60)
    print("MIXTURE_INFO:")
    print("="*60)
    
    if 'mixture_info' in f:
        mixture_info = f['mixture_info']
        print(f"\nType: {type(mixture_info)}")
        print(f"Shape: {mixture_info.shape}")
        print(f"Dtype: {mixture_info.dtype}")
        
        # It's a structured array/dataset
        print("\nField names (columns):")
        if mixture_info.dtype.names:
            for field_name in mixture_info.dtype.names:
                print(f"  - {field_name}")
        
        # Load first few rows
        print("\nFirst 3 rows:")
        data = mixture_info[:3]
        for i, row in enumerate(data):
            print(f"\n  Row {i}:")
            if mixture_info.dtype.names:
                for field_name in mixture_info.dtype.names:
                    value = row[field_name]
                    print(f"    {field_name}: {value}")
        
        # Check for tissue_names field
        if mixture_info.dtype.names and 'tissue_names' in mixture_info.dtype.names:
            print("\n" + "="*60)
            print("EXTRACTING TISSUE NAMES FROM FIELD:")
            print("="*60)
            
            # Get unique tissue names
            all_tissue_names = mixture_info['tissue_names'][:]
            print(f"\nTotal entries: {len(all_tissue_names)}")
            print(f"First 5 entries: {all_tissue_names[:5]}")
            
            # Try to extract unique tissue list
            if len(all_tissue_names) > 0:
                first_entry = all_tissue_names[0]
                print(f"\nFirst entry type: {type(first_entry)}")
                print(f"First entry value: {repr(first_entry)}")
                
                # Try decoding
                if isinstance(first_entry, bytes):
                    decoded = first_entry.decode('utf-8')
                    print(f"Decoded: {decoded}")
                elif isinstance(first_entry, np.ndarray):
                    print(f"Array shape: {first_entry.shape}")
                    print(f"Array dtype: {first_entry.dtype}")
                    if first_entry.dtype.kind == 'S' or first_entry.dtype.kind == 'U':
                        decoded = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in first_entry]
                        print(f"Decoded array: {decoded}")
    
    print("\n" + "="*60)
    print("CHECKING METADATA CSV FOR TISSUE NAMES:")
    print("="*60)
    
    import pandas as pd
    metadata_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/combined_metadata.csv'
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"\nMetadata shape: {metadata.shape}")
        print(f"Columns: {metadata.columns.tolist()}")
        
        if 'tissue_full' in metadata.columns:
            unique_tissues = metadata['tissue_full'].unique()
            print(f"\nUnique tissues from metadata ({len(unique_tissues)}):")
            for i, tissue in enumerate(sorted(unique_tissues)):
                print(f"  {i}: {tissue}")
    except Exception as e:
        print(f"\nError loading metadata: {e}")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
