"""
Detailed inspection of HDF5 metadata to understand structure
"""

import h5py
import numpy as np

hdf5_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset_chr1.h5'

print("="*80)
print("DETAILED HDF5 INSPECTION")
print("="*80)

with h5py.File(hdf5_path, 'r') as f:
    print("\n1. TOP-LEVEL STRUCTURE")
    print("-" * 80)
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Dataset):
            print(f"   {key}:")
            print(f"      Type: Dataset")
            print(f"      Shape: {item.shape}")
            print(f"      Dtype: {item.dtype}")
            if item.size < 1000:  # Small enough to show
                print(f"      Data: {item[:]}")
        elif isinstance(item, h5py.Group):
            print(f"   {key}:")
            print(f"      Type: Group")
            print(f"      Keys: {list(item.keys())}")
    
    print("\n2. METADATA DETAILED INSPECTION")
    print("-" * 80)
    if 'metadata' in f.keys():
        metadata = f['metadata']
        
        if isinstance(metadata, h5py.Dataset):
            print(f"   Metadata is a Dataset")
            print(f"   Shape: {metadata.shape}")
            print(f"   Dtype: {metadata.dtype}")
            print(f"   Size: {metadata.size}")
            
            # Try to read first few entries
            if metadata.size > 0:
                print(f"\n   First 3 entries:")
                for i in range(min(3, metadata.shape[0])):
                    entry = metadata[i]
                    print(f"      [{i}]: {entry[:200] if hasattr(entry, '__len__') else entry}...")
                    
                    # If it's bytes/string, try to decode
                    if isinstance(entry, (bytes, np.bytes_)):
                        try:
                            decoded = entry.decode('utf-8')
                            print(f"         Decoded: {decoded[:200]}...")
                        except:
                            pass
        
        elif isinstance(metadata, h5py.Group):
            print(f"   Metadata is a Group with {len(metadata.keys())} keys")
            
            for key in metadata.keys():
                item = metadata[key]
                print(f"\n   metadata['{key}']:")
                
                if isinstance(item, h5py.Dataset):
                    print(f"      Type: Dataset")
                    print(f"      Shape: {item.shape}")
                    print(f"      Dtype: {item.dtype}")
                    
                    # Show first few entries if reasonable size
                    if item.shape[0] <= 10:
                        print(f"      Data: {item[:]}")
                    elif item.shape[0] <= 51089:  # Region-level data
                        print(f"      First 5: {item[:5]}")
                        if item.dtype.kind in ['S', 'U', 'O']:  # String-like
                            # Check for chr1
                            sample = [str(item[i]) for i in range(min(10, item.shape[0]))]
                            print(f"      Sample: {sample}")
                            
                            # Count chr1
                            try:
                                chr1_count = sum(1 for i in range(item.shape[0]) 
                                                if 'chr1' in str(item[i]).lower())
                                print(f"      Chr1 count: {chr1_count} / {item.shape[0]}")
                            except:
                                pass
                
                elif isinstance(item, h5py.Group):
                    print(f"      Type: Group")
                    print(f"      Subkeys: {list(item.keys())}")
    
    print("\n3. CHECKING FOR CHROMOSOME INFORMATION")
    print("-" * 80)
    
    # Check common places where chromosome info might be
    possible_locations = [
        'metadata/regions',
        'metadata/chrom',
        'metadata/chromosome', 
        'metadata/region_ids',
        'metadata/coordinates',
        'chrom',
        'chromosome',
        'regions',
        'region_ids'
    ]
    
    for location in possible_locations:
        try:
            parts = location.split('/')
            item = f
            for part in parts:
                item = item[part]
            
            print(f"\n   ✓ FOUND: {location}")
            print(f"      Shape: {item.shape}")
            print(f"      Dtype: {item.dtype}")
            print(f"      First 5:")
            for i in range(min(5, item.shape[0])):
                print(f"         [{i}]: {item[i]}")
            
            # If it looks like region IDs, count chr1
            if item.dtype.kind in ['S', 'U', 'O']:
                chr1_count = 0
                for i in range(min(1000, item.shape[0])):
                    if 'chr1' in str(item[i]).lower():
                        chr1_count += 1
                        if chr1_count <= 3:  # Show first 3 chr1 entries
                            print(f"         Example chr1: {item[i]}")
                
                print(f"      Chr1 count (first 1000): {chr1_count}")
                
        except KeyError:
            continue
    
    print("\n4. SAMPLE DATA")
    print("-" * 80)
    print(f"   File 0, Region 0, Position 0:")
    print(f"      DNA token: {f['dna_tokens'][0, 0, 0]}")
    print(f"      Methylation: {f['methylation'][0, 0, 0]}")
    print(f"      N reads: {f['n_reads'][0, 0]}")
    print(f"      Tissue label (file): {f['tissue_labels'][0]}")

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Look for chromosome information in the output above")
print("2. If found, note the exact path (e.g., 'metadata/regions')")
print("3. Use that to extract chr1 regions")
print("="*80)
