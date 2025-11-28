import h5py
import numpy as np

# This will be run on your HPC to inspect the HDF5 file
hdf5_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5'

print("Opening HDF5 file...")
with h5py.File(hdf5_path, 'r') as f:
    print(f"\nTop-level keys: {list(f.keys())}")
    
    # Check structure
    print(f"\nDataset shapes:")
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            print(f"  {key}: {f[key].shape}, dtype: {f[key].dtype}")
    
    # Sample some data to see chromosome info
    if 'chrom' in f.keys():
        chroms = f['chrom'][:100]
        unique_chroms = np.unique(chroms)
        print(f"\nFirst 100 chromosomes (unique): {unique_chroms}")
    
    # Check if chromosome is a separate field or part of region_id
    if 'region_id' in f.keys():
        region_ids = f['region_id'][:20]
        print(f"\nFirst 20 region_ids:")
        for rid in region_ids:
            print(f"  {rid}")

print("\nDone!")
