#!/usr/bin/env python3
"""
Convert NPZ training files to HDF5 format for efficient random access.
Validates data integrity during conversion.

FIXED VERSION: Uses tissue_index based on tissue_top_level (22 classes: 0-21).

If combined_metadata.csv doesn't have tissue_index, creates mapping from
train_files.csv automatically.

Usage:
    python convert_to_hdf5_v2.py
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import warnings

# Configuration
NPZ_DATA_DIR = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/all_data")
METADATA_FILE = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/all_data/combined_metadata.csv")
TRAIN_CSV = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/data_splits/train_files.csv")
VAL_CSV = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/data_splits/val_files.csv")
TEST_CSV = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/data_splits/test_files.csv")
OUTPUT_HDF5 = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5")
EXPECTED_REGIONS = 51089
EXPECTED_SEQ_LENGTH = 150

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def create_tissue_mapping():
    """
    Create tissue mapping from train/val/test CSV files.
    Maps tissue names to indices (0-21).
    """
    print("\n📍 Creating tissue mapping from CSV files...")
    
    # Load all split CSVs
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Combine to get all tissue mappings
    all_splits = pd.concat([train_df, val_df, test_df])
    
    # Create mapping
    tissue_mapping = {}
    for _, row in all_splits.iterrows():
        tissue_name = row['tissue']
        tissue_idx = row['tissue_index']
        if tissue_name not in tissue_mapping:
            tissue_mapping[tissue_name] = tissue_idx
        else:
            # Verify consistency
            if tissue_mapping[tissue_name] != tissue_idx:
                raise ValueError(f"Inconsistent tissue mapping: {tissue_name} -> {tissue_mapping[tissue_name]} vs {tissue_idx}")
    
    print(f"   ✓ Created mapping for {len(tissue_mapping)} tissues")
    print(f"   ✓ Tissue indices: {min(tissue_mapping.values())} to {max(tissue_mapping.values())}")
    
    # Print mapping
    print("\n   Tissue Mapping:")
    for tissue, idx in sorted(tissue_mapping.items(), key=lambda x: x[1]):
        print(f"     {idx:2d}: {tissue}")
    
    return tissue_mapping


def validate_npz_file(npz_path, filename, expected_tissue_idx=None):
    """
    Validate a single NPZ file for data integrity.
    
    Returns:
        dict: Validation results and data info
    """
    results = {
        'filename': filename,
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        data = np.load(npz_path)
        
        # Check required fields
        required_fields = ['dna_tokens', 'methylation', 'n_reads', 'region_ids']
        missing_fields = [f for f in required_fields if f not in data.files]
        if missing_fields:
            results['errors'].append(f"Missing fields: {missing_fields}")
            results['valid'] = False
            return results
        
        # Validate shapes
        dna_tokens = data['dna_tokens']
        methylation = data['methylation']
        n_reads = data['n_reads']
        region_ids = data['region_ids']
        
        # Check dna_tokens shape
        if dna_tokens.shape != (EXPECTED_REGIONS, EXPECTED_SEQ_LENGTH):
            results['errors'].append(
                f"dna_tokens shape mismatch: expected ({EXPECTED_REGIONS}, {EXPECTED_SEQ_LENGTH}), "
                f"got {dna_tokens.shape}"
            )
            results['valid'] = False
        
        # Check methylation shape
        if methylation.shape != (EXPECTED_REGIONS, EXPECTED_SEQ_LENGTH):
            results['errors'].append(
                f"methylation shape mismatch: expected ({EXPECTED_REGIONS}, {EXPECTED_SEQ_LENGTH}), "
                f"got {methylation.shape}"
            )
            results['valid'] = False
        
        # Check n_reads shape
        if n_reads.shape != (EXPECTED_REGIONS,):
            results['errors'].append(
                f"n_reads shape mismatch: expected ({EXPECTED_REGIONS},), got {n_reads.shape}"
            )
            results['valid'] = False
        
        # Check region_ids shape
        if region_ids.shape != (EXPECTED_REGIONS,):
            results['errors'].append(
                f"region_ids shape mismatch: expected ({EXPECTED_REGIONS},), got {region_ids.shape}"
            )
            results['valid'] = False
        
        # Check for NaN/Inf values
        if np.any(np.isnan(dna_tokens)):
            results['errors'].append("dna_tokens contains NaN values")
            results['valid'] = False
        
        if np.any(np.isnan(methylation[methylation < 2])):
            results['errors'].append("methylation contains NaN values (excluding no-CpG positions)")
            results['valid'] = False
        
        if np.any(np.isnan(n_reads)) or np.any(np.isinf(n_reads)):
            results['errors'].append("n_reads contains NaN or Inf values")
            results['valid'] = False
        
        # Check methylation values (should be 0, 1, or 2)
        unique_meth = np.unique(methylation)
        if not np.all(np.isin(unique_meth, [0, 1, 2])):
            results['errors'].append(
                f"methylation contains invalid values: {unique_meth[~np.isin(unique_meth, [0, 1, 2])]}"
            )
            results['valid'] = False
        
        # Collect statistics
        results['stats'] = {
            'n_regions': len(region_ids),
            'total_reads': int(n_reads.sum()),
            'mean_reads': float(n_reads.mean()),
            'median_reads': float(np.median(n_reads)),
            'min_reads': int(n_reads.min()),
            'max_reads': int(n_reads.max()),
            'methylation_rate': float((methylation == 1).sum() / (methylation < 2).sum()),
            'missing_cpg_rate': float((methylation == 2).sum() / methylation.size),
            'n_cpgs': int((methylation < 2).sum()),
        }
        
        # Check for low coverage regions
        low_coverage_regions = (n_reads < 10).sum()
        if low_coverage_regions > 0:
            results['warnings'].append(
                f"{low_coverage_regions} regions with <10 reads ({low_coverage_regions/EXPECTED_REGIONS*100:.1f}%)"
            )
        
    except Exception as e:
        results['errors'].append(f"Error loading file: {str(e)}")
        results['valid'] = False
    
    return results


def convert_to_hdf5():
    """
    Main conversion function: validate and convert all NPZ files to HDF5.
    """
    print("="*80)
    print("NPZ to HDF5 Conversion (Using CSV Tissue Indices)")
    print("="*80)
    
    # Load metadata
    print(f"\n1. Loading metadata from: {METADATA_FILE}")
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    
    metadata = pd.read_csv(METADATA_FILE)
    print(f"   ✓ Loaded {len(metadata)} file entries")
    print(f"   ✓ Unique samples: {metadata['sample_name'].nunique()}")
    
    # Check if tissue_index exists, if not create mapping
    if 'tissue_index' not in metadata.columns:
        print(f"   ⚠ tissue_index column not found in metadata")
        tissue_mapping = create_tissue_mapping()
        
        # Add tissue_index to metadata
        metadata['tissue_index'] = metadata['tissue_top_level'].map(tissue_mapping)
        
        # Check for unmapped
        unmapped = metadata[metadata['tissue_index'].isna()]
        if len(unmapped) > 0:
            print(f"\n   ✗ Error: {len(unmapped)} files have unmapped tissues:")
            for tissue in unmapped['tissue_top_level'].unique():
                print(f"      - {tissue}")
            raise ValueError("Some tissues could not be mapped. Check tissue names in CSV files.")
        
        print(f"   ✓ Successfully mapped all tissues")
    else:
        print(f"   ✓ tissue_index column found in metadata")
    
    n_tissues = int(metadata['tissue_index'].max()) + 1
    unique_tissues = sorted(metadata['tissue_index'].unique())
    print(f"   ✓ Using {n_tissues} tissue types (indices {min(unique_tissues)}-{max(unique_tissues)})")
    
    # Validate all files first
    print(f"\n2. Validating {len(metadata)} NPZ files...")
    validation_results = []
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Validating"):
        npz_path = NPZ_DATA_DIR / row['filename']
        
        if not npz_path.exists():
            validation_results.append({
                'filename': row['filename'],
                'valid': False,
                'errors': [f"File not found: {npz_path}"],
                'warnings': [],
                'stats': {}
            })
            continue
        
        result = validate_npz_file(npz_path, row['filename'])
        validation_results.append(result)
    
    # Summarize validation
    valid_files = [r for r in validation_results if r['valid']]
    invalid_files = [r for r in validation_results if not r['valid']]
    files_with_warnings = [r for r in validation_results if r['warnings']]
    
    print(f"\n3. Validation Summary:")
    print(f"   ✓ Valid files: {len(valid_files)}/{len(metadata)}")
    if invalid_files:
        print(f"   ✗ Invalid files: {len(invalid_files)}")
        print(f"\n   First 5 invalid files:")
        for result in invalid_files[:5]:
            print(f"      - {result['filename']}")
            for error in result['errors']:
                print(f"        ERROR: {error}")
    
    if files_with_warnings:
        print(f"   ⚠ Files with warnings: {len(files_with_warnings)}")
        print(f"\n   First 5 warnings:")
        for result in files_with_warnings[:5]:
            print(f"      - {result['filename']}")
            for warning in result['warnings']:
                print(f"        WARNING: {warning}")
    
    if invalid_files:
        print(f"\n   ✗ Aborting conversion due to invalid files.")
        print(f"   Please fix the errors above and try again.")
        sys.exit(1)
    
    # Print dataset statistics
    print(f"\n4. Dataset Statistics:")
    all_stats = [r['stats'] for r in valid_files if r['stats']]
    
    total_reads = sum(s['total_reads'] for s in all_stats)
    mean_meth_rate = np.mean([s['methylation_rate'] for s in all_stats])
    mean_missing_rate = np.mean([s['missing_cpg_rate'] for s in all_stats])
    
    print(f"   Total regions: {len(all_stats) * EXPECTED_REGIONS:,}")
    print(f"   Total reads: {total_reads:,}")
    print(f"   Mean methylation rate: {mean_meth_rate*100:.2f}%")
    print(f"   Mean missing CpG rate: {mean_missing_rate*100:.2f}% (expected: ~97%)")
    
    # Create HDF5 file
    print(f"\n5. Converting to HDF5: {OUTPUT_HDF5}")
    OUTPUT_HDF5.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(OUTPUT_HDF5, 'w') as h5f:
        # Determine total size for datasets
        n_files = len(valid_files)
        total_regions = n_files * EXPECTED_REGIONS
        
        print(f"   Creating datasets for {n_files} files...")
        print(f"   Total data points: {total_regions:,} regions")
        
        # Create datasets with chunking for efficient access
        dna_tokens_ds = h5f.create_dataset(
            'dna_tokens',
            shape=(0, EXPECTED_REGIONS, EXPECTED_SEQ_LENGTH),
            maxshape=(None, EXPECTED_REGIONS, EXPECTED_SEQ_LENGTH),
            dtype='uint8',
            chunks=(1, EXPECTED_REGIONS, EXPECTED_SEQ_LENGTH),
            compression='gzip',
            compression_opts=4
        )
        
        methylation_ds = h5f.create_dataset(
            'methylation',
            shape=(0, EXPECTED_REGIONS, EXPECTED_SEQ_LENGTH),
            maxshape=(None, EXPECTED_REGIONS, EXPECTED_SEQ_LENGTH),
            dtype='uint8',
            chunks=(1, EXPECTED_REGIONS, EXPECTED_SEQ_LENGTH),
            compression='gzip',
            compression_opts=4
        )
        
        n_reads_ds = h5f.create_dataset(
            'n_reads',
            shape=(0, EXPECTED_REGIONS),
            maxshape=(None, EXPECTED_REGIONS),
            dtype='int32',
            chunks=(1, EXPECTED_REGIONS),
            compression='gzip',
            compression_opts=4
        )
        
        tissue_labels_ds = h5f.create_dataset(
            'tissue_labels',
            shape=(0,),
            maxshape=(None,),
            dtype='uint8',
            chunks=(1,)
        )
        
        # Store file mapping
        filenames = []
        sample_names = []
        aug_versions = []
        tissue_indices = []
        
        # Convert and store data
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Converting"):
            npz_path = NPZ_DATA_DIR / row['filename']
            
            try:
                data = np.load(npz_path)
                
                # Resize datasets
                current_size = dna_tokens_ds.shape[0]
                new_size = current_size + 1
                
                dna_tokens_ds.resize(new_size, axis=0)
                methylation_ds.resize(new_size, axis=0)
                n_reads_ds.resize(new_size, axis=0)
                tissue_labels_ds.resize(new_size, axis=0)
                
                # Write data
                dna_tokens_ds[current_size] = data['dna_tokens']
                methylation_ds[current_size] = data['methylation']
                n_reads_ds[current_size] = data['n_reads']
                
                # Use tissue_index from CSV metadata
                tissue_idx = int(row['tissue_index'])
                tissue_labels_ds[current_size] = tissue_idx
                
                # Store metadata
                filenames.append(row['filename'])
                sample_names.append(row['sample_name'])
                aug_versions.append(row['aug_version'])
                tissue_indices.append(tissue_idx)
                
            except Exception as e:
                print(f"\n   ✗ Error converting {row['filename']}: {e}")
                raise
        
        # Create metadata group
        meta_grp = h5f.create_group('metadata')
        
        # Store metadata as datasets
        meta_grp.create_dataset('filenames', data=np.array(filenames, dtype='S'))
        meta_grp.create_dataset('sample_names', data=np.array(sample_names, dtype='S'))
        meta_grp.create_dataset('aug_versions', data=np.array(aug_versions, dtype='uint8'))
        meta_grp.create_dataset('tissue_indices', data=np.array(tissue_indices, dtype='uint8'))
        
        # Also store tissue names for reference
        tissue_names_list = []
        for idx, row in metadata.iterrows():
            tissue_name = row.get('tissue_top_level', 'Unknown')
            tissue_names_list.append(tissue_name)
        
        meta_grp.create_dataset('tissue_names', data=np.array(tissue_names_list, dtype='S'))
        
        # Store global attributes
        h5f.attrs['n_files'] = n_files
        h5f.attrs['n_regions_per_file'] = EXPECTED_REGIONS
        h5f.attrs['seq_length'] = EXPECTED_SEQ_LENGTH
        h5f.attrs['total_regions'] = total_regions
        h5f.attrs['n_tissues'] = n_tissues
        h5f.attrs['conversion_date'] = pd.Timestamp.now().isoformat()
        h5f.attrs['tissue_source'] = 'CSV tissue_index (22 classes)'
    
    print(f"\n6. Conversion Complete!")
    print(f"   ✓ HDF5 file created: {OUTPUT_HDF5}")
    print(f"   ✓ File size: {OUTPUT_HDF5.stat().st_size / 1e9:.2f} GB")
    
    # Verify HDF5 file
    print(f"\n7. Verifying HDF5 file...")
    with h5py.File(OUTPUT_HDF5, 'r') as h5f:
        print(f"   ✓ dna_tokens: {h5f['dna_tokens'].shape}")
        print(f"   ✓ methylation: {h5f['methylation'].shape}")
        print(f"   ✓ n_reads: {h5f['n_reads'].shape}")
        print(f"   ✓ tissue_labels: {h5f['tissue_labels'].shape}")
        print(f"   ✓ n_tissues attribute: {h5f.attrs['n_tissues']}")
        
        # Check tissue label range
        tissue_labels = h5f['tissue_labels'][:]
        print(f"\n   Tissue label verification:")
        print(f"     - Min: {tissue_labels.min()}")
        print(f"     - Max: {tissue_labels.max()}")
        print(f"     - Unique: {len(np.unique(tissue_labels))}")
        print(f"     - Range: [{tissue_labels.min()}, {tissue_labels.max()}]")
        
        # Quick test read
        print(f"\n   Testing random access...")
        test_idx = np.random.randint(0, h5f['dna_tokens'].shape[0])
        test_region = np.random.randint(0, EXPECTED_REGIONS)
        
        dna = h5f['dna_tokens'][test_idx, test_region]
        meth = h5f['methylation'][test_idx, test_region]
        tissue = h5f['tissue_labels'][test_idx]
        
        print(f"   ✓ Random sample [file={test_idx}, region={test_region}]")
        print(f"     DNA tokens shape: {dna.shape}")
        print(f"     Methylation shape: {meth.shape}")
        print(f"     Tissue label: {tissue}")
    
    print("\n" + "="*80)
    print("✅ Conversion successful! HDF5 file ready for training.")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"  - Files converted: {n_files}")
    print(f"  - Tissue classes: {n_tissues} (0-{n_tissues-1})")
    print(f"  - Total regions: {total_regions:,}")
    print(f"  - File size: {OUTPUT_HDF5.stat().st_size / 1e9:.2f} GB")
    print()


if __name__ == "__main__":
    try:
        convert_to_hdf5()
    except KeyboardInterrupt:
        print("\n\nConversion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
