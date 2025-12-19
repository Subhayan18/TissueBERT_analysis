#!/usr/bin/env python3
"""
Stage 2 Blood-Masked Mixture Generator - FIXED VERSION
=======================================================

Key fix: Use ABSOLUTE proportions, not renormalized!

Training strategy:
- Input: Mixed methylation (includes blood signal 60-100%)
- Labels: Non-blood tissues with ABSOLUTE proportions (sum < 1.0)

Example:
  Mixture: 90% Blood + 5% Liver + 3% Pancreas + 2% Lung
  
  OLD (broken):
    Labels: [0.50 Liver, 0.30 Pancreas, 0.20 Lung]  # Renormalized
    Problem: Model sees 90% blood but told it's 50% liver!
  
  NEW (fixed):
    Labels: [0.05 Liver, 0.03 Pancreas, 0.02 Lung]  # Absolute
    Model learns: "5% liver signal in blood-dominant mixture"

This trains the model to detect TRACE signals in blood-dominated samples.

Author: Stage 2 Residual Deconvolution
Date: December 2024
"""

# Fix numpy/h5py compatibility issue
import sys
import subprocess

def check_and_fix_dependencies():
    """Check and fix numpy/h5py compatibility"""
    try:
        import numpy as np
        import h5py
        # Test if h5py works
        _ = h5py.File
        print("✓ h5py loaded successfully")
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            print("⚠️  Detected numpy/h5py incompatibility")
            print("  Reinstalling h5py...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                 "--upgrade", "--force-reinstall", "h5py", 
                                 "--break-system-packages"])
            print("✓ h5py reinstalled")
            # Re-import
            import h5py
        else:
            raise

# Fix dependencies before importing
check_and_fix_dependencies()

import numpy as np
import h5py
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def load_data(hdf5_path: str, metadata_csv: str):
    """Load methylation data and metadata"""
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    # Load HDF5
    print(f"\nLoading: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        n_samples = f['methylation'].shape[0]
        n_regions = f['methylation'].shape[1]
        seq_length = f['methylation'].shape[2]
    
    print(f"✓ HDF5 loaded: {n_samples} samples, {n_regions} regions, {seq_length} positions")
    
    # Load metadata
    print(f"\nLoading: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)
    print(f"✓ Metadata loaded: {len(metadata)} rows")
    
    # Get tissue information
    tissue_counts = metadata['tissue_top_level'].value_counts()
    print(f"\nTissue distribution:")
    for tissue, count in tissue_counts.items():
        print(f"  {tissue}: {count}")
    
    return metadata, n_regions, seq_length


def organize_samples_by_tissue(metadata: pd.DataFrame) -> Tuple[Dict, List[str]]:
    """Organize samples by tissue and augmentation version"""
    tissue_samples = defaultdict(lambda: defaultdict(list))
    
    for idx, row in metadata.iterrows():
        tissue = row['tissue_top_level']
        aug = row['aug_version']
        tissue_samples[tissue][aug].append(idx)
    
    # Get unique tissues (excluding Blood for labels, but keeping for mixing)
    all_tissues = sorted(tissue_samples.keys())
    non_blood_tissues = [t for t in all_tissues if t != 'Blood']
    
    print(f"\n✓ Organized samples:")
    print(f"  Total tissues: {len(all_tissues)}")
    print(f"  Non-blood tissues: {len(non_blood_tissues)}")
    
    return tissue_samples, all_tissues, non_blood_tissues


def generate_stage2_absolute_mixtures(
    hdf5_path: str,
    tissue_samples: Dict,
    all_tissues: List[str],
    non_blood_tissues: List[str],
    n_mixtures: int,
    n_regions: int,
    blood_range: Tuple[float, float] = (0.60, 0.95),
    n_tissue_range: Tuple[int, int] = (2, 5)
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate Stage 2 mixtures with ABSOLUTE labels (not renormalized)
    
    Key change: Labels are absolute proportions that sum to (1 - blood_prop)
    """
    print(f"\n{'='*80}")
    print("GENERATING STAGE 2 MIXTURES (ABSOLUTE LABELS)")
    print(f"{'='*80}")
    print(f"Target: {n_mixtures} mixtures")
    print(f"Blood range: {blood_range[0]*100:.0f}-{blood_range[1]*100:.0f}%")
    print(f"Non-blood tissues per mixture: {n_tissue_range[0]}-{n_tissue_range[1]}")
    print(f"\n⚠️  CRITICAL: Labels are ABSOLUTE proportions (not renormalized)")
    print(f"    Example: [0.05 Liver, 0.03 Pancreas] sums to 0.08 (not 1.0)")
    
    mixed_methylation_list = []
    true_proportions_list = []
    mixture_info_list = []
    
    # Create tissue name to index mapping (non-blood only)
    tissue_to_idx = {tissue: idx for idx, tissue in enumerate(non_blood_tissues)}
    n_nonblood_tissues = len(non_blood_tissues)
    
    attempts = 0
    max_attempts = n_mixtures * 10
    mixtures_generated = 0
    
    with h5py.File(hdf5_path, 'r') as f:
        while mixtures_generated < n_mixtures and attempts < max_attempts:
            attempts += 1
            
            try:
                # 1. Choose blood proportion
                blood_prop = np.random.uniform(blood_range[0], blood_range[1])
                
                # 2. Remaining proportion for non-blood tissues
                remaining_prop = 1.0 - blood_prop
                
                # 3. Choose number of non-blood tissues
                n_tissues = np.random.randint(n_tissue_range[0], n_tissue_range[1] + 1)
                
                # 4. Select random non-blood tissues
                selected_tissues = np.random.choice(non_blood_tissues, size=n_tissues, replace=False)
                
                # 5. Generate proportions for selected tissues using Dirichlet
                tissue_props = np.random.dirichlet(np.ones(n_tissues))
                tissue_props = tissue_props * remaining_prop  # Scale to remaining proportion
                
                # 6. Load blood sample
                blood_aug = np.random.randint(0, 5)
                if blood_aug not in tissue_samples['Blood'] or len(tissue_samples['Blood'][blood_aug]) == 0:
                    continue
                blood_idx = np.random.choice(tissue_samples['Blood'][blood_aug])
                blood_meth = f['methylation'][blood_idx]  # [n_regions, seq_length]
                
                # 7. Compute region means for blood
                blood_means = blood_meth.mean(axis=1)  # [n_regions]
                
                # 8. Initialize mixture
                mixed_means = blood_prop * blood_means
                
                # 9. Add non-blood tissues
                tissue_contributions = {'Blood': blood_prop}
                
                for tissue, prop in zip(selected_tissues, tissue_props):
                    # Get sample
                    aug = np.random.randint(0, 5)
                    if aug not in tissue_samples[tissue] or len(tissue_samples[tissue][aug]) == 0:
                        aug = np.random.choice([a for a in tissue_samples[tissue] 
                                              if len(tissue_samples[tissue][a]) > 0])
                    
                    tissue_idx = np.random.choice(tissue_samples[tissue][aug])
                    tissue_meth = f['methylation'][tissue_idx]
                    tissue_means = tissue_meth.mean(axis=1)
                    
                    # Add to mixture
                    mixed_means += prop * tissue_means
                    tissue_contributions[tissue] = float(prop)
                
                # 10. Create labels: ABSOLUTE proportions (exclude blood, DON'T renormalize!)
                proportions_absolute = np.zeros(n_nonblood_tissues, dtype=np.float32)
                
                for tissue, prop in tissue_contributions.items():
                    if tissue != 'Blood':
                        proportions_absolute[tissue_to_idx[tissue]] = prop
                
                # CRITICAL: Do NOT renormalize! Labels should sum to (1 - blood_prop)
                label_sum = proportions_absolute.sum()
                expected_sum = 1.0 - blood_prop
                
                # Sanity check
                if not np.isclose(label_sum, expected_sum, atol=0.01):
                    print(f"Warning: Label sum {label_sum:.3f} != expected {expected_sum:.3f}")
                    continue
                
                # Store
                mixed_methylation_list.append(mixed_means)
                true_proportions_list.append(proportions_absolute)
                mixture_info_list.append({
                    'blood_proportion': float(blood_prop),
                    'n_nonblood_tissues': int(n_tissues),
                    'tissue_contributions': tissue_contributions,
                    'label_sum': float(label_sum),  # Should be ~(1 - blood_prop)
                    'absolute_labels': True  # Flag to indicate non-renormalized
                })
                
                mixtures_generated += 1
                
            except Exception as e:
                continue
            
            if mixtures_generated % 100 == 0:
                print(f"  Generated {mixtures_generated}/{n_mixtures} mixtures...")
    
    mixed_methylation = np.array(mixed_methylation_list, dtype=np.float32)
    true_proportions = np.array(true_proportions_list, dtype=np.float32)
    
    print(f"\n✓ Generated {mixtures_generated} Stage 2 mixtures")
    print(f"  Shape: mixed_methylation={mixed_methylation.shape}, proportions={true_proportions.shape}")
    print(f"\n  Label statistics:")
    print(f"    Mean label sum: {true_proportions.sum(axis=1).mean():.4f}")
    print(f"    Label sum range: [{true_proportions.sum(axis=1).min():.4f}, {true_proportions.sum(axis=1).max():.4f}]")
    print(f"    Expected range: [0.05, 0.40] (1 - blood_prop)")
    
    return mixed_methylation, true_proportions, mixture_info_list


def save_mixture_dataset(
    output_path: str,
    mixed_methylation: np.ndarray,
    true_proportions: np.ndarray,
    mixture_info: List[Dict],
    tissue_names: List[str],
    split: str
):
    """Save absolute-label mixture dataset to HDF5"""
    print(f"\nSaving to: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Save arrays
        f.create_dataset('mixed_methylation', data=mixed_methylation, compression='gzip')
        f.create_dataset('true_proportions', data=true_proportions, compression='gzip')
        
        # Save attributes
        f.attrs['tissue_names'] = [t.encode('utf-8') for t in tissue_names]
        f.attrs['n_tissues'] = len(tissue_names)
        f.attrs['n_regions'] = mixed_methylation.shape[1]
        f.attrs['n_mixtures'] = len(mixed_methylation)
        f.attrs['phase'] = 'stage2_absolute'
        f.attrs['split'] = split
        f.attrs['description'] = 'Stage 2 ABSOLUTE labels: input has blood, labels are absolute non-blood proportions (sum < 1.0)'
        f.attrs['label_type'] = 'absolute'  # IMPORTANT FLAG
        
        # Save mixture info
        mixture_info_json = [json.dumps(info) for info in mixture_info]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('mixture_info', data=mixture_info_json, dtype=dt)
    
    print(f"✓ Saved {len(mixed_methylation)} mixtures")
    print(f"  Label type: ABSOLUTE (not renormalized)")


def main():
    parser = argparse.ArgumentParser(description='Generate Stage 2 mixtures with ABSOLUTE labels')
    parser.add_argument('--hdf5', type=str, required=True,
                        help='Path to methylation_dataset.h5')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to combined_metadata.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for mixture datasets')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("STAGE 2 MIXTURE GENERATION - ABSOLUTE LABELS (FIXED)")
    print("="*80)
    print(f"\nInput HDF5: {args.hdf5}")
    print(f"Metadata: {args.metadata}")
    print(f"Output: {args.output_dir}")
    
    # Load data
    metadata, n_regions, seq_length = load_data(args.hdf5, args.metadata)
    tissue_samples, all_tissues, non_blood_tissues = organize_samples_by_tissue(metadata)
    
    # Generate validation set
    print(f"\n{'='*80}")
    print("GENERATING VALIDATION SET")
    print(f"{'='*80}")
    val_meth, val_props, val_info = generate_stage2_absolute_mixtures(
        args.hdf5, tissue_samples, all_tissues, non_blood_tissues,
        n_mixtures=1500, n_regions=n_regions
    )
    
    # Save validation
    val_path = output_dir / 'stage2_validation_absolute.h5'
    save_mixture_dataset(val_path, val_meth, val_props, val_info, 
                        non_blood_tissues, 'validation')
    
    # Generate test set
    print(f"\n{'='*80}")
    print("GENERATING TEST SET")
    print(f"{'='*80}")
    test_meth, test_props, test_info = generate_stage2_absolute_mixtures(
        args.hdf5, tissue_samples, all_tissues, non_blood_tissues,
        n_mixtures=1500, n_regions=n_regions
    )
    
    # Save test
    test_path = output_dir / 'stage2_test_absolute.h5'
    save_mixture_dataset(test_path, test_meth, test_props, test_info,
                        non_blood_tissues, 'test')
    
    print(f"\n{'='*80}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")
    print(f"\n⚠️  IMPORTANT: These datasets use ABSOLUTE labels")
    print(f"    - Labels sum to (1 - blood_prop), not 1.0")
    print(f"    - Model must use MSE loss (not CrossEntropy)")
    print(f"    - No softmax in final layer (use sigmoid or linear)")
    print(f"="*80)


if __name__ == '__main__':
    main()
