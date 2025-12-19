#!/usr/bin/env python3
"""
Generate Stage 2 Blood-Masked Validation/Test Datasets
=======================================================

Creates pre-computed validation and test datasets for Stage 2 training.

Key difference from Phase 3:
- Input: Same mixed methylation (WITH blood: 60-100%)
- Output: Blood-masked proportions (21 tissues, renormalized)

Generates 2 HDF5 files:
- stage2_validation_mixtures.h5 (1500 samples)
- stage2_test_mixtures.h5 (1500 samples)

Author: Stage 2 Blood Deconvolution
Date: December 2024
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import json

VAL_SEED = 44  # Different from Phase 3
TEST_SEED = 45


def load_data(hdf5_path: str, metadata_path: str) -> Tuple[h5py.File, pd.DataFrame]:
    """Load HDF5 and metadata"""
    print(f"Loading data from:\n  HDF5: {hdf5_path}\n  Metadata: {metadata_path}")
    
    h5_file = h5py.File(hdf5_path, 'r')
    metadata = pd.read_csv(metadata_path)
    
    print(f"✓ Loaded {len(metadata)} samples")
    print(f"✓ Methylation shape: {h5_file['methylation'].shape}")
    
    return h5_file, metadata


def compute_region_means(methylation: np.ndarray) -> np.ndarray:
    """Compute region means from methylation patterns"""
    valid_mask = (methylation != 2).astype(float)
    region_means = np.sum(methylation * valid_mask, axis=1) / (np.sum(valid_mask, axis=1) + 1e-8)
    return region_means


def get_tissue_samples(metadata: pd.DataFrame, tissue: str, aug_version: int) -> pd.DataFrame:
    """Get samples for specific tissue and augmentation"""
    samples = metadata[
        (metadata['tissue_top_level'] == tissue) &
        (metadata['aug_version'] == aug_version) &
        (metadata['is_synthetic'] == False)
    ]
    
    if len(samples) == 0:
        samples = metadata[
            (metadata['tissue_top_level'] == tissue) &
            (metadata['aug_version'] == aug_version)
        ]
    
    return samples


def generate_bloodmasked_mixtures(
    h5_file: h5py.File,
    metadata: pd.DataFrame,
    n_mixtures: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate blood-masked mixtures for Stage 2
    
    Strategy:
    1. Generate realistic cfDNA mixtures (60-100% blood + other tissues)
    2. Mix methylation INCLUDING blood
    3. Create labels EXCLUDING blood (renormalized)
    
    Returns:
        mixed_methylation: [n_mixtures, 51089] - includes blood signal
        true_proportions: [n_mixtures, 21] - excludes blood, renormalized
        mixture_info: List of mixture metadata
    """
    np.random.seed(seed)
    
    # Get tissue information
    all_tissues = sorted(metadata['tissue_top_level'].unique())  # 22 tissues
    blood_idx_full = all_tissues.index('Blood')
    
    # Non-blood tissues for labels
    nonblood_tissues = [t for t in all_tissues if t != 'Blood']  # 21 tissues
    tissue_to_idx_nonblood = {tissue: idx for idx, tissue in enumerate(nonblood_tissues)}
    
    print(f"\nGenerating {n_mixtures} Stage 2 blood-masked mixtures...")
    print(f"  Total tissues: {len(all_tissues)}")
    print(f"  Output tissues: {len(nonblood_tissues)} (blood masked)")
    
    mixed_methylation_list = []
    true_proportions_list = []
    mixture_info_list = []
    
    mixtures_generated = 0
    max_attempts = n_mixtures * 10
    attempts = 0
    
    while mixtures_generated < n_mixtures and attempts < max_attempts:
        attempts += 1
        
        # Random augmentation
        aug_version = np.random.randint(0, 5)
        
        # Blood proportion (60-100%)
        blood_prop = np.random.uniform(0.60, 1.0)
        
        # Number of other tissues (1-5)
        n_other = np.random.randint(1, 6)
        
        # Select other tissues (non-blood)
        available_other = [t for t in nonblood_tissues 
                          if len(get_tissue_samples(metadata, t, aug_version)) > 0]
        
        if len(available_other) < n_other:
            continue
        
        selected_other = np.random.choice(available_other, size=n_other, replace=False).tolist()
        
        # Generate proportions for other tissues
        remaining_prop = 1.0 - blood_prop
        alpha = np.ones(n_other) * 0.5
        other_proportions = np.random.dirichlet(alpha) * remaining_prop
        
        # === LOAD AND MIX (includes blood) ===
        try:
            # Blood component
            blood_samples = get_tissue_samples(metadata, 'Blood', aug_version)
            if len(blood_samples) == 0:
                continue
            blood_sample_idx = blood_samples.sample(1).index[0]
            blood_meth = h5_file['methylation'][blood_sample_idx]
            blood_means = compute_region_means(blood_meth)
            
            # Start with blood
            mixed_region_means = blood_prop * blood_means
            
            # Add other tissues
            selected_indices = {'Blood': int(blood_sample_idx)}
            
            for tissue, prop in zip(selected_other, other_proportions):
                tissue_samples = get_tissue_samples(metadata, tissue, aug_version)
                if len(tissue_samples) == 0:
                    raise ValueError(f"No samples for {tissue}")
                
                sample_idx = tissue_samples.sample(1).index[0]
                methylation = h5_file['methylation'][sample_idx]
                region_means = compute_region_means(methylation)
                
                mixed_region_means += prop * region_means
                selected_indices[tissue] = int(sample_idx)
        
        except Exception as e:
            continue
        
        # === CREATE BLOOD-MASKED LABELS ===
        # Only non-blood tissues, renormalized to 1.0
        proportions_masked = np.zeros(len(nonblood_tissues), dtype=np.float32)
        
        for tissue, prop in zip(selected_other, other_proportions):
            proportions_masked[tissue_to_idx_nonblood[tissue]] = prop
        
        # Renormalize
        if proportions_masked.sum() > 0:
            proportions_masked = proportions_masked / proportions_masked.sum()
        else:
            # Edge case: 100% blood → uniform over non-blood
            proportions_masked = np.ones(len(nonblood_tissues), dtype=np.float32) / len(nonblood_tissues)
        
        # Verify
        assert np.isclose(proportions_masked.sum(), 1.0), f"Proportions sum to {proportions_masked.sum()}"
        
        # Store
        mixed_methylation_list.append(mixed_region_means)
        true_proportions_list.append(proportions_masked)
        mixture_info_list.append({
            'mixture_id': int(mixtures_generated),
            'blood_proportion_in_mixture': float(blood_prop),  # For reference only
            'n_other_tissues': int(n_other),
            'other_tissues': [str(t) for t in selected_other],
            'other_proportions_masked': [float(p) for p in proportions_masked[proportions_masked > 0]],
            'sample_indices': selected_indices,
            'aug_version': int(aug_version)
        })
        
        mixtures_generated += 1
        
        if mixtures_generated % 100 == 0:
            print(f"  Generated {mixtures_generated}/{n_mixtures} mixtures...")
    
    if mixtures_generated < n_mixtures:
        print(f"  WARNING: Only generated {mixtures_generated}/{n_mixtures} after {attempts} attempts")
    
    mixed_methylation = np.array(mixed_methylation_list, dtype=np.float32)
    true_proportions = np.array(true_proportions_list, dtype=np.float32)
    
    print(f"✓ Generated {mixtures_generated} Stage 2 mixtures")
    print(f"  Shape: mixed_methylation={mixed_methylation.shape}, proportions={true_proportions.shape}")
    
    return mixed_methylation, true_proportions, mixture_info_list


def save_mixture_dataset(
    output_path: str,
    mixed_methylation: np.ndarray,
    true_proportions: np.ndarray,
    mixture_info: List[Dict],
    tissue_names: List[str],
    split: str
):
    """Save blood-masked mixture dataset to HDF5"""
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
        f.attrs['phase'] = 'stage2_bloodmasked'
        f.attrs['split'] = split
        f.attrs['description'] = 'Blood-masked mixtures: input has blood, labels exclude blood'
        
        # Save mixture info
        mixture_info_json = [json.dumps(info) for info in mixture_info]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('mixture_info', data=mixture_info_json, dtype=dt)
    
    print(f"✓ Saved {len(mixed_methylation)} mixtures")
    print(f"  Datasets: mixed_methylation {mixed_methylation.shape}, true_proportions {true_proportions.shape}")


def main():
    parser = argparse.ArgumentParser(description='Generate Stage 2 blood-masked mixture datasets')
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
    print("STAGE 2 BLOOD-MASKED MIXTURE DATASET GENERATION")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    h5_file, metadata = load_data(args.hdf5, args.metadata)
    
    # Get tissue names (non-blood only)
    all_tissues = sorted(metadata['tissue_top_level'].unique())
    tissue_names = [t for t in all_tissues if t != 'Blood']
    
    print(f"\nAll tissues ({len(all_tissues)}): {', '.join(all_tissues)}")
    print(f"Output tissues ({len(tissue_names)}): {', '.join(tissue_names)}")
    
    # Generate validation set
    print(f"\n{'='*80}")
    print(f"VALIDATION SET (n=1500)")
    print(f"{'='*80}")
    
    val_mixed, val_props, val_info = generate_bloodmasked_mixtures(
        h5_file, metadata, n_mixtures=1500, seed=VAL_SEED
    )
    
    val_path = output_dir / 'stage2_validation_mixtures.h5'
    save_mixture_dataset(val_path, val_mixed, val_props, val_info, tissue_names, 'validation')
    
    # Generate test set
    print(f"\n{'='*80}")
    print(f"TEST SET (n=1500)")
    print(f"{'='*80}")
    
    test_mixed, test_props, test_info = generate_bloodmasked_mixtures(
        h5_file, metadata, n_mixtures=1500, seed=TEST_SEED
    )
    
    test_path = output_dir / 'stage2_test_mixtures.h5'
    save_mixture_dataset(test_path, test_mixed, test_props, test_info, tissue_names, 'test')
    
    # Close HDF5
    h5_file.close()
    
    print(f"\n{'='*80}")
    print("STAGE 2 DATASET GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"\nThese datasets can now be used for Stage 2 training.")


if __name__ == '__main__':
    main()
