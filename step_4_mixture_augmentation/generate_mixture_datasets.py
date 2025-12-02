#!/usr/bin/env python3
"""
Generate Synthetic Mixture Datasets for Tissue Deconvolution Training
======================================================================

This script generates pre-computed validation and test datasets for all three
training phases. Training data is generated on-the-fly during training.

Generates 6 HDF5 files:
- Phase 1 (2-tissue): validation (500) + test (500)
- Phase 2 (3-5 tissue): validation (1000) + test (1000)
- Phase 3 (realistic cfDNA): validation (1500) + test (1500)

Author: Mixture Deconvolution Project
Date: December 2024
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import sys

# Seed configuration
VAL_SEED = 42
TEST_SEED = 43

# Dataset sizes per phase
PHASE_CONFIGS = {
    1: {'val_size': 500, 'test_size': 500, 'description': '2-tissue mixtures'},
    2: {'val_size': 1000, 'test_size': 1000, 'description': '3-5 tissue mixtures'},
    3: {'val_size': 1500, 'test_size': 1500, 'description': 'Realistic cfDNA mixtures'}
}

# Proportion strategies for Phase 1 (2-tissue)
PHASE1_PROPORTIONS = [
    (0.5, 0.5),   # 50-50
    (0.6, 0.4),   # 60-40
    (0.4, 0.6),   # 40-60
    (0.7, 0.3),   # 70-30
    (0.3, 0.7),   # 30-70
    (0.8, 0.2),   # 80-20
    (0.2, 0.8),   # 20-80
]


def load_data(hdf5_path: str, metadata_path: str) -> Tuple[h5py.File, pd.DataFrame]:
    """Load HDF5 and metadata."""
    print(f"Loading data from:\n  HDF5: {hdf5_path}\n  Metadata: {metadata_path}")
    
    h5_file = h5py.File(hdf5_path, 'r')
    metadata = pd.read_csv(metadata_path)
    
    print(f"✓ Loaded {len(metadata)} samples")
    print(f"✓ Methylation shape: {h5_file['methylation'].shape}")
    
    return h5_file, metadata


def compute_region_means(methylation: np.ndarray) -> np.ndarray:
    """
    Compute region means from methylation patterns.
    
    Args:
        methylation: [51089, 150] array with values {0, 1, 2}
                    0/1 = unmeth/meth, 2 = missing
    
    Returns:
        region_means: [51089] array of mean methylation per region
    """
    valid_mask = (methylation != 2).astype(float)
    region_means = np.sum(methylation * valid_mask, axis=1) / (np.sum(valid_mask, axis=1) + 1e-8)
    return region_means


def get_tissue_samples(metadata: pd.DataFrame, tissue: str, aug_version: int) -> pd.DataFrame:
    """Get all samples for a specific tissue and augmentation version."""
    samples = metadata[
        (metadata['tissue_top_level'] == tissue) & 
        (metadata['aug_version'] == aug_version) &
        (metadata['is_synthetic'] == False)  # Prefer real samples
    ]
    
    # If no real samples, allow synthetic
    if len(samples) == 0:
        samples = metadata[
            (metadata['tissue_top_level'] == tissue) & 
            (metadata['aug_version'] == aug_version)
        ]
    
    return samples


def generate_phase1_mixtures(
    h5_file: h5py.File,
    metadata: pd.DataFrame,
    n_mixtures: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate 2-tissue mixtures for Phase 1.
    
    Strategy:
    - Sample all possible tissue pairs
    - Use various proportion splits (50-50, 60-40, 70-30, 80-20, etc.)
    - Ensure tissues are different
    - Use same augmentation version
    
    Returns:
        mixed_methylation: [N, 51089] array of mixed region means
        true_proportions: [N, 22] array of ground truth proportions
        mixture_info: List of metadata dicts
    """
    print(f"\nGenerating Phase 1 mixtures (n={n_mixtures}, seed={seed})...")
    
    np.random.seed(seed)
    
    unique_tissues = sorted(metadata['tissue_top_level'].unique())
    n_tissues = len(unique_tissues)
    tissue_to_idx = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
    
    print(f"  Available tissues: {n_tissues}")
    
    mixed_methylation_list = []
    true_proportions_list = []
    mixture_info_list = []
    
    mixtures_generated = 0
    attempts = 0
    max_attempts = n_mixtures * 100
    
    while mixtures_generated < n_mixtures and attempts < max_attempts:
        attempts += 1
        
        # Sample two different tissues
        tissue_a, tissue_b = np.random.choice(unique_tissues, size=2, replace=False)
        
        # Sample augmentation version (0-4)
        aug_version = np.random.randint(0, 5)
        
        # Get samples for each tissue
        samples_a = get_tissue_samples(metadata, tissue_a, aug_version)
        samples_b = get_tissue_samples(metadata, tissue_b, aug_version)
        
        if len(samples_a) == 0 or len(samples_b) == 0:
            continue
        
        # Randomly select one sample from each tissue
        sample_a_idx = np.random.choice(samples_a.index)
        sample_b_idx = np.random.choice(samples_b.index)
        
        # Select proportion strategy
        prop_a, prop_b = PHASE1_PROPORTIONS[np.random.randint(len(PHASE1_PROPORTIONS))]
        
        # Load methylation and compute region means
        meth_a = h5_file['methylation'][sample_a_idx]
        meth_b = h5_file['methylation'][sample_b_idx]
        
        region_means_a = compute_region_means(meth_a)
        region_means_b = compute_region_means(meth_b)
        
        # Linear mixing
        mixed_region_means = prop_a * region_means_a + prop_b * region_means_b
        
        # Create proportion vector [22]
        proportions = np.zeros(n_tissues, dtype=np.float32)
        proportions[tissue_to_idx[tissue_a]] = prop_a
        proportions[tissue_to_idx[tissue_b]] = prop_b
        
        # Verify proportions sum to 1.0
        assert np.isclose(proportions.sum(), 1.0), f"Proportions sum to {proportions.sum()}"
        
        # Store
        mixed_methylation_list.append(mixed_region_means)
        true_proportions_list.append(proportions)
        mixture_info_list.append({
            'mixture_id': int(mixtures_generated),
            'tissue_a': tissue_a,
            'tissue_b': tissue_b,
            'prop_a': float(prop_a),
            'prop_b': float(prop_b),
            'sample_a_idx': int(sample_a_idx),
            'sample_b_idx': int(sample_b_idx),
            'aug_version': int(aug_version)
        })
        
        mixtures_generated += 1
        
        if mixtures_generated % 100 == 0:
            print(f"  Generated {mixtures_generated}/{n_mixtures} mixtures...")
    
    if mixtures_generated < n_mixtures:
        print(f"  WARNING: Only generated {mixtures_generated}/{n_mixtures} mixtures after {attempts} attempts")
    
    mixed_methylation = np.array(mixed_methylation_list, dtype=np.float32)
    true_proportions = np.array(true_proportions_list, dtype=np.float32)
    
    print(f"✓ Generated {mixtures_generated} Phase 1 mixtures")
    print(f"  Shape: mixed_methylation={mixed_methylation.shape}, proportions={true_proportions.shape}")
    
    return mixed_methylation, true_proportions, mixture_info_list


def generate_phase2_mixtures(
    h5_file: h5py.File,
    metadata: pd.DataFrame,
    n_mixtures: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate 3-5 tissue mixtures for Phase 2.
    
    Strategy:
    - Randomly sample 3, 4, or 5 different tissues
    - Use Dirichlet distribution for proportions (alpha=2.0 for moderate diversity)
    - Ensure all tissues are different
    - Use same augmentation version
    
    Returns:
        mixed_methylation: [N, 51089] array of mixed region means
        true_proportions: [N, 22] array of ground truth proportions
        mixture_info: List of metadata dicts
    """
    print(f"\nGenerating Phase 2 mixtures (n={n_mixtures}, seed={seed})...")
    
    np.random.seed(seed)
    
    unique_tissues = sorted(metadata['tissue_top_level'].unique())
    n_tissues = len(unique_tissues)
    tissue_to_idx = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
    
    mixed_methylation_list = []
    true_proportions_list = []
    mixture_info_list = []
    
    mixtures_generated = 0
    attempts = 0
    max_attempts = n_mixtures * 100
    
    while mixtures_generated < n_mixtures and attempts < max_attempts:
        attempts += 1
        
        # Randomly choose number of tissues (3-5)
        n_components = np.random.randint(3, 6)
        
        # Sample n_components different tissues
        selected_tissues = np.random.choice(unique_tissues, size=n_components, replace=False)
        
        # Sample augmentation version
        aug_version = np.random.randint(0, 5)
        
        # Get samples for each tissue
        tissue_samples = {}
        valid = True
        
        for tissue in selected_tissues:
            samples = get_tissue_samples(metadata, tissue, aug_version)
            if len(samples) == 0:
                valid = False
                break
            tissue_samples[tissue] = samples
        
        if not valid:
            continue
        
        # Generate proportions using Dirichlet distribution
        alpha = np.ones(n_components) * 2.0  # Moderate diversity
        proportions_components = np.random.dirichlet(alpha)
        
        # Randomly select one sample from each tissue and mix
        mixed_region_means = np.zeros(51089, dtype=np.float32)
        selected_indices = {}
        
        for tissue, prop in zip(selected_tissues, proportions_components):
            sample_idx = np.random.choice(tissue_samples[tissue].index)
            selected_indices[tissue] = sample_idx
            
            meth = h5_file['methylation'][sample_idx]
            region_means = compute_region_means(meth)
            
            mixed_region_means += prop * region_means
        
        # Create full proportion vector [22]
        proportions = np.zeros(n_tissues, dtype=np.float32)
        for tissue, prop in zip(selected_tissues, proportions_components):
            proportions[tissue_to_idx[tissue]] = prop
        
        # Verify proportions sum to 1.0
        assert np.isclose(proportions.sum(), 1.0), f"Proportions sum to {proportions.sum()}"
        
        # Store
        mixed_methylation_list.append(mixed_region_means)
        true_proportions_list.append(proportions)
        mixture_info_list.append({
            'mixture_id': int(mixtures_generated),
            'n_components': int(n_components),
            'tissues': [str(t) for t in selected_tissues],
            'proportions': [float(p) for p in proportions_components],
            'sample_indices': {str(k): int(v) for k, v in selected_indices.items()},
            'aug_version': int(aug_version)
        })
        
        mixtures_generated += 1
        
        if mixtures_generated % 100 == 0:
            print(f"  Generated {mixtures_generated}/{n_mixtures} mixtures...")
    
    if mixtures_generated < n_mixtures:
        print(f"  WARNING: Only generated {mixtures_generated}/{n_mixtures} mixtures after {attempts} attempts")
    
    mixed_methylation = np.array(mixed_methylation_list, dtype=np.float32)
    true_proportions = np.array(true_proportions_list, dtype=np.float32)
    
    print(f"✓ Generated {mixtures_generated} Phase 2 mixtures")
    print(f"  Shape: mixed_methylation={mixed_methylation.shape}, proportions={true_proportions.shape}")
    
    return mixed_methylation, true_proportions, mixture_info_list


def generate_phase3_mixtures(
    h5_file: h5py.File,
    metadata: pd.DataFrame,
    n_mixtures: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate realistic cfDNA-like mixtures for Phase 3.
    
    Strategy:
    - Blood is always dominant (60-90%)
    - 2-6 additional tissues with smaller proportions
    - Simulates cancer patient cfDNA composition
    - Uses Beta distribution for realistic proportion skew
    
    Returns:
        mixed_methylation: [N, 51089] array of mixed region means
        true_proportions: [N, 22] array of ground truth proportions
        mixture_info: List of metadata dicts
    """
    print(f"\nGenerating Phase 3 mixtures (n={n_mixtures}, seed={seed})...")
    
    np.random.seed(seed)
    
    unique_tissues = sorted(metadata['tissue_top_level'].unique())
    n_tissues = len(unique_tissues)
    tissue_to_idx = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
    
    # Ensure Blood is available
    if 'Blood' not in unique_tissues:
        raise ValueError("Blood tissue not found in dataset - required for Phase 3")
    
    non_blood_tissues = [t for t in unique_tissues if t != 'Blood']
    
    mixed_methylation_list = []
    true_proportions_list = []
    mixture_info_list = []
    
    mixtures_generated = 0
    attempts = 0
    max_attempts = n_mixtures * 100
    
    while mixtures_generated < n_mixtures and attempts < max_attempts:
        attempts += 1
        
        # Blood proportion: 60-90%
        blood_prop = np.random.uniform(0.6, 0.9)
        
        # Number of additional tissues: 2-6
        n_other = np.random.randint(2, 7)
        n_other = min(n_other, len(non_blood_tissues))  # Can't exceed available tissues
        
        # Sample non-blood tissues
        selected_other = np.random.choice(non_blood_tissues, size=n_other, replace=False)
        
        # Remaining proportion for other tissues
        remaining_prop = 1.0 - blood_prop
        
        # Use Dirichlet with small alpha for skewed distribution (some tissues minor)
        alpha = np.ones(n_other) * 0.5  # Skewed towards few dominant
        other_proportions = np.random.dirichlet(alpha) * remaining_prop
        
        # Sample augmentation version
        aug_version = np.random.randint(0, 5)
        
        # Get Blood sample
        blood_samples = get_tissue_samples(metadata, 'Blood', aug_version)
        if len(blood_samples) == 0:
            continue
        
        # Get other tissue samples
        tissue_samples = {}
        valid = True
        
        for tissue in selected_other:
            samples = get_tissue_samples(metadata, tissue, aug_version)
            if len(samples) == 0:
                valid = False
                break
            tissue_samples[tissue] = samples
        
        if not valid:
            continue
        
        # Mix: Start with Blood
        blood_idx = np.random.choice(blood_samples.index)
        meth_blood = h5_file['methylation'][blood_idx]
        mixed_region_means = blood_prop * compute_region_means(meth_blood)
        
        selected_indices = {'Blood': blood_idx}
        
        # Add other tissues
        for tissue, prop in zip(selected_other, other_proportions):
            sample_idx = np.random.choice(tissue_samples[tissue].index)
            selected_indices[tissue] = sample_idx
            
            meth = h5_file['methylation'][sample_idx]
            region_means = compute_region_means(meth)
            
            mixed_region_means += prop * region_means
        
        # Create full proportion vector [22]
        proportions = np.zeros(n_tissues, dtype=np.float32)
        proportions[tissue_to_idx['Blood']] = blood_prop
        
        for tissue, prop in zip(selected_other, other_proportions):
            proportions[tissue_to_idx[tissue]] = prop
        
        # Verify proportions sum to 1.0
        assert np.isclose(proportions.sum(), 1.0), f"Proportions sum to {proportions.sum()}"
        
        # Store
        mixed_methylation_list.append(mixed_region_means)
        true_proportions_list.append(proportions)
        mixture_info_list.append({
            'mixture_id': int(mixtures_generated),
            'blood_proportion': float(blood_prop),
            'n_other_tissues': int(n_other),
            'other_tissues': [str(t) for t in selected_other],
            'other_proportions': [float(p) for p in other_proportions],
            'sample_indices': {str(k): int(v) for k, v in selected_indices.items()},
            'aug_version': int(aug_version)
        })
        
        mixtures_generated += 1
        
        if mixtures_generated % 100 == 0:
            print(f"  Generated {mixtures_generated}/{n_mixtures} mixtures...")
    
    if mixtures_generated < n_mixtures:
        print(f"  WARNING: Only generated {mixtures_generated}/{n_mixtures} mixtures after {attempts} attempts")
    
    mixed_methylation = np.array(mixed_methylation_list, dtype=np.float32)
    true_proportions = np.array(true_proportions_list, dtype=np.float32)
    
    print(f"✓ Generated {mixtures_generated} Phase 3 mixtures")
    print(f"  Shape: mixed_methylation={mixed_methylation.shape}, proportions={true_proportions.shape}")
    
    return mixed_methylation, true_proportions, mixture_info_list


def save_mixture_dataset(
    output_path: str,
    mixed_methylation: np.ndarray,
    true_proportions: np.ndarray,
    mixture_info: List[Dict],
    tissue_names: List[str],
    phase: int,
    split: str
):
    """Save mixture dataset to HDF5 file."""
    print(f"\nSaving to: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Save arrays
        f.create_dataset('mixed_methylation', data=mixed_methylation, compression='gzip')
        f.create_dataset('true_proportions', data=true_proportions, compression='gzip')
        
        # Save tissue names as attributes
        f.attrs['tissue_names'] = [t.encode('utf-8') for t in tissue_names]
        f.attrs['n_tissues'] = len(tissue_names)
        f.attrs['n_regions'] = mixed_methylation.shape[1]
        f.attrs['n_mixtures'] = len(mixed_methylation)
        f.attrs['phase'] = phase
        f.attrs['split'] = split
        
        # Save mixture info as JSON strings
        import json
        mixture_info_json = [json.dumps(info) for info in mixture_info]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('mixture_info', data=mixture_info_json, dtype=dt)
    
    print(f"✓ Saved {len(mixed_methylation)} mixtures")
    print(f"  Datasets: mixed_methylation {mixed_methylation.shape}, true_proportions {true_proportions.shape}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic mixture datasets')
    parser.add_argument('--hdf5', type=str, required=True,
                        help='Path to methylation_dataset.h5')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to combined_metadata.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for mixture datasets')
    parser.add_argument('--phases', type=str, default='1,2,3',
                        help='Phases to generate (comma-separated, e.g., "1,2,3")')
    
    args = parser.parse_args()
    
    # Parse phases
    phases = [int(p) for p in args.phases.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MIXTURE DATASET GENERATION")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Phases to generate: {phases}")
    print()
    
    # Load data
    h5_file, metadata = load_data(args.hdf5, args.metadata)
    
    # Get tissue names
    tissue_names = sorted(metadata['tissue_top_level'].unique())
    print(f"\nTissues in dataset ({len(tissue_names)}): {', '.join(tissue_names)}")
    
    # Generate datasets for each phase
    for phase in phases:
        if phase not in PHASE_CONFIGS:
            print(f"\nWARNING: Phase {phase} not recognized, skipping...")
            continue
        
        config = PHASE_CONFIGS[phase]
        print(f"\n{'='*80}")
        print(f"PHASE {phase}: {config['description']}")
        print(f"{'='*80}")
        
        # Generate validation set
        print(f"\n--- Validation Set (n={config['val_size']}) ---")
        if phase == 1:
            val_mixed, val_props, val_info = generate_phase1_mixtures(
                h5_file, metadata, config['val_size'], VAL_SEED)
        elif phase == 2:
            val_mixed, val_props, val_info = generate_phase2_mixtures(
                h5_file, metadata, config['val_size'], VAL_SEED)
        elif phase == 3:
            val_mixed, val_props, val_info = generate_phase3_mixtures(
                h5_file, metadata, config['val_size'], VAL_SEED)
        
        val_path = output_dir / f'phase{phase}_validation_mixtures.h5'
        save_mixture_dataset(val_path, val_mixed, val_props, val_info, 
                           tissue_names, phase, 'validation')
        
        # Generate test set
        print(f"\n--- Test Set (n={config['test_size']}) ---")
        if phase == 1:
            test_mixed, test_props, test_info = generate_phase1_mixtures(
                h5_file, metadata, config['test_size'], TEST_SEED)
        elif phase == 2:
            test_mixed, test_props, test_info = generate_phase2_mixtures(
                h5_file, metadata, config['test_size'], TEST_SEED)
        elif phase == 3:
            test_mixed, test_props, test_info = generate_phase3_mixtures(
                h5_file, metadata, config['test_size'], TEST_SEED)
        
        test_path = output_dir / f'phase{phase}_test_mixtures.h5'
        save_mixture_dataset(test_path, test_mixed, test_props, test_info,
                           tissue_names, phase, 'test')
    
    # Close HDF5
    h5_file.close()
    
    # Final summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in: {output_dir}")
    print("\nFiles created:")
    for phase in phases:
        print(f"  Phase {phase}:")
        print(f"    - phase{phase}_validation_mixtures.h5 ({PHASE_CONFIGS[phase]['val_size']} mixtures)")
        print(f"    - phase{phase}_test_mixtures.h5 ({PHASE_CONFIGS[phase]['test_size']} mixtures)")
    
    total_mixtures = sum(PHASE_CONFIGS[p]['val_size'] + PHASE_CONFIGS[p]['test_size'] 
                        for p in phases)
    print(f"\nTotal mixtures generated: {total_mixtures}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
