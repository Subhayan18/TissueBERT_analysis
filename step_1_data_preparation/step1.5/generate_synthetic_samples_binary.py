#!/usr/bin/env python3
"""
Step 1.5b - Generate Synthetic Samples (BINARY METHYLATION - PRODUCTION)
=========================================================================

Production version for binary methylation data using flip-based noise.
Supports SLURM array jobs for parallel processing.

Features:
- Flip-based noise for binary (0/1) methylation data
- Processes all augmentation versions (aug0-aug4)
- Adaptive flip rate (reduces if validation fails)
- SLURM array job support for parallel processing
- Comprehensive logging and validation

Author: Step 1.5b (Binary Methylation)
Date: 2024-11-17
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy import stats
from collections import defaultdict
import json
import sys


# Tissue hierarchy for reference
TISSUE_HIERARCHY = {
    'Brain': ['Cortex', 'Cerebellum', 'Neuron', 'Oligodendrocytes'],
    'Digestive': ['Colon', 'Gastric', 'Pancreas', 'Liver', 'Small'],
    'Cardiovascular': ['Heart', 'Aorta', 'Coronary'],
    'Connective': ['Dermal', 'Bone', 'Skeletal'],
    'Respiratory': ['Lung'],
    'Blood': ['Blood'],
    'Urogenital': ['Kidney', 'Bladder', 'Prostate'],
    'Skin': ['Epidermal', 'Dermal'],
    'Adipose': ['Adipocytes'],
}


def add_flip_noise(original_meth, flip_rate=0.10, correlation_length=5):
    """
    Add noise to BINARY methylation by flipping values with spatial correlation.
    
    Args:
        original_meth: Binary array (0, 1, or 2 for non-CpG)
        flip_rate: Probability of flipping each CpG value
        correlation_length: Nearby CpGs more likely to flip together
    
    Returns:
        Modified methylation array with some values flipped
    """
    new_meth = original_meth.copy().astype(float)
    cpg_mask = (original_meth < 2)
    
    n_cpg = cpg_mask.sum()
    
    if n_cpg == 0:
        return new_meth
    
    cpg_indices = np.where(cpg_mask)[0]
    
    if len(cpg_indices) < 2:
        # Only one CpG, simple random flip
        if np.random.random() < flip_rate:
            new_meth[cpg_indices[0]] = 1.0 - new_meth[cpg_indices[0]]
        return new_meth
    
    # Generate base flip probabilities
    base_flip_probs = np.random.random(len(cpg_indices))
    
    # Add spatial correlation
    window = min(correlation_length, len(cpg_indices))
    correlated_probs = np.convolve(base_flip_probs, 
                                    np.ones(window)/window, 
                                    mode='same')
    
    # Flip where probability exceeds threshold
    flip_threshold = 1.0 - flip_rate
    flip_cpgs = correlated_probs > flip_threshold
    
    # Apply flips: 0→1, 1→0
    for i, should_flip in enumerate(flip_cpgs):
        if should_flip:
            cpg_pos = cpg_indices[i]
            new_meth[cpg_pos] = 1.0 - new_meth[cpg_pos]
    
    # Non-CpG positions stay as 2
    new_meth[~cpg_mask] = 2
    
    return new_meth


def validate_synthetic(original_file: Path, synthetic_data: dict):
    """
    Validate synthetic sample against original.
    
    Returns:
        (is_valid, metrics_dict)
    """
    original = np.load(original_file)
    
    # Extract CpG methylation values
    original_meth = original['methylation'][original['methylation'] < 2]
    synthetic_meth = synthetic_data['methylation'][synthetic_data['methylation'] < 2]
    
    metrics = {}
    
    # Test 1: KS test
    ks_stat, ks_pvalue = stats.ks_2samp(original_meth, synthetic_meth)
    metrics['ks_statistic'] = ks_stat
    metrics['ks_pvalue'] = ks_pvalue
    
    # Test 2: Mean methylation
    original_mean = original_meth.mean()
    synthetic_mean = synthetic_meth.mean()
    metrics['original_mean'] = original_mean
    metrics['synthetic_mean'] = synthetic_mean
    metrics['mean_diff'] = abs(original_mean - synthetic_mean)
    
    # Test 3: Regional correlation
    original_regional_means = original['methylation'].mean(axis=1)
    synthetic_regional_means = synthetic_data['methylation'].mean(axis=1)
    
    valid_regions = (original_regional_means < 2) & (synthetic_regional_means < 2)
    
    if valid_regions.sum() > 10:
        correlation = np.corrcoef(
            original_regional_means[valid_regions],
            synthetic_regional_means[valid_regions]
        )[0, 1]
        metrics['regional_correlation'] = correlation
    else:
        metrics['regional_correlation'] = np.nan
    
    # Test 4: Hamming distance
    hamming_dist = np.mean(original_meth != synthetic_meth)
    metrics['hamming_distance'] = hamming_dist
    
    # Overall validation
    is_valid = (
        ks_pvalue > 0.01 and
        metrics['mean_diff'] < 0.15 and
        (np.isnan(metrics['regional_correlation']) or 
         metrics['regional_correlation'] > 0.85)
    )
    
    return is_valid, metrics


def generate_synthetic_sample_with_retry(original_file: Path,
                                         sample_name: str,
                                         tissue: str,
                                         synthetic_id: int,
                                         initial_flip_rate: float = 0.10,
                                         correlation_length: int = 5):
    """
    Generate synthetic sample with adaptive flip rate if validation fails.
    
    Tries initial flip_rate, then reduces by 1% until validation passes
    or minimum rate (5%) reached.
    """
    flip_rate = initial_flip_rate
    min_flip_rate = 0.05
    
    original = np.load(original_file)
    
    while flip_rate >= min_flip_rate:
        # Create synthetic data
        synthetic_data = {
            'dna_tokens': original['dna_tokens'].copy(),
            'methylation': original['methylation'].copy(),
            'region_ids': original['region_ids'].copy(),
            'n_reads': original['n_reads'].copy(),
            'tissue_label': original['tissue_label'].copy(),
            'sample_name': f"{sample_name}_synthetic_{synthetic_id}",
            'tissue_name': original['tissue_name'],
        }
        
        # Add flip-based noise
        n_regions = synthetic_data['methylation'].shape[0]
        
        for region_idx in range(n_regions):
            original_meth = original['methylation'][region_idx]
            new_meth = add_flip_noise(original_meth, flip_rate, correlation_length)
            synthetic_data['methylation'][region_idx] = new_meth
        
        # Validate
        is_valid, metrics = validate_synthetic(original_file, synthetic_data)
        
        if is_valid:
            metrics['flip_rate_used'] = flip_rate
            return synthetic_data, True, metrics
        
        # Reduce flip rate and retry
        flip_rate -= 0.01
    
    # All attempts failed, return last attempt
    metrics['flip_rate_used'] = flip_rate + 0.01  # Last attempted rate
    return synthetic_data, False, metrics


def identify_rare_tissues(metadata: pd.DataFrame, min_samples: int = 4):
    """Identify tissues with fewer than min_samples."""
    tissue_counts = metadata.groupby('tissue')['sample_name'].nunique()
    rare_tissues = tissue_counts[tissue_counts < min_samples].index.tolist()
    return rare_tissues


def calculate_synthetic_needs(metadata: pd.DataFrame, 
                              target_per_split: dict = {'train': 2, 'val': 1, 'test': 1}):
    """Calculate how many synthetic samples needed per rare tissue."""
    tissue_counts = metadata.groupby('tissue')['sample_name'].nunique()
    total_needed = sum(target_per_split.values())
    
    synthetic_needs = {}
    for tissue, current_count in tissue_counts.items():
        if current_count < total_needed:
            synthetic_needs[tissue] = total_needed - current_count
    
    return synthetic_needs


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic samples for rare tissues (binary methylation)'
    )
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to updated_metadata.csv')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training .npz files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save synthetic samples')
    parser.add_argument('--min-samples', type=int, default=4,
                       help='Minimum samples per tissue (default: 4)')
    parser.add_argument('--flip-rate', type=float, default=0.10,
                       help='Initial flip rate for noise generation (default: 0.10)')
    parser.add_argument('--correlation-length', type=int, default=5,
                       help='Spatial correlation length (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--tissue-filter', type=str, default=None,
                       help='Process only this tissue (for SLURM array jobs)')
    parser.add_argument('--slurm-array-id', type=int, default=None,
                       help='SLURM array task ID (auto-assigns tissues)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SYNTHETIC SAMPLE GENERATION (BINARY METHYLATION)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Metadata: {args.metadata}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Min samples: {args.min_samples}")
    print(f"  Initial flip rate: {args.flip_rate}")
    print(f"  Correlation length: {args.correlation_length}")
    print(f"  Random seed: {args.seed}")
    
    # Load metadata
    metadata = pd.read_csv(args.metadata)
    
    # Identify rare tissues
    rare_tissues = identify_rare_tissues(metadata, args.min_samples)
    synthetic_needs = calculate_synthetic_needs(metadata)
    
    print(f"\n" + "="*70)
    print("RARE TISSUES IDENTIFIED")
    print("="*70)
    
    tissue_counts = metadata.groupby('tissue')['sample_name'].nunique()
    for tissue in sorted(rare_tissues):
        count = tissue_counts[tissue]
        needed = synthetic_needs.get(tissue, 0)
        print(f"  {tissue:25s} {count} samples → needs {needed} synthetic")
    
    # Filter tissues if specified (for SLURM parallelization)
    if args.slurm_array_id is not None:
        # Map SLURM array ID to tissue
        rare_tissues_sorted = sorted(rare_tissues)
        if args.slurm_array_id >= len(rare_tissues_sorted):
            print(f"\n✗ ERROR: SLURM array ID {args.slurm_array_id} out of range (max: {len(rare_tissues_sorted)-1})")
            return 1
        tissues_to_process = [rare_tissues_sorted[args.slurm_array_id]]
        print(f"\n📋 SLURM Array Job: Processing tissue {args.slurm_array_id} = {tissues_to_process[0]}")
    elif args.tissue_filter:
        tissues_to_process = [args.tissue_filter]
        print(f"\n📋 Processing filtered tissue: {args.tissue_filter}")
    else:
        tissues_to_process = rare_tissues
        print(f"\n📋 Processing all {len(rare_tissues)} rare tissues")
    
    # Generate synthetic samples
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC SAMPLES")
    print("="*70)
    
    synthetic_metadata = []
    validation_results = []
    
    for tissue in tissues_to_process:
        if tissue not in synthetic_needs:
            continue
        
        n_needed = synthetic_needs[tissue]
        
        print(f"\n→ Processing tissue: {tissue} (needs {n_needed} synthetic samples)")
        
        # Get original samples for this tissue (all augmentations)
        original_samples = metadata[metadata['tissue'] == tissue]['sample_name'].unique()
        
        samples_generated = 0
        
        for base_sample in original_samples:
            if samples_generated >= n_needed:
                break
            
            # How many synthetic samples to generate from this original
            n_to_generate = min(
                n_needed - samples_generated,
                (n_needed + len(original_samples) - 1) // len(original_samples)
            )
            
            for synthetic_id in range(1, n_to_generate + 1):
                print(f"  Generating synthetic_{samples_generated + synthetic_id} from {base_sample}...")
                
                # Get all augmentation files for this base sample
                aug_files = metadata[metadata['sample_name'] == base_sample].sort_values('aug_version')
                
                synthetic_success = True
                
                # Generate synthetic version for each augmentation
                for _, row in aug_files.iterrows():
                    original_file = data_dir / row['filename']
                    
                    if not original_file.exists():
                        print(f"    ⚠ Warning: {original_file} not found, skipping")
                        continue
                    
                    aug_version = row['aug_version']
                    
                    # Generate with adaptive retry
                    synthetic_data, is_valid, metrics = generate_synthetic_sample_with_retry(
                        original_file,
                        base_sample,
                        tissue,
                        samples_generated + synthetic_id,
                        initial_flip_rate=args.flip_rate,
                        correlation_length=args.correlation_length
                    )
                    
                    if not is_valid:
                        print(f"    ⚠ Warning: aug{aug_version} failed validation (kept anyway)")
                        synthetic_success = False
                    
                    # Save synthetic sample
                    synthetic_filename = f"{synthetic_data['sample_name']}_aug{aug_version}.npz"
                    output_path = output_dir / synthetic_filename
                    
                    np.savez_compressed(output_path, **synthetic_data)
                    
                    # Add to metadata
                    synthetic_metadata.append({
                        'filename': synthetic_filename,
                        'sample_name': synthetic_data['sample_name'],
                        'tissue': tissue,
                        'tissue_index': row['tissue_index'],
                        'aug_version': aug_version,
                        'n_regions': 51089,
                        'total_reads': row['total_reads'],
                        'seq_length': 150,
                        'is_synthetic': True,
                        'original_sample': base_sample,
                    })
                    
                    # Add validation results
                    validation_results.append({
                        'synthetic_sample': synthetic_data['sample_name'],
                        'aug_version': aug_version,
                        'original_sample': base_sample,
                        'tissue': tissue,
                        'validation_passed': is_valid,
                        **metrics
                    })
                
                status = "✓" if synthetic_success else "⚠"
                print(f"    {status} Synthetic_{samples_generated + synthetic_id} completed ({len(aug_files)} augmentations)")
                
                samples_generated += 1
    
    # Save metadata
    print("\n" + "="*70)
    print("SAVING METADATA")
    print("="*70)
    
    # Save synthetic metadata
    synthetic_meta_df = pd.DataFrame(synthetic_metadata)
    synthetic_meta_path = output_dir / 'synthetic_metadata.csv'
    
    if synthetic_meta_path.exists():
        # Append to existing (for SLURM array jobs)
        existing = pd.read_csv(synthetic_meta_path)
        synthetic_meta_df = pd.concat([existing, synthetic_meta_df], ignore_index=True)
    
    synthetic_meta_df.to_csv(synthetic_meta_path, index=False)
    
    # Save validation results
    validation_df = pd.DataFrame(validation_results)
    validation_path = output_dir / 'validation_results.csv'
    
    if validation_path.exists():
        # Append to existing
        existing = pd.read_csv(validation_path)
        validation_df = pd.concat([existing, validation_df], ignore_index=True)
    
    validation_df.to_csv(validation_path, index=False)
    
    # Create/update combined metadata
    original_metadata = metadata.copy()
    original_metadata['is_synthetic'] = False
    original_metadata['original_sample'] = original_metadata['sample_name']
    
    # Load all synthetic metadata (in case of multiple SLURM jobs)
    all_synthetic = pd.read_csv(synthetic_meta_path)
    
    combined_metadata = pd.concat([original_metadata, all_synthetic], ignore_index=True)
    combined_meta_path = output_dir / 'combined_metadata.csv'
    combined_metadata.to_csv(combined_meta_path, index=False)
    
    print(f"  Synthetic metadata: {synthetic_meta_path}")
    print(f"  Validation results: {validation_path}")
    print(f"  Combined metadata: {combined_meta_path}")
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nSynthetic samples generated: {len(synthetic_metadata)}")
    
    if len(validation_results) > 0:
        n_passed = sum(v['validation_passed'] for v in validation_results)
        print(f"Validation pass rate: {n_passed}/{len(validation_results)} ({100*n_passed/len(validation_results):.1f}%)")
    
    print(f"\nFiles saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
