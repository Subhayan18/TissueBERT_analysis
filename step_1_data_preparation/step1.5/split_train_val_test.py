#!/usr/bin/env python3
"""
Step 1.5c - Train/Validation/Test Splitting
============================================

This script performs stratified splitting of samples into train/validation/test sets
while ensuring:
1. Split by SAMPLE (all augmentations stay together)
2. Stratification by tissue type
3. Proper handling of rare tissues
4. Balanced tissue representation across splits

Target split: 70% train / 15% validation / 15% test

Author: Step 1.5c
Date: 2024-11-17
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict


class StratifiedSplitter:
    """
    Performs stratified train/validation/test splitting by sample.
    """
    
    def __init__(self, 
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        Initialize splitter.
        
        Args:
            train_ratio: Proportion for training (default 0.70)
            val_ratio: Proportion for validation (default 0.15)
            test_ratio: Proportion for test (default 0.15)
            random_seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
    
    def split_samples(self, metadata: pd.DataFrame) -> Tuple[List, List, List]:
        """
        Split samples into train/val/test with stratification by tissue.
        
        Args:
            metadata: DataFrame with columns ['sample_name', 'tissue', ...]
        
        Returns:
            (train_samples, val_samples, test_samples)
        """
        # Get unique samples per tissue
        tissue_samples = defaultdict(list)
        
        for sample in metadata['sample_name'].unique():
            tissue = metadata[metadata['sample_name'] == sample]['tissue'].iloc[0]
            tissue_samples[tissue].append(sample)
        
        print("\n" + "="*70)
        print("SAMPLE DISTRIBUTION BY TISSUE")
        print("="*70)
        
        for tissue in sorted(tissue_samples.keys()):
            n_samples = len(tissue_samples[tissue])
            print(f"  {tissue:25s} {n_samples:3d} samples")
        
        # Split each tissue separately
        train_samples = []
        val_samples = []
        test_samples = []
        
        print("\n" + "="*70)
        print("SPLITTING STRATEGY PER TISSUE")
        print("="*70)
        
        for tissue in sorted(tissue_samples.keys()):
            samples = tissue_samples[tissue]
            n_samples = len(samples)
            
            # Shuffle samples
            shuffled = samples.copy()
            np.random.shuffle(shuffled)
            
            # Calculate split sizes
            if n_samples >= 4:
                # Standard stratified split
                n_train = max(1, int(n_samples * self.train_ratio))
                n_val = max(1, int(n_samples * self.val_ratio))
                n_test = max(1, n_samples - n_train - n_val)  # Remainder goes to test
                
            elif n_samples == 3:
                # 2 train, 0 val, 1 test (no validation possible)
                n_train = 2
                n_val = 0
                n_test = 1
                
            elif n_samples == 2:
                # 1 train, 0 val, 1 test
                n_train = 1
                n_val = 0
                n_test = 1
                
            else:  # n_samples == 1
                # All in training (cannot split)
                n_train = 1
                n_val = 0
                n_test = 0
            
            # Assign samples
            tissue_train = shuffled[:n_train]
            tissue_val = shuffled[n_train:n_train + n_val]
            tissue_test = shuffled[n_train + n_val:]
            
            train_samples.extend(tissue_train)
            val_samples.extend(tissue_val)
            test_samples.extend(tissue_test)
            
            # Print split for this tissue
            split_str = f"{n_train}/{n_val}/{n_test}"
            print(f"  {tissue:25s} {n_samples:3d} samples → {split_str:9s} (train/val/test)")
        
        return train_samples, val_samples, test_samples
    
    def get_file_splits(self, 
                       metadata: pd.DataFrame,
                       train_samples: List[str],
                       val_samples: List[str],
                       test_samples: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get file-level splits (including all augmentations).
        
        Args:
            metadata: Full metadata DataFrame
            train_samples: List of training sample names
            val_samples: List of validation sample names
            test_samples: List of test sample names
        
        Returns:
            (train_df, val_df, test_df)
        """
        train_df = metadata[metadata['sample_name'].isin(train_samples)].copy()
        val_df = metadata[metadata['sample_name'].isin(val_samples)].copy()
        test_df = metadata[metadata['sample_name'].isin(test_samples)].copy()
        
        return train_df, val_df, test_df


def print_split_summary(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame):
    """Print detailed summary of the splits."""
    
    print("\n" + "="*70)
    print("SPLIT SUMMARY")
    print("="*70)
    
    # Overall statistics
    total_samples = (train_df['sample_name'].nunique() + 
                    val_df['sample_name'].nunique() + 
                    test_df['sample_name'].nunique())
    total_files = len(train_df) + len(val_df) + len(test_df)
    
    print(f"\nOverall:")
    print(f"  Total unique samples: {total_samples}")
    print(f"  Total files: {total_files}")
    print(f"")
    print(f"  Training:   {train_df['sample_name'].nunique():4d} samples ({len(train_df):5d} files) "
          f"[{100*train_df['sample_name'].nunique()/total_samples:.1f}%]")
    print(f"  Validation: {val_df['sample_name'].nunique():4d} samples ({len(val_df):5d} files) "
          f"[{100*val_df['sample_name'].nunique()/total_samples:.1f}%]")
    print(f"  Test:       {test_df['sample_name'].nunique():4d} samples ({len(test_df):5d} files) "
          f"[{100*test_df['sample_name'].nunique()/total_samples:.1f}%]")
    
    # Per-tissue breakdown
    print("\n" + "="*70)
    print("PER-TISSUE DISTRIBUTION")
    print("="*70)
    print(f"{'Tissue':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-"*70)
    
    all_tissues = sorted(set(train_df['tissue'].unique()) | 
                        set(val_df['tissue'].unique()) | 
                        set(test_df['tissue'].unique()))
    
    for tissue in all_tissues:
        n_train = train_df[train_df['tissue'] == tissue]['sample_name'].nunique()
        n_val = val_df[val_df['tissue'] == tissue]['sample_name'].nunique()
        n_test = test_df[test_df['tissue'] == tissue]['sample_name'].nunique()
        n_total = n_train + n_val + n_test
        
        print(f"{tissue:<25} {n_train:8d} {n_val:8d} {n_test:8d} {n_total:8d}")
    
    # Synthetic vs Original breakdown
    if 'is_synthetic' in train_df.columns:
        print("\n" + "="*70)
        print("SYNTHETIC vs ORIGINAL SAMPLES")
        print("="*70)
        
        train_synth = train_df[train_df['is_synthetic'] == True]['sample_name'].nunique()
        train_orig = train_df[train_df['is_synthetic'] == False]['sample_name'].nunique()
        
        val_synth = val_df[val_df['is_synthetic'] == True]['sample_name'].nunique()
        val_orig = val_df[val_df['is_synthetic'] == False]['sample_name'].nunique()
        
        test_synth = test_df[test_df['is_synthetic'] == True]['sample_name'].nunique()
        test_orig = test_df[test_df['is_synthetic'] == False]['sample_name'].nunique()
        
        print(f"  Training:   {train_orig:3d} original + {train_synth:3d} synthetic = {train_orig+train_synth:3d} total")
        print(f"  Validation: {val_orig:3d} original + {val_synth:3d} synthetic = {val_orig+val_synth:3d} total")
        print(f"  Test:       {test_orig:3d} original + {test_synth:3d} synthetic = {test_orig+test_synth:3d} total")


def verify_split_integrity(metadata: pd.DataFrame,
                          train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          test_df: pd.DataFrame) -> bool:
    """
    Verify that the split is valid.
    
    Checks:
    1. No sample appears in multiple splits
    2. All samples are accounted for
    3. All augmentations of a sample are in the same split
    """
    print("\n" + "="*70)
    print("VERIFYING SPLIT INTEGRITY")
    print("="*70)
    
    train_samples = set(train_df['sample_name'].unique())
    val_samples = set(val_df['sample_name'].unique())
    test_samples = set(test_df['sample_name'].unique())
    
    all_samples = set(metadata['sample_name'].unique())
    split_samples = train_samples | val_samples | test_samples
    
    # Check 1: No overlap
    train_val_overlap = train_samples & val_samples
    train_test_overlap = train_samples & test_samples
    val_test_overlap = val_samples & test_samples
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("  ✗ ERROR: Samples appear in multiple splits!")
        if train_val_overlap:
            print(f"    Train/Val overlap: {train_val_overlap}")
        if train_test_overlap:
            print(f"    Train/Test overlap: {train_test_overlap}")
        if val_test_overlap:
            print(f"    Val/Test overlap: {val_test_overlap}")
        return False
    else:
        print("  ✓ No overlap between splits")
    
    # Check 2: All samples accounted for
    missing_samples = all_samples - split_samples
    if missing_samples:
        print(f"  ✗ ERROR: {len(missing_samples)} samples not in any split!")
        print(f"    Missing: {list(missing_samples)[:5]}...")
        return False
    else:
        print("  ✓ All samples accounted for")
    
    # Check 3: All augmentations together
    augmentation_errors = []
    for sample in all_samples:
        sample_files = metadata[metadata['sample_name'] == sample]
        
        in_train = sample_files['filename'].isin(train_df['filename']).any()
        in_val = sample_files['filename'].isin(val_df['filename']).any()
        in_test = sample_files['filename'].isin(test_df['filename']).any()
        
        splits_present = sum([in_train, in_val, in_test])
        if splits_present > 1:
            augmentation_errors.append(sample)
    
    if augmentation_errors:
        print(f"  ✗ ERROR: {len(augmentation_errors)} samples have augmentations in multiple splits!")
        print(f"    Examples: {augmentation_errors[:5]}")
        return False
    else:
        print("  ✓ All augmentations stay with their sample")
    
    # Check 4: File counts match
    expected_files = len(metadata)
    actual_files = len(train_df) + len(val_df) + len(test_df)
    
    if expected_files != actual_files:
        print(f"  ✗ ERROR: File count mismatch!")
        print(f"    Expected: {expected_files}, Got: {actual_files}")
        return False
    else:
        print(f"  ✓ File counts match ({actual_files} files)")
    
    print("\n  ✓✓✓ ALL CHECKS PASSED ✓✓✓")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Split samples into train/validation/test sets'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to combined_metadata.csv (from Step 1.5b)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save split files'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Training set ratio (default: 0.70)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("\n" + "="*70)
    print("STEP 1.5c: TRAIN/VALIDATION/TEST SPLITTING")
    print("="*70)
    print(f"\nLoading metadata from: {args.metadata}")
    
    metadata = pd.read_csv(args.metadata)
    
    print(f"Total files loaded: {len(metadata)}")
    print(f"Unique samples: {metadata['sample_name'].nunique()}")
    print(f"Unique tissues: {metadata['tissue'].nunique()}")
    
    # Initialize splitter
    splitter = StratifiedSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    # Perform split at sample level
    train_samples, val_samples, test_samples = splitter.split_samples(metadata)
    
    # Get file-level splits (including all augmentations)
    train_df, val_df, test_df = splitter.get_file_splits(
        metadata, train_samples, val_samples, test_samples
    )
    
    # Print summary
    print_split_summary(train_df, val_df, test_df)
    
    # Verify integrity
    is_valid = verify_split_integrity(metadata, train_df, val_df, test_df)
    
    if not is_valid:
        print("\n✗✗✗ SPLIT VALIDATION FAILED ✗✗✗")
        print("Please review the errors above before using these splits.")
        return 1
    
    # Save splits
    print("\n" + "="*70)
    print("SAVING SPLIT FILES")
    print("="*70)
    
    train_path = output_dir / 'train_files.csv'
    val_path = output_dir / 'val_files.csv'
    test_path = output_dir / 'test_files.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  Training files:   {train_path}")
    print(f"  Validation files: {val_path}")
    print(f"  Test files:       {test_path}")
    
    # Save sample lists (useful for reference)
    train_samples_path = output_dir / 'train_samples.txt'
    val_samples_path = output_dir / 'val_samples.txt'
    test_samples_path = output_dir / 'test_samples.txt'
    
    with open(train_samples_path, 'w') as f:
        f.write('\n'.join(sorted(train_samples)))
    
    with open(val_samples_path, 'w') as f:
        f.write('\n'.join(sorted(val_samples)))
    
    with open(test_samples_path, 'w') as f:
        f.write('\n'.join(sorted(test_samples)))
    
    print(f"\n  Sample lists:")
    print(f"    {train_samples_path}")
    print(f"    {val_samples_path}")
    print(f"    {test_samples_path}")
    
    # Save split configuration
    config = {
        'random_seed': args.seed,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'n_train_samples': len(train_samples),
        'n_val_samples': len(val_samples),
        'n_test_samples': len(test_samples),
        'n_train_files': len(train_df),
        'n_val_files': len(val_df),
        'n_test_files': len(test_df),
    }
    
    config_path = output_dir / 'split_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Configuration: {config_path}")
    
    print("\n" + "="*70)
    print("SPLITTING COMPLETE")
    print("="*70)
    print(f"\nAll files saved to: {output_dir}")
    print("\nNext step: Use train_files.csv, val_files.csv, test_files.csv for model training!")
    
    return 0


if __name__ == "__main__":
    exit(main())
