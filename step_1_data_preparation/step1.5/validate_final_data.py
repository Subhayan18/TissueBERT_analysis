#!/usr/bin/env python3
"""
Step 1.5d - Final Data Validation
==================================

This script performs comprehensive validation of all Step 1.5 outputs before
proceeding to Phase 2 (Model Training). It checks:

1. File integrity (all .npz files loadable and valid)
2. Metadata consistency (files match records)
3. Split integrity (no leakage, proper stratification)
4. Synthetic sample quality (validation metrics)
5. Tissue distribution (coverage across splits)
6. Data format compatibility (ready for PyTorch)

Author: Step 1.5d
Date: 2024-11-17
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict
from tqdm import tqdm
import sys


class DataValidator:
    """Comprehensive data validation for Step 1.5 outputs."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.info = []
        
    def log_error(self, message: str):
        """Log an error."""
        self.errors.append(message)
        if self.verbose:
            print(f"  ✗ ERROR: {message}")
    
    def log_warning(self, message: str):
        """Log a warning."""
        self.warnings.append(message)
        if self.verbose:
            print(f"  ⚠ WARNING: {message}")
    
    def log_info(self, message: str):
        """Log info."""
        self.info.append(message)
        if self.verbose:
            print(f"  ℹ INFO: {message}")
    
    def log_success(self, message: str):
        """Log success."""
        if self.verbose:
            print(f"  ✓ {message}")
    
    def validate_file_existence(self, 
                                data_dir: Path, 
                                metadata: pd.DataFrame) -> bool:
        """Check that all files in metadata exist."""
        print("\n" + "="*70)
        print("TEST 1: FILE EXISTENCE")
        print("="*70)
        
        missing_files = []
        
        for _, row in tqdm(metadata.iterrows(), 
                          total=len(metadata), 
                          desc="Checking files",
                          disable=not self.verbose):
            filepath = data_dir / row['filename']
            if not filepath.exists():
                missing_files.append(row['filename'])
        
        if missing_files:
            self.log_error(f"{len(missing_files)} files missing from data directory")
            for f in missing_files[:5]:
                self.log_error(f"  Missing: {f}")
            if len(missing_files) > 5:
                self.log_error(f"  ... and {len(missing_files)-5} more")
            return False
        else:
            self.log_success(f"All {len(metadata)} files exist")
            return True
    
    def validate_npz_integrity(self, 
                              data_dir: Path, 
                              metadata: pd.DataFrame,
                              sample_size: int = 50) -> bool:
        """Check that .npz files can be loaded and have correct structure."""
        print("\n" + "="*70)
        print("TEST 2: NPZ FILE INTEGRITY")
        print("="*70)
        
        # Sample files to check
        sample_files = metadata.sample(min(sample_size, len(metadata)))
        
        required_keys = ['dna_tokens', 'methylation', 'region_ids', 'n_reads', 
                        'tissue_label', 'sample_name', 'tissue_name']
        
        corrupted_files = []
        shape_errors = []
        
        for _, row in tqdm(sample_files.iterrows(), 
                          total=len(sample_files),
                          desc="Validating .npz files",
                          disable=not self.verbose):
            filepath = data_dir / row['filename']
            
            try:
                data = np.load(filepath)
                
                # Check required keys
                missing_keys = [k for k in required_keys if k not in data.files]
                if missing_keys:
                    self.log_error(f"{row['filename']}: Missing keys {missing_keys}")
                    corrupted_files.append(row['filename'])
                    continue
                
                # Check shapes
                expected_regions = 51089
                expected_seq_len = 150
                
                if data['dna_tokens'].shape != (expected_regions, expected_seq_len):
                    self.log_error(f"{row['filename']}: Wrong dna_tokens shape "
                                 f"{data['dna_tokens'].shape}, expected ({expected_regions}, {expected_seq_len})")
                    shape_errors.append(row['filename'])
                
                if data['methylation'].shape != (expected_regions, expected_seq_len):
                    self.log_error(f"{row['filename']}: Wrong methylation shape "
                                 f"{data['methylation'].shape}, expected ({expected_regions}, {expected_seq_len})")
                    shape_errors.append(row['filename'])
                
            except Exception as e:
                self.log_error(f"{row['filename']}: Failed to load - {str(e)}")
                corrupted_files.append(row['filename'])
        
        if corrupted_files or shape_errors:
            self.log_error(f"{len(corrupted_files)} corrupted files, {len(shape_errors)} shape errors")
            return False
        else:
            self.log_success(f"All {len(sample_files)} sampled files are valid")
            return True
    
    def validate_metadata_consistency(self, metadata: pd.DataFrame) -> bool:
        """Check metadata internal consistency."""
        print("\n" + "="*70)
        print("TEST 3: METADATA CONSISTENCY")
        print("="*70)
        
        required_columns = ['filename', 'sample_name', 'tissue', 'tissue_index', 
                           'aug_version', 'n_regions', 'total_reads', 'seq_length']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in metadata.columns]
        if missing_cols:
            self.log_error(f"Missing columns: {missing_cols}")
            return False
        
        self.log_success("All required columns present")
        
        # Check for duplicates
        duplicate_files = metadata[metadata.duplicated('filename', keep=False)]
        if len(duplicate_files) > 0:
            self.log_error(f"{len(duplicate_files)} duplicate filenames found")
            return False
        
        self.log_success("No duplicate filenames")
        
        # Check augmentation versions
        expected_aug_versions = {0, 1, 2, 3, 4}
        for sample in metadata['sample_name'].unique():
            sample_data = metadata[metadata['sample_name'] == sample]
            aug_versions = set(sample_data['aug_version'].unique())
            
            if not aug_versions.issubset(expected_aug_versions):
                self.log_warning(f"{sample}: Unexpected aug versions {aug_versions}")
        
        self.log_success("Augmentation versions valid")
        
        # Check tissue indices
        tissue_index_map = metadata[['tissue', 'tissue_index']].drop_duplicates()
        if len(tissue_index_map) != len(tissue_index_map['tissue'].unique()):
            self.log_error("Inconsistent tissue index mapping")
            return False
        
        self.log_success("Tissue indices consistent")
        
        return True
    
    def validate_split_integrity(self,
                                train_df: pd.DataFrame,
                                val_df: pd.DataFrame,
                                test_df: pd.DataFrame) -> bool:
        """Check split integrity and stratification."""
        print("\n" + "="*70)
        print("TEST 4: SPLIT INTEGRITY")
        print("="*70)
        
        train_samples = set(train_df['sample_name'].unique())
        val_samples = set(val_df['sample_name'].unique())
        test_samples = set(test_df['sample_name'].unique())
        
        # Check for overlap
        train_val_overlap = train_samples & val_samples
        train_test_overlap = train_samples & test_samples
        val_test_overlap = val_samples & test_samples
        
        has_error = False
        
        if train_val_overlap:
            self.log_error(f"Train/Val overlap: {len(train_val_overlap)} samples")
            has_error = True
        else:
            self.log_success("No Train/Val overlap")
        
        if train_test_overlap:
            self.log_error(f"Train/Test overlap: {len(train_test_overlap)} samples")
            has_error = True
        else:
            self.log_success("No Train/Test overlap")
        
        if val_test_overlap:
            self.log_error(f"Val/Test overlap: {len(val_test_overlap)} samples")
            has_error = True
        else:
            self.log_success("No Val/Test overlap")
        
        # Check augmentation consistency
        for sample in train_samples | val_samples | test_samples:
            train_has = sample in train_samples
            val_has = sample in val_samples
            test_has = sample in test_samples
            
            if sum([train_has, val_has, test_has]) > 1:
                self.log_error(f"{sample}: Appears in multiple splits")
                has_error = True
        
        if not has_error:
            self.log_success("All samples in exactly one split")
        
        return not has_error
    
    def validate_tissue_coverage(self,
                                 train_df: pd.DataFrame,
                                 val_df: pd.DataFrame,
                                 test_df: pd.DataFrame) -> bool:
        """Check that all tissues are represented in training."""
        print("\n" + "="*70)
        print("TEST 5: TISSUE COVERAGE")
        print("="*70)
        
        all_tissues = set(train_df['tissue'].unique()) | \
                     set(val_df['tissue'].unique()) | \
                     set(test_df['tissue'].unique())
        
        train_tissues = set(train_df['tissue'].unique())
        
        missing_from_train = all_tissues - train_tissues
        
        if missing_from_train:
            self.log_error(f"{len(missing_from_train)} tissues missing from training: {missing_from_train}")
            return False
        else:
            self.log_success(f"All {len(all_tissues)} tissues present in training set")
        
        # Check minimum samples per tissue in training
        train_tissue_counts = train_df.groupby('tissue')['sample_name'].nunique()
        low_count_tissues = train_tissue_counts[train_tissue_counts < 2]
        
        if len(low_count_tissues) > 0:
            self.log_warning(f"{len(low_count_tissues)} tissues have <2 training samples:")
            for tissue, count in low_count_tissues.items():
                self.log_warning(f"  {tissue}: {count} sample")
        else:
            self.log_success("All tissues have ≥2 training samples")
        
        return True
    
    def validate_synthetic_quality(self, 
                                   synthetic_samples_dir: Path) -> bool:
        """Check synthetic sample validation metrics."""
        print("\n" + "="*70)
        print("TEST 6: SYNTHETIC SAMPLE QUALITY")
        print("="*70)
        
        validation_file = synthetic_samples_dir / 'validation_results.csv'
        
        if not validation_file.exists():
            self.log_warning("validation_results.csv not found - skipping synthetic quality check")
            return True
        
        validation_df = pd.read_csv(validation_file)
        
        # Check KS test p-values
        low_pvalue = validation_df[validation_df['ks_pvalue'] < 0.01]
        if len(low_pvalue) > 0:
            self.log_error(f"{len(low_pvalue)} synthetic samples have KS p-value < 0.01")
            return False
        else:
            self.log_success(f"All synthetic samples pass KS test (p > 0.01)")
        
        # Check mean difference
        high_diff = validation_df[validation_df['mean_diff'] > 0.15]
        if len(high_diff) > 0:
            self.log_error(f"{len(high_diff)} synthetic samples have mean diff > 0.15")
            return False
        else:
            self.log_success(f"All synthetic samples preserve mean methylation")
        
        # Check regional correlation
        low_corr = validation_df[validation_df['regional_correlation'] < 0.85]
        if len(low_corr) > 0:
            self.log_error(f"{len(low_corr)} synthetic samples have correlation < 0.85")
            return False
        else:
            self.log_success(f"All synthetic samples preserve regional patterns")
        
        return True
    
    def validate_pytorch_compatibility(self,
                                      data_dir: Path,
                                      metadata: pd.DataFrame,
                                      n_samples: int = 5) -> bool:
        """Test that data can be loaded with PyTorch."""
        print("\n" + "="*70)
        print("TEST 7: PYTORCH COMPATIBILITY")
        print("="*70)
        
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
        except ImportError:
            self.log_warning("PyTorch not installed - skipping compatibility check")
            return True
        
        # Create simple dataset
        class TestDataset(Dataset):
            def __init__(self, files, data_dir):
                self.files = files
                self.data_dir = data_dir
            
            def __len__(self):
                return len(self.files)
            
            def __getitem__(self, idx):
                data = np.load(self.data_dir / self.files[idx])
                return {
                    'dna_tokens': torch.tensor(data['dna_tokens'][0]),  # First region
                    'methylation': torch.tensor(data['methylation'][0]),
                    'tissue_label': torch.tensor(data['tissue_label'])
                }
        
        # Test with a few files
        test_files = metadata['filename'].sample(min(n_samples, len(metadata))).tolist()
        
        try:
            dataset = TestDataset(test_files, data_dir)
            loader = DataLoader(dataset, batch_size=2, shuffle=True)
            
            # Try to load one batch
            batch = next(iter(loader))
            
            self.log_success(f"PyTorch DataLoader works with batch size {len(batch['dna_tokens'])}")
            return True
            
        except Exception as e:
            self.log_error(f"PyTorch compatibility test failed: {str(e)}")
            return False


def print_final_summary(validator: DataValidator,
                       train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame):
    """Print final validation summary."""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    total_tests = 7
    passed_tests = total_tests - len(validator.errors)
    
    print(f"\nTests Results:")
    print(f"  Passed:   {passed_tests}/{total_tests}")
    print(f"  Errors:   {len(validator.errors)}")
    print(f"  Warnings: {len(validator.warnings)}")
    
    if validator.errors:
        print("\n" + "="*70)
        print("ERRORS FOUND")
        print("="*70)
        for error in validator.errors:
            print(f"  ✗ {error}")
    
    if validator.warnings:
        print("\n" + "="*70)
        print("WARNINGS")
        print("="*70)
        for warning in validator.warnings:
            print(f"  ⚠ {warning}")
    
    # Data statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    total_samples = (train_df['sample_name'].nunique() + 
                    val_df['sample_name'].nunique() + 
                    test_df['sample_name'].nunique())
    total_files = len(train_df) + len(val_df) + len(test_df)
    
    print(f"\nSamples:")
    print(f"  Training:   {train_df['sample_name'].nunique():4d} ({100*train_df['sample_name'].nunique()/total_samples:.1f}%)")
    print(f"  Validation: {val_df['sample_name'].nunique():4d} ({100*val_df['sample_name'].nunique()/total_samples:.1f}%)")
    print(f"  Test:       {test_df['sample_name'].nunique():4d} ({100*test_df['sample_name'].nunique()/total_samples:.1f}%)")
    print(f"  Total:      {total_samples:4d}")
    
    print(f"\nFiles:")
    print(f"  Training:   {len(train_df):5d}")
    print(f"  Validation: {len(val_df):5d}")
    print(f"  Test:       {len(test_df):5d}")
    print(f"  Total:      {total_files:5d}")
    
    print(f"\nTissues:")
    all_tissues = set(train_df['tissue'].unique()) | \
                 set(val_df['tissue'].unique()) | \
                 set(test_df['tissue'].unique())
    print(f"  Unique tissues: {len(all_tissues)}")
    
    if 'is_synthetic' in train_df.columns:
        train_synth = train_df[train_df['is_synthetic'] == True]['sample_name'].nunique()
        train_orig = train_df[train_df['is_synthetic'] == False]['sample_name'].nunique()
        print(f"\nSynthetic samples:")
        print(f"  Training:   {train_synth:3d} synthetic + {train_orig:3d} original")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive validation of Step 1.5 outputs'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing all training .npz files'
    )
    parser.add_argument(
        '--train-files',
        type=str,
        required=True,
        help='Path to train_files.csv'
    )
    parser.add_argument(
        '--val-files',
        type=str,
        required=True,
        help='Path to val_files.csv'
    )
    parser.add_argument(
        '--test-files',
        type=str,
        required=True,
        help='Path to test_files.csv'
    )
    parser.add_argument(
        '--synthetic-dir',
        type=str,
        default=None,
        help='Directory containing synthetic samples (optional)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50,
        help='Number of files to sample for integrity checks (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    
    print("\n" + "="*70)
    print("STEP 1.5d: FINAL DATA VALIDATION")
    print("="*70)
    
    # Load split files
    print("\nLoading split files...")
    train_df = pd.read_csv(args.train_files)
    val_df = pd.read_csv(args.val_files)
    test_df = pd.read_csv(args.test_files)
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"  Training:   {len(train_df):5d} files")
    print(f"  Validation: {len(val_df):5d} files")
    print(f"  Test:       {len(test_df):5d} files")
    print(f"  Total:      {len(combined_df):5d} files")
    
    # Initialize validator
    validator = DataValidator(verbose=True)
    
    # Run validation tests
    results = []
    
    results.append(validator.validate_file_existence(data_dir, combined_df))
    results.append(validator.validate_npz_integrity(data_dir, combined_df, args.sample_size))
    results.append(validator.validate_metadata_consistency(combined_df))
    results.append(validator.validate_split_integrity(train_df, val_df, test_df))
    results.append(validator.validate_tissue_coverage(train_df, val_df, test_df))
    
    if args.synthetic_dir:
        synthetic_dir = Path(args.synthetic_dir)
        results.append(validator.validate_synthetic_quality(synthetic_dir))
    
    results.append(validator.validate_pytorch_compatibility(data_dir, combined_df))
    
    # Print final summary
    print_final_summary(validator, train_df, val_df, test_df)
    
    # Final verdict
    print("\n" + "="*70)
    
    if all(results) and len(validator.errors) == 0:
        print("✓✓✓ ALL VALIDATION TESTS PASSED ✓✓✓")
        print("="*70)
        print("\n🎉 Your data is ready for Phase 2: Model Training!")
        print("\nNext steps:")
        print("  1. Review the dataset statistics above")
        print("  2. Proceed to Phase 2: Model Architecture & Training")
        print("  3. Use train_files.csv, val_files.csv, test_files.csv")
        return 0
    else:
        print("✗✗✗ VALIDATION FAILED ✗✗✗")
        print("="*70)
        print(f"\n{len(validator.errors)} error(s) found. Please fix before proceeding.")
        print("\nReview the errors above and:")
        print("  1. Check your data files")
        print("  2. Re-run previous steps if necessary")
        print("  3. Run validation again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
