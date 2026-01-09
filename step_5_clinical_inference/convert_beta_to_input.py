#!/usr/bin/env python3
"""
Convert PDAC Beta Value TSV to Model Input Format (HDF5)
=========================================================

Converts beta value matrix (TSV) to HDF5 format for inference.

Input:  BETA_all.tsv with columns: chr, start, end, startCpG, endCpG, Sample1, Sample2, ...
Output: HDF5 file with methylation array [n_samples, 51089, 150] + metadata CSV

Author: PDAC Clinical Prediction Pipeline
Date: January 2026
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert PDAC beta value TSV to HDF5 format for model input',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_tsv',
        type=str,
        required=True,
        help='Input TSV file with beta values (BETA_all.tsv)'
    )
    
    parser.add_argument(
        '--output_h5',
        type=str,
        required=True,
        help='Output HDF5 file path (e.g., pdac_samples.h5)'
    )
    
    parser.add_argument(
        '--output_metadata',
        type=str,
        default=None,
        help='Output metadata CSV path (default: same as h5 with .csv extension)'
    )
    
    parser.add_argument(
        '--seq_length',
        type=int,
        default=150,
        help='Sequence length per region (default: 150 bp)'
    )
    
    parser.add_argument(
        '--n_regions',
        type=int,
        default=51089,
        help='Expected number of regions (default: 51089)'
    )
    
    parser.add_argument(
        '--beta_to_binary_threshold',
        type=float,
        default=0.5,
        help='Threshold to convert beta values to binary methylation (default: 0.5)'
    )
    
    parser.add_argument(
        '--simulation_method',
        type=str,
        choices=['binary', 'probabilistic', 'replicate'],
        default='probabilistic',
        help='Method to simulate 150bp read-level data from beta values'
    )
    
    parser.add_argument(
        '--n_simulated_reads',
        type=int,
        default=10,
        help='Number of simulated reads per region for probabilistic method (default: 10)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    return parser.parse_args()


def beta_to_methylation_binary(beta_value, seq_length=150, threshold=0.5):
    """
    Convert single beta value to binary methylation pattern (simple method)
    
    Args:
        beta_value: Float between 0-1 (or NaN)
        seq_length: Length of output sequence
        threshold: Threshold for binarization (default: 0.5)
    
    Returns:
        array: [seq_length] with values 0 (unmeth), 1 (meth), or 2 (missing)
    """
    if pd.isna(beta_value):
        return np.full(seq_length, 2, dtype=np.int8)  # All missing
    
    # Convert beta to binary (0 or 1)
    meth_status = 1 if beta_value >= threshold else 0
    
    # Replicate across all positions
    return np.full(seq_length, meth_status, dtype=np.int8)


def beta_to_methylation_probabilistic(beta_value, seq_length=150, n_reads=10, seed=None):
    """
    Convert beta value to methylation pattern using probabilistic sampling
    
    This simulates multiple reads and aggregates them to create a more
    realistic methylation pattern that reflects coverage variability.
    
    Args:
        beta_value: Float between 0-1 (or NaN)
        seq_length: Length of output sequence
        n_reads: Number of simulated reads to aggregate
        seed: Random seed for reproducibility
    
    Returns:
        array: [seq_length] with values 0 (unmeth), 1 (meth), or 2 (missing)
    """
    if pd.isna(beta_value):
        return np.full(seq_length, 2, dtype=np.int8)  # All missing
    
    if seed is not None:
        np.random.seed(seed)
    
    # Simulate n_reads, each with seq_length positions
    # Each position has probability = beta_value of being methylated
    reads = np.random.random((n_reads, seq_length)) < beta_value
    
    # Aggregate: majority vote per position
    # If >=50% of reads are methylated at position i, mark as methylated
    consensus = (reads.sum(axis=0) / n_reads) >= 0.5
    
    return consensus.astype(np.int8)


def beta_to_methylation_replicate(beta_value, seq_length=150):
    """
    Convert beta value to methylation pattern by replicating the proportion
    
    Creates a pattern where approximately beta_value fraction of positions
    are methylated, distributed uniformly.
    
    Args:
        beta_value: Float between 0-1 (or NaN)
        seq_length: Length of output sequence
    
    Returns:
        array: [seq_length] with values 0 (unmeth), 1 (meth), or 2 (missing)
    """
    if pd.isna(beta_value):
        return np.full(seq_length, 2, dtype=np.int8)  # All missing
    
    # Calculate number of methylated positions
    n_meth = int(np.round(beta_value * seq_length))
    
    # Create pattern: first n_meth positions are methylated
    pattern = np.zeros(seq_length, dtype=np.int8)
    pattern[:n_meth] = 1
    
    # Shuffle to distribute methylation uniformly
    np.random.shuffle(pattern)
    
    return pattern


def convert_beta_to_model_input(beta_values, seq_length=150, method='probabilistic', 
                                 threshold=0.5, n_reads=10, verbose=False):
    """
    Convert array of beta values to model input format
    
    Args:
        beta_values: Array of beta values [n_regions]
        seq_length: Sequence length per region
        method: Conversion method ('binary', 'probabilistic', 'replicate')
        threshold: Threshold for binary method
        n_reads: Number of reads for probabilistic method
        verbose: Print progress
    
    Returns:
        methylation_array: [n_regions, seq_length] array with values 0, 1, or 2
    """
    n_regions = len(beta_values)
    methylation_array = np.zeros((n_regions, seq_length), dtype=np.int8)
    
    iterator = tqdm(range(n_regions), desc="Converting regions") if verbose else range(n_regions)
    
    for i in iterator:
        beta = beta_values[i]
        
        if method == 'binary':
            methylation_array[i] = beta_to_methylation_binary(beta, seq_length, threshold)
        elif method == 'probabilistic':
            methylation_array[i] = beta_to_methylation_probabilistic(beta, seq_length, n_reads)
        elif method == 'replicate':
            methylation_array[i] = beta_to_methylation_replicate(beta, seq_length)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return methylation_array


def load_beta_tsv(tsv_path, verbose=False):
    """
    Load beta value TSV file
    
    Args:
        tsv_path: Path to TSV file
        verbose: Print information
    
    Returns:
        df: DataFrame with columns: chr, start, end, startCpG, endCpG, Sample1, Sample2, ...
        sample_names: List of sample column names
        region_info: DataFrame with region coordinates
    """
    if verbose:
        print(f"\nLoading TSV file: {tsv_path}")
    
    # Read TSV
    df = pd.read_csv(tsv_path, sep='\t')
    
    if verbose:
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:10])}...")
    
    # Identify region columns and sample columns
    region_cols = ['chr', 'start', 'end', 'startCpG', 'endCpG']
    sample_cols = [col for col in df.columns if col not in region_cols]
    
    if verbose:
        print(f"  Region columns: {region_cols}")
        print(f"  Number of samples: {len(sample_cols)}")
        print(f"  Sample names: {sample_cols[:5]}...")
    
    # Extract region info
    region_info = df[region_cols].copy()
    
    return df, sample_cols, region_info


def process_and_save_hdf5(df, sample_cols, region_info, output_h5, output_metadata,
                          seq_length, n_regions, method, threshold, n_reads, verbose):
    """
    Process all samples and save as HDF5 file
    
    Args:
        df: DataFrame with beta values
        sample_cols: List of sample column names
        region_info: DataFrame with region coordinates
        output_h5: Output HDF5 file path
        output_metadata: Output metadata CSV path
        seq_length: Sequence length per region
        n_regions: Expected number of regions
        method: Conversion method
        threshold: Binary threshold
        n_reads: Number of reads for probabilistic
        verbose: Print progress
    """
    # Check number of regions
    actual_regions = len(df)
    if actual_regions != n_regions:
        print(f"\n⚠️  WARNING: Expected {n_regions} regions, but TSV has {actual_regions} regions")
        print(f"    Using actual number: {actual_regions}")
        n_regions = actual_regions
    
    n_samples = len(sample_cols)
    
    if verbose:
        print(f"\nProcessing {n_samples} samples...")
        print(f"Output HDF5: {output_h5}")
        print(f"Output metadata: {output_metadata}")
    
    # Create HDF5 file with methylation dataset
    with h5py.File(output_h5, 'w') as h5f:
        # Create dataset: [n_samples, n_regions, seq_length]
        methylation_dset = h5f.create_dataset(
            'methylation',
            shape=(n_samples, n_regions, seq_length),
            dtype='i1',  # int8
            compression='gzip',
            compression_opts=4
        )
        
        if verbose:
            print(f"\nCreated HDF5 dataset:")
            print(f"  Shape: [{n_samples}, {n_regions}, {seq_length}]")
            print(f"  Dtype: int8")
            print(f"  Compression: gzip level 4")
        
        # Process each sample
        sample_info = []
        
        for sample_idx, sample_name in enumerate(tqdm(sample_cols, desc="Processing samples")):
            # Extract beta values for this sample
            beta_values = df[sample_name].values
            
            # Convert to model input format
            methylation_array = convert_beta_to_model_input(
                beta_values,
                seq_length=seq_length,
                method=method,
                threshold=threshold,
                n_reads=n_reads,
                verbose=False  # Don't show per-region progress
            )
            
            # Write to HDF5
            methylation_dset[sample_idx] = methylation_array
            
            # Calculate statistics
            n_missing = (methylation_array == 2).sum()
            n_meth = (methylation_array == 1).sum()
            n_unmeth = (methylation_array == 0).sum()
            missing_pct = (n_missing / methylation_array.size) * 100
            meth_pct = (n_meth / (n_meth + n_unmeth)) * 100 if (n_meth + n_unmeth) > 0 else 0
            
            sample_info.append({
                'sample_idx': sample_idx,
                'sample_name': sample_name,
                'n_regions': n_regions,
                'seq_length': seq_length,
                'n_methylated': int(n_meth),
                'n_unmethylated': int(n_unmeth),
                'n_missing': int(n_missing),
                'methylation_pct': float(meth_pct),
                'missing_pct': float(missing_pct)
            })
    
    # Save metadata CSV
    metadata_df = pd.DataFrame(sample_info)
    metadata_df.to_csv(output_metadata, index=False)
    
    # Save region info
    region_info_path = Path(output_metadata).parent / 'region_info.csv'
    region_info.to_csv(region_info_path, index=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print("CONVERSION COMPLETE")
        print(f"{'='*70}")
        print(f"  Samples processed: {n_samples}")
        print(f"  Output HDF5: {output_h5}")
        print(f"  Output metadata: {output_metadata}")
        print(f"  Region info: {region_info_path}")
        print(f"\nSummary statistics:")
        print(f"  Avg methylation: {metadata_df['methylation_pct'].mean():.2f}%")
        print(f"  Avg missing data: {metadata_df['missing_pct'].mean():.2f}%")
        print(f"\nHDF5 structure:")
        print(f"  Dataset 'methylation': [{n_samples}, {n_regions}, {seq_length}]")
        print(f"  File size: {Path(output_h5).stat().st_size / 1024 / 1024:.2f} MB")
        print(f"{'='*70}")
    
    return metadata_df


def main():
    """Main conversion pipeline"""
    args = parse_args()
    
    # Set default metadata path if not provided
    if args.output_metadata is None:
        args.output_metadata = str(Path(args.output_h5).with_suffix('.csv'))
    
    print("="*70)
    print("PDAC BETA VALUE TO HDF5 CONVERTER")
    print("="*70)
    print(f"Input TSV: {args.input_tsv}")
    print(f"Output HDF5: {args.output_h5}")
    print(f"Output metadata: {args.output_metadata}")
    print(f"Conversion method: {args.simulation_method}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Expected regions: {args.n_regions}")
    print("="*70)
    
    # Load TSV
    df, sample_cols, region_info = load_beta_tsv(args.input_tsv, verbose=args.verbose)
    
    # Process and save as HDF5
    metadata_df = process_and_save_hdf5(
        df=df,
        sample_cols=sample_cols,
        region_info=region_info,
        output_h5=args.output_h5,
        output_metadata=args.output_metadata,
        seq_length=args.seq_length,
        n_regions=args.n_regions,
        method=args.simulation_method,
        threshold=args.beta_to_binary_threshold,
        n_reads=args.n_simulated_reads,
        verbose=args.verbose
    )
    
    print("\n✓ Conversion completed successfully!")
    print(f"\nNext step: Run inference using:")
    print(f"  python run_pdac_inference.py \\")
    print(f"    --input_h5 {args.output_h5} \\")
    print(f"    --input_metadata {args.output_metadata} \\")
    print(f"    --checkpoint <path_to_checkpoint.pt> \\")
    print(f"    --output_dir <output_directory>")


if __name__ == '__main__':
    main()
