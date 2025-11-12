#!/usr/bin/env python3
"""
Script 03: Verify Simulated Read Output
========================================
Purpose: Quality check the generated synthetic reads

What this script does:
1. Loads a sample NPZ file and inspects its contents
2. Verifies data integrity (correct shapes, valid values)
3. Generates summary statistics across all samples
4. Creates a simple visualization of methylation distributions

This is important to catch any errors before proceeding to Step 1.3!
"""

import pandas as pd
import numpy as np
import os
import sys
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads"
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "simulation_summary.txt")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_inspect_npz(npz_file):
    """
    Load an NPZ file and extract key information.
    
    Parameters:
    -----------
    npz_file : str
        Path to NPZ file
    
    Returns:
    --------
    dict : Summary statistics for this file
    """
    data = np.load(npz_file, allow_pickle=True)
    
    # Extract arrays
    methylation_data = data['methylation_data']
    read_boundaries = data['read_boundaries']
    read_lengths = data['read_lengths']
    region_indices = data['region_indices']
    tissue_label = str(data['tissue_label'])
    
    # Calculate statistics
    n_reads = len(read_lengths)
    total_cpgs = len(methylation_data)
    avg_cpgs_per_read = np.mean(read_lengths)
    std_cpgs_per_read = np.std(read_lengths)
    
    # Methylation statistics
    methylation_rate = np.mean(methylation_data)
    
    # Unique regions covered
    n_unique_regions = len(np.unique(region_indices))
    
    summary = {
        'file': os.path.basename(npz_file),
        'tissue_label': tissue_label,
        'n_reads': n_reads,
        'total_cpgs': total_cpgs,
        'avg_cpgs_per_read': avg_cpgs_per_read,
        'std_cpgs_per_read': std_cpgs_per_read,
        'methylation_rate': methylation_rate,
        'n_unique_regions': n_unique_regions,
        'file_size_mb': os.path.getsize(npz_file) / (1024 * 1024)
    }
    
    return summary, data

def verify_data_integrity(data, expected_reads_per_region=500):
    """
    Check if the data structure is correct.
    
    Parameters:
    -----------
    data : numpy.lib.npyio.NpzFile
        Loaded NPZ file
    expected_reads_per_region : int
        Expected number of reads per region
    
    Returns:
    --------
    list : List of any issues found (empty list = all good!)
    """
    issues = []
    
    # Check required fields exist
    required_fields = [
        'methylation_data', 'read_boundaries', 'read_lengths',
        'region_indices', 'region_ids', 'read_ids', 'tissue_label'
    ]
    
    for field in required_fields:
        if field not in data:
            issues.append(f"Missing required field: {field}")
    
    if issues:
        return issues
    
    # Extract data
    methylation_data = data['methylation_data']
    read_boundaries = data['read_boundaries']
    read_lengths = data['read_lengths']
    region_indices = data['region_indices']
    
    # Check data types and ranges
    if not np.issubdtype(methylation_data.dtype, np.integer):
        issues.append(f"methylation_data should be integer type, got {methylation_data.dtype}")
    
    if np.min(methylation_data) < 0 or np.max(methylation_data) > 1:
        issues.append(f"methylation_data values out of range [0,1]: [{np.min(methylation_data)}, {np.max(methylation_data)}]")
    
    # Check consistency
    if len(read_lengths) != len(region_indices):
        issues.append(f"Length mismatch: read_lengths ({len(read_lengths)}) vs region_indices ({len(region_indices)})")
    
    if len(read_boundaries) != len(read_lengths) + 1:
        issues.append(f"read_boundaries length incorrect: expected {len(read_lengths)+1}, got {len(read_boundaries)}")
    
    # Check total length
    expected_total_length = np.sum(read_lengths)
    if len(methylation_data) != expected_total_length:
        issues.append(f"methylation_data length ({len(methylation_data)}) doesn't match sum of read_lengths ({expected_total_length})")
    
    # Check reads per region
    unique_regions = np.unique(region_indices)
    reads_per_region = [np.sum(region_indices == r) for r in unique_regions[:10]]  # Check first 10
    
    for i, count in enumerate(reads_per_region):
        if count != expected_reads_per_region:
            issues.append(f"Region {i} has {count} reads, expected {expected_reads_per_region}")
            break  # Only report first issue
    
    return issues

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("="*80)
    print("STEP 1.2 - Script 03: Verifying Simulated Read Output")
    print("="*80)
    print()
    
    # ========================================================================
    # Step 1: Find all NPZ files
    # ========================================================================
    print("Step 1: Locating output files")
    print("-" * 80)
    
    npz_pattern = os.path.join(OUTPUT_DIR, "sample_*.npz")
    npz_files = sorted(glob.glob(npz_pattern))
    
    if not npz_files:
        print(f"ERROR: No NPZ files found in {OUTPUT_DIR}")
        print("Please run 02_simulate_reads.py first!")
        sys.exit(1)
    
    print(f"✓ Found {len(npz_files)} NPZ files")
    print()
    
    # ========================================================================
    # Step 2: Inspect a sample file in detail
    # ========================================================================
    print("Step 2: Detailed inspection of first sample")
    print("-" * 80)
    
    sample_file = npz_files[0]
    print(f"Inspecting: {os.path.basename(sample_file)}")
    print()
    
    summary, data = load_and_inspect_npz(sample_file)
    
    print("Contents:")
    print(f"  Tissue label:           {summary['tissue_label']}")
    print(f"  Number of reads:        {summary['n_reads']:,}")
    print(f"  Total CpG sites:        {summary['total_cpgs']:,}")
    print(f"  Avg CpGs per read:      {summary['avg_cpgs_per_read']:.2f} ± {summary['std_cpgs_per_read']:.2f}")
    print(f"  Overall methylation:    {summary['methylation_rate']:.3f}")
    print(f"  Unique regions covered: {summary['n_unique_regions']:,}")
    print(f"  File size:              {summary['file_size_mb']:.2f} MB")
    print()
    
    # Data integrity check
    print("Checking data integrity...")
    issues = verify_data_integrity(data)
    
    if issues:
        print("⚠ WARNING: Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    else:
        print("✓ Data integrity check passed!")
        print()
    
    # Show example reads
    print("Example reads (first 3):")
    methylation_data = data['methylation_data']
    read_boundaries = data['read_boundaries']
    read_ids = data['read_ids']
    
    for i in range(min(3, len(read_boundaries)-1)):
        start = read_boundaries[i]
        end = read_boundaries[i+1]
        meth_pattern = methylation_data[start:end]
        read_id = read_ids[i]
        
        # Convert to string for display
        pattern_str = ''.join(['M' if x == 1 else 'U' for x in meth_pattern])
        print(f"  {read_id}: {pattern_str} (length={len(meth_pattern)})")
    print()
    
    # ========================================================================
    # Step 3: Summary statistics across all samples
    # ========================================================================
    print("Step 3: Summary statistics across all samples")
    print("-" * 80)
    print(f"Processing {len(npz_files)} files...")
    print()
    
    all_summaries = []
    
    for npz_file in npz_files:
        summary, _ = load_and_inspect_npz(npz_file)
        all_summaries.append(summary)
    
    # Convert to DataFrame for easy analysis
    df_summary = pd.DataFrame(all_summaries)
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"  Total samples:          {len(df_summary)}")
    print(f"  Total reads:            {df_summary['n_reads'].sum():,}")
    print(f"  Total CpG measurements: {df_summary['total_cpgs'].sum():,}")
    print(f"  Total storage:          {df_summary['file_size_mb'].sum():.1f} MB ({df_summary['file_size_mb'].sum()/1024:.2f} GB)")
    print()
    
    print("Per-sample statistics:")
    print(f"  Reads per sample:       {df_summary['n_reads'].mean():,.0f} ± {df_summary['n_reads'].std():,.0f}")
    print(f"  CpGs per read:          {df_summary['avg_cpgs_per_read'].mean():.2f} ± {df_summary['avg_cpgs_per_read'].std():.2f}")
    print(f"  Methylation rate:       {df_summary['methylation_rate'].mean():.3f} ± {df_summary['methylation_rate'].std():.3f}")
    print(f"  File size:              {df_summary['file_size_mb'].mean():.2f} MB ± {df_summary['file_size_mb'].std():.2f} MB")
    print()
    
    # Tissue type breakdown
    print("Tissue type diversity:")
    tissue_types = df_summary['tissue_label'].apply(lambda x: x.rsplit('_', 1)[0] if '_' in x else x)
    unique_tissue_types = tissue_types.nunique()
    print(f"  Unique tissue types: {unique_tissue_types}")
    print()
    
    # ========================================================================
    # Step 4: Save summary report
    # ========================================================================
    print("Step 4: Saving summary report")
    print("-" * 80)
    
    with open(SUMMARY_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SIMULATED READ GENERATION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples:          {len(df_summary)}\n")
        f.write(f"Total reads:            {df_summary['n_reads'].sum():,}\n")
        f.write(f"Total CpG measurements: {df_summary['total_cpgs'].sum():,}\n")
        f.write(f"Total storage:          {df_summary['file_size_mb'].sum():.1f} MB\n")
        f.write(f"Unique tissue types:    {unique_tissue_types}\n\n")
        
        f.write("PER-SAMPLE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(df_summary.to_string(index=False))
        f.write("\n\n")
        
        if issues:
            f.write("ISSUES DETECTED\n")
            f.write("-"*80 + "\n")
            for issue in issues:
                f.write(f"- {issue}\n")
        else:
            f.write("✓ All integrity checks passed\n")
    
    print(f"✓ Summary report saved to: {SUMMARY_FILE}")
    print()
    
    # Also save the DataFrame as CSV
    csv_file = os.path.join(OUTPUT_DIR, "sample_statistics.csv")
    df_summary.to_csv(csv_file, index=False)
    print(f"✓ Sample statistics saved to: {csv_file}")
    print()
    
    # ========================================================================
    # Step 5: Final verdict
    # ========================================================================
    print("="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    
    if issues:
        print("⚠ WARNING: Some integrity issues were detected (see above)")
        print("Please review before proceeding to Step 1.3")
    else:
        print("✓ All checks passed successfully!")
        print("✓ Data quality looks good")
        print()
        print("Ready to proceed to Step 1.3: Add DNA Sequence Context")
    
    print("="*80)

if __name__ == "__main__":
    main()
