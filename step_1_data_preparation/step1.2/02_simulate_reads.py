#!/usr/bin/env python3
"""
Script 02: Simulate Read-Level Training Data
=============================================
Purpose: Generate realistic 150bp cfDNA reads from block-level beta values

What this script does:
1. Loads the panel_beta_matrix.tsv and tissue labels
2. For each region in each sample:
   - Generates 500 synthetic reads (150bp each)
   - Each read has methylation pattern based on the beta value
   - Adds 5% biological noise
3. Saves reads as compressed NPZ files (one per sample)

Scientific Background:
- Beta value = average methylation across many molecules
- Real cfDNA = individual molecules with binary methylation (0 or 1 per CpG)
- We simulate individual reads by sampling from beta distribution

Output Format (NPZ file per sample):
- region_ids: array of region identifiers
- read_data: sparse matrix of methylation patterns
- metadata: read counts, tissue type, etc.
"""

import pandas as pd
import numpy as np
import sys
import os
from scipy import sparse
import time

# ============================================================================
# CONFIGURATION - Modify these parameters if needed
# ============================================================================
INPUT_FILE = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/panel_beta_matrix.tsv"
TISSUE_LABELS_FILE = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads/tissue_labels.txt"
OUTPUT_DIR = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads"

# Simulation parameters
READS_PER_REGION = 500        # Number of synthetic reads per region
BIOLOGICAL_NOISE = 0.05       # 5% error rate in methylation calling
READ_LENGTH = 150             # Base pairs per read (typical cfDNA fragment)
RANDOM_SEED = 42              # For reproducibility

# Performance settings
CHUNK_SIZE = 1000             # Process regions in chunks to manage memory

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def simulate_read_methylation(beta_value, n_cpgs, noise_rate=0.05):
    """
    Simulate methylation pattern for a single read based on beta value.
    
    The beta value represents the average methylation across many molecules.
    For each CpG in a read, we:
    1. Sample methylation state (0 or 1) based on beta value
    2. Add technical/biological noise
    
    Parameters:
    -----------
    beta_value : float
        Average methylation level (0 to 1)
    n_cpgs : int
        Number of CpG sites in this read
    noise_rate : float
        Probability of methylation calling error
    
    Returns:
    --------
    numpy.ndarray : Binary methylation pattern (0=unmethylated, 1=methylated)
    """
    # Handle missing values
    if np.isnan(beta_value):
        # If no data, return random pattern
        return np.random.binomial(1, 0.5, n_cpgs)
    
    # Clip beta to valid range [0, 1]
    beta_value = np.clip(beta_value, 0.0, 1.0)
    
    # Sample methylation state for each CpG based on beta value
    # This simulates individual molecules from a population
    methylation_pattern = np.random.binomial(1, beta_value, n_cpgs)
    
    # Add biological/technical noise (flip some CpG states)
    if noise_rate > 0:
        noise_mask = np.random.random(n_cpgs) < noise_rate
        methylation_pattern = np.where(noise_mask, 1 - methylation_pattern, methylation_pattern)
    
    return methylation_pattern.astype(np.uint8)

def estimate_cpgs_per_region(region_size=500):
    """
    Estimate number of CpG sites in a genomic region.
    
    The human genome has ~28 million CpGs in ~3 billion bp
    Average density: ~0.01 CpGs per bp
    For a 500bp region: ~5 CpGs on average
    
    We'll use a simplified model here.
    
    Parameters:
    -----------
    region_size : int
        Size of genomic region in base pairs
    
    Returns:
    --------
    int : Estimated number of CpGs
    """
    # Average CpG density in human genome
    avg_cpg_density = 0.01
    
    # Sample from Poisson distribution for realistic variation
    mean_cpgs = region_size * avg_cpg_density
    n_cpgs = np.random.poisson(mean_cpgs)
    
    # Ensure at least 1 CpG per region
    return max(1, n_cpgs)

def process_sample(sample_idx, sample_name, beta_values, region_ids, output_dir):
    """
    Process one sample: generate synthetic reads for all regions.
    
    Parameters:
    -----------
    sample_idx : int
        Sample index (column number)
    sample_name : str
        Simplified tissue name
    beta_values : numpy.ndarray
        Beta values for all regions in this sample
    region_ids : list
        List of region identifiers
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    str : Path to saved NPZ file
    """
    n_regions = len(region_ids)
    
    print(f"  Processing sample {sample_idx + 1}: {sample_name}")
    print(f"    Generating {READS_PER_REGION} reads × {n_regions:,} regions = {READS_PER_REGION * n_regions:,} total reads")
    
    # Storage for this sample's reads
    # We'll use a list to collect read data, then convert to sparse matrix
    all_read_data = []
    all_region_indices = []
    all_read_ids = []
    
    # Process regions in chunks to manage memory
    for chunk_start in range(0, n_regions, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_regions)
        
        if chunk_start % 5000 == 0:
            print(f"    Progress: {chunk_start:,}/{n_regions:,} regions ({100*chunk_start/n_regions:.1f}%)")
        
        # Process each region in this chunk
        for region_idx in range(chunk_start, chunk_end):
            beta = beta_values[region_idx]
            
            # Generate reads for this region
            for read_num in range(READS_PER_REGION):
                # Estimate CpGs in this region (simplified: assume ~5 CpGs per 500bp)
                n_cpgs = estimate_cpgs_per_region(region_size=500)
                
                # Simulate methylation pattern for this read
                meth_pattern = simulate_read_methylation(beta, n_cpgs, BIOLOGICAL_NOISE)
                
                # Store read data
                all_read_data.append(meth_pattern)
                all_region_indices.append(region_idx)
                all_read_ids.append(f"read_{region_idx}_{read_num}")
    
    print(f"    Generated {len(all_read_data):,} reads")
    
    # ========================================================================
    # Save to NPZ file
    # ========================================================================
    output_file = os.path.join(output_dir, f"sample_{sample_idx:03d}_{sample_name}.npz")
    
    print(f"    Saving to: {output_file}")
    
    # Convert read data to efficient storage format
    # Each read may have different number of CpGs, so we store:
    # 1. Concatenated methylation patterns
    # 2. Pointers to where each read starts/ends
    
    read_lengths = [len(read) for read in all_read_data]
    concatenated_meth = np.concatenate(all_read_data)
    read_boundaries = np.cumsum([0] + read_lengths)
    
    # Save as compressed NPZ
    np.savez_compressed(
        output_file,
        # Read methylation data
        methylation_data=concatenated_meth,      # All methylation values concatenated
        read_boundaries=read_boundaries,          # Start/end positions of each read
        read_lengths=np.array(read_lengths),      # Length of each read (n_cpgs)
        
        # Metadata
        region_indices=np.array(all_region_indices),  # Which region each read came from
        region_ids=region_ids,                         # Original region identifiers
        read_ids=all_read_ids,                         # Unique read identifiers
        tissue_label=sample_name,                      # Tissue type for this sample
        
        # Configuration used
        reads_per_region=READS_PER_REGION,
        biological_noise=BIOLOGICAL_NOISE,
        read_length=READ_LENGTH
    )
    
    # Get file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"    ✓ Saved ({file_size_mb:.1f} MB)")
    print()
    
    return output_file

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("="*80)
    print("STEP 1.2 - Script 02: Simulating Read-Level Training Data")
    print("="*80)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED}")
    print()
    
    # ========================================================================
    # Step 1: Load data
    # ========================================================================
    print("Step 1: Loading input data")
    print("-" * 80)
    
    print("Loading panel_beta_matrix.tsv...")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File not found: {INPUT_FILE}")
        sys.exit(1)
    
    df = pd.read_csv(INPUT_FILE, sep='\t', index_col=0)
    print(f"✓ Loaded: {df.shape[0]:,} regions × {df.shape[1]} columns")
    
    # Separate metadata and sample columns
    # First 5 columns are metadata (start, end, startCpG, endCpG, length)
    # Remaining columns are sample methylation data
    if df.shape[1] < 6:
        print(f"ERROR: Expected at least 6 columns, found {df.shape[1]}")
        sys.exit(1)
    
    metadata_cols = df.columns[:5]
    sample_cols = df.columns[5:]
    df_samples = df[sample_cols]
    
    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  Sample columns: {len(sample_cols)}")
    print()
    
    print("Loading tissue labels...")
    if not os.path.exists(TISSUE_LABELS_FILE):
        print(f"ERROR: File not found: {TISSUE_LABELS_FILE}")
        print("Please run 01_inspect_data.py first!")
        sys.exit(1)
    
    with open(TISSUE_LABELS_FILE, 'r') as f:
        tissue_labels = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded: {len(tissue_labels)} tissue labels")
    print()
    
    # Verify dimensions match (tissue labels should match sample columns)
    if len(tissue_labels) != len(sample_cols):
        print(f"ERROR: Mismatch between sample columns ({len(sample_cols)}) and tissue labels ({len(tissue_labels)})")
        sys.exit(1)
    
    # ========================================================================
    # Step 2: Setup
    # ========================================================================
    print("Step 2: Configuration")
    print("-" * 80)
    print(f"Reads per region:    {READS_PER_REGION}")
    print(f"Biological noise:    {BIOLOGICAL_NOISE*100}%")
    print(f"Read length:         {READ_LENGTH} bp")
    print(f"Processing chunks:   {CHUNK_SIZE} regions")
    print()
    
    region_ids = df.index.tolist()
    n_samples = len(sample_cols)
    n_regions = df.shape[0]
    
    print(f"Total reads to generate: {n_samples * n_regions * READS_PER_REGION:,}")
    print()
    
    # ========================================================================
    # Step 3: Process each sample
    # ========================================================================
    print("Step 3: Generating synthetic reads")
    print("-" * 80)
    print(f"Processing {n_samples} samples...")
    print()
    
    start_time = time.time()
    saved_files = []
    
    for sample_idx in range(n_samples):
        sample_name = tissue_labels[sample_idx]
        beta_values = df_samples.iloc[:, sample_idx].values
        
        output_file = process_sample(
            sample_idx, 
            sample_name, 
            beta_values, 
            region_ids, 
            OUTPUT_DIR
        )
        saved_files.append(output_file)
    
    elapsed_time = time.time() - start_time
    
    # ========================================================================
    # Step 4: Summary
    # ========================================================================
    print("="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"✓ Processed {n_samples} samples in {elapsed_time/60:.1f} minutes")
    print(f"✓ Generated {n_samples * n_regions * READS_PER_REGION:,} total reads")
    print(f"✓ Saved {len(saved_files)} NPZ files to: {OUTPUT_DIR}")
    print()
    
    # Calculate total output size
    total_size_mb = sum(os.path.getsize(f) for f in saved_files) / (1024 * 1024)
    print(f"Total output size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    print()
    print("Ready to proceed to Script 03: verify_output.py")
    print("="*80)

if __name__ == "__main__":
    main()
