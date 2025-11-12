#!/usr/bin/env python3
"""
Script 01: Inspect Panel Beta Matrix
=====================================
Purpose: Load and validate the panel_beta_matrix.tsv file from Step 1.1
         This is a quick sanity check before running the main simulation.

What this script does:
1. Loads the TSV file
2. Extracts and simplifies tissue labels from column headers
3. Prints basic statistics (number of regions, samples, data range)
4. Saves the simplified tissue labels for the next script

Expected Input:
- panel_beta_matrix.tsv with columns like:
  GSM5652224_Neuron-Z000000TH.hg38, GSM5652225_Cortex-Neuron-Z0000042F.hg38, ...

Expected Output:
- Console output showing data dimensions and statistics
- tissue_labels.txt: simplified tissue names (one per line)
"""

import pandas as pd
import numpy as np
import sys
import os

# ============================================================================
# CONFIGURATION - Modify these paths if needed
# ============================================================================
INPUT_FILE = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/panel_beta_matrix.tsv"
OUTPUT_DIR = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads"
TISSUE_LABELS_FILE = os.path.join(OUTPUT_DIR, "tissue_labels.txt")

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def simplify_tissue_name(column_name):
    """
    Simplify tissue names from Loyfer atlas format.
    
    Example transformations:
    GSM5652224_Neuron-Z000000TH.hg38 -> Neuron
    GSM5652225_Cortex-Neuron-Z0000042F.hg38 -> Cortex-Neuron_1
    GSM5652226_Cortex-Neuron-Z0000042H.hg38 -> Cortex-Neuron_2
    
    Parameters:
    -----------
    column_name : str
        Original column name from the TSV file
    
    Returns:
    --------
    str : Simplified tissue name
    """
    # Remove GSM identifier and sample ID
    # Split by underscore and take the middle part
    parts = column_name.split('_')
    if len(parts) >= 2:
        # Get the tissue part (between GSM and sample ID)
        tissue_part = parts[1].split('-Z')[0]  # Remove sample ID after -Z
        tissue_part = tissue_part.replace('.hg38', '')  # Remove genome version
        return tissue_part
    return column_name

def main():
    print("="*80)
    print("STEP 1.2 - Script 01: Inspecting Panel Beta Matrix")
    print("="*80)
    print()
    
    # Create output directory if it doesn't exist
    print(f"Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("✓ Output directory ready")
    print()
    
    # ========================================================================
    # Step 1: Load the data
    # ========================================================================
    print("Step 1: Loading panel_beta_matrix.tsv...")
    print(f"Input file: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File not found: {INPUT_FILE}")
        sys.exit(1)
    
    # First, peek at the file structure
    print("Inspecting file structure...")
    with open(INPUT_FILE, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
        print(f"  First line (header): {first_line[:100]}...")
        print(f"  Second line (data):  {second_line[:100]}...")
    print()
    
    try:
        # Load the TSV file
        # First column is region IDs, rest are sample methylation values
        # Use low_memory=False to ensure consistent dtype inference
        df = pd.read_csv(INPUT_FILE, sep='\t', index_col=0, low_memory=False)
        print(f"✓ Successfully loaded data")
        print()
        
        # Convert all columns to numeric, coercing errors to NaN
        print("Converting data to numeric format...")
        original_shape = df.shape
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Check if conversion created new NaN values (indicates non-numeric data)
        new_nans = df.isnull().sum().sum()
        if new_nans > 0:
            print(f"⚠ Warning: {new_nans:,} non-numeric values converted to NaN")
        
        print(f"✓ Data conversion complete")
        print()
        
        # Show a preview of the data
        print("Data preview (first 3 regions, first 3 samples):")
        print(df.iloc[:3, :3])
        print()
    except Exception as e:
        print(f"ERROR loading file: {e}")
        sys.exit(1)
    
    # ========================================================================
    # Step 2: Basic data statistics
    # ========================================================================
    print("Step 2: Data Statistics")
    print("-" * 80)
    print(f"Number of regions (rows):    {df.shape[0]:,}")
    print(f"Number of samples (columns): {df.shape[1]:,}")
    print(f"Total data points:           {df.shape[0] * df.shape[1]:,}")
    print()
    
    # Identify metadata vs data columns
    # Columns 1-5 are metadata (start, end, startCpG, endCpG, length)
    # Columns 6+ are methylation data
    if df.shape[1] < 6:
        print("ERROR: Expected at least 6 columns (5 metadata + samples)")
        sys.exit(1)
    
    metadata_cols = df.columns[:5]
    data_cols = df.columns[5:]
    
    print(f"Metadata columns (first 5): {len(metadata_cols)}")
    print(f"Sample data columns: {len(data_cols)}")
    print()
    
    # Separate metadata and methylation data
    df_metadata = df[metadata_cols]
    df_data = df[data_cols]
    
    # Check data types
    print("Data type check (sample columns only):")
    dtypes = df_data.dtypes.value_counts()
    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count} columns")
    print()
    
    # Check data range (beta values should be between 0 and 1)
    # Only calculate statistics on methylation data columns
    print("Beta value statistics (sample columns only):")
    min_val = df_data.min().min()
    max_val = df_data.max().max()
    mean_val = df_data.mean().mean()
    median_val = df_data.median().median()
    
    print(f"  Minimum: {min_val:.4f}")
    print(f"  Maximum: {max_val:.4f}")
    print(f"  Mean:    {mean_val:.4f}")
    print(f"  Median:  {median_val:.4f}")
    print()
    
    # Validate beta value range
    if min_val < 0 or max_val > 1:
        print("⚠ WARNING: Beta values outside expected range [0, 1]")
        print(f"  Found values: [{min_val:.4f}, {max_val:.4f}]")
        
        # Find problematic columns
        problematic_cols = []
        for col in df_data.columns:
            col_min = df_data[col].min()
            col_max = df_data[col].max()
            if col_min < 0 or col_max > 1:
                problematic_cols.append((col, col_min, col_max))
        
        if problematic_cols:
            print(f"  Problematic columns ({len(problematic_cols)} total):")
            for col, col_min, col_max in problematic_cols[:5]:  # Show first 5
                print(f"    {col}: [{col_min:.4f}, {col_max:.4f}]")
            if len(problematic_cols) > 5:
                print(f"    ... and {len(problematic_cols)-5} more")
        print()
        print("ERROR: Data validation failed. Please check your input file.")
        sys.exit(1)
    else:
        print("✓ All beta values in valid range [0, 1]")
    print()
    
    # Check for missing values (only in methylation data)
    n_missing = df_data.isnull().sum().sum()
    print(f"Missing values (sample columns): {n_missing:,} ({100*n_missing/(df_data.shape[0]*df_data.shape[1]):.2f}%)")
    print()
    
    # ========================================================================
    # Step 3: Extract and simplify tissue labels
    # ========================================================================
    print("Step 3: Processing tissue labels")
    print("-" * 80)
    
    # Get sample column names (columns 6+, excluding first 5 metadata columns)
    sample_columns = df_data.columns.tolist()
    print(f"Found {len(sample_columns)} sample columns")
    print()
    
    # Simplify tissue names
    simplified_labels = []
    tissue_count = {}  # Keep track of tissue types for numbering
    
    for col in sample_columns:
        simplified = simplify_tissue_name(col)
        
        # Add numbering for duplicate tissue types
        if simplified in tissue_count:
            tissue_count[simplified] += 1
            simplified_with_number = f"{simplified}_{tissue_count[simplified]}"
        else:
            tissue_count[simplified] = 1
            simplified_with_number = simplified
        
        simplified_labels.append(simplified_with_number)
    
    # Show examples of label conversion
    print("Example label conversions:")
    for i in range(min(5, len(sample_columns))):
        print(f"  {sample_columns[i][:50]:50s} -> {simplified_labels[i]}")
    print()
    
    # Count unique tissue types
    unique_tissues = set([label.rsplit('_', 1)[0] for label in simplified_labels])
    print(f"Number of unique tissue types: {len(unique_tissues)}")
    print()
    
    # ========================================================================
    # Step 4: Save tissue labels
    # ========================================================================
    print("Step 4: Saving tissue labels")
    print("-" * 80)
    
    # Create a mapping file for reference
    label_mapping = pd.DataFrame({
        'original_id': sample_columns,
        'simplified_label': simplified_labels
    })
    
    mapping_file = os.path.join(OUTPUT_DIR, "tissue_label_mapping.tsv")
    label_mapping.to_csv(mapping_file, sep='\t', index=False)
    print(f"✓ Saved label mapping to: {mapping_file}")
    
    # Save just the simplified labels (one per line) for easy loading
    with open(TISSUE_LABELS_FILE, 'w') as f:
        for label in simplified_labels:
            f.write(f"{label}\n")
    print(f"✓ Saved simplified labels to: {TISSUE_LABELS_FILE}")
    print()
    
    # ========================================================================
    # Step 5: Summary
    # ========================================================================
    print("="*80)
    print("INSPECTION COMPLETE")
    print("="*80)
    print(f"✓ Data loaded successfully: {df.shape[0]:,} regions × {len(sample_columns)} samples")
    print(f"✓ Beta values in valid range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"✓ Tissue labels simplified and saved")
    print()
    print("Ready to proceed to Script 02: simulate_reads.py")
    print("="*80)

if __name__ == "__main__":
    main()
