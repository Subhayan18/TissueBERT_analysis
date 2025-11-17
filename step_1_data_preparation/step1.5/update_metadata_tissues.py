#!/usr/bin/env python3
"""
Step 1.5 - Update Metadata with Clean Tissue Types
==================================================

This script:
1. Reads the original metadata.csv
2. Extracts clean tissue names (first part before hyphen)
3. Assigns unique tissue indices (0, 1, 2, ...)
4. Creates updated_metadata.csv with corrected tissue information

Usage:
    python update_metadata_tissues.py --input metadata.csv --output updated_metadata.csv
"""

import pandas as pd
import argparse
from pathlib import Path


def extract_tissue_name(sample_name):
    """
    Extract clean tissue name from sample_name.
    
    Examples:
        sample_040_Liver-Hepatocytes_3 -> Liver
        sample_064_Bone -> Bone
        sample_117_Colon-Left-Endocrine_2 -> Colon
        sample_090_Blood-NK_2 -> Blood
    
    Args:
        sample_name: Full sample name from metadata
    
    Returns:
        Clean tissue name (first part before hyphen)
    """
    # Remove the sample_XXX_ prefix (e.g., "sample_040_")
    # Split by underscore and take everything after the second underscore
    parts = sample_name.split('_', 2)  # Split on first TWO underscores
    
    if len(parts) < 3:
        # Handle edge case where format is just "sample_XXX"
        return sample_name
    
    tissue_part = parts[2]  # Get everything after "sample_XXX_"
    
    # Now extract the tissue name (first part before hyphen)
    if '-' in tissue_part:
        tissue_name = tissue_part.split('-')[0]
    else:
        # No hyphen, check if there's an underscore with number suffix
        if '_' in tissue_part:
            tissue_name = tissue_part.split('_')[0]
        else:
            tissue_name = tissue_part
    
    return tissue_name


def main():
    parser = argparse.ArgumentParser(
        description='Update metadata.csv with clean tissue types and indices'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to original metadata.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output updated_metadata.csv'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview changes without saving'
    )
    
    args = parser.parse_args()
    
    # Read original metadata
    print(f"Reading metadata from: {args.input}")
    df = pd.read_csv(args.input)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Extract clean tissue names
    print("\nExtracting clean tissue names...")
    df['tissue'] = df['sample_name'].apply(extract_tissue_name)
    
    # Show some examples of the transformation
    print("\n" + "="*70)
    print("TISSUE NAME EXTRACTION EXAMPLES:")
    print("="*70)
    examples = df[['sample_name', 'tissue_type', 'tissue']].drop_duplicates('sample_name').head(20)
    for _, row in examples.iterrows():
        print(f"{row['sample_name']:50s} -> {row['tissue']:20s} (was: {row['tissue_type']})")
    
    # Count unique tissues
    unique_tissues = df['tissue'].unique()
    n_unique = len(unique_tissues)
    print(f"\n" + "="*70)
    print(f"TISSUE SUMMARY:")
    print("="*70)
    print(f"Total unique tissue types: {n_unique}")
    print(f"\nAll unique tissues (alphabetically):")
    for i, tissue in enumerate(sorted(unique_tissues), 1):
        count = (df['tissue'] == tissue).sum()
        n_samples = df[df['tissue'] == tissue]['sample_name'].nunique()
        print(f"  {i:3d}. {tissue:30s} ({n_samples:3d} samples, {count:4d} files)")
    
    # Create tissue index mapping (alphabetically sorted)
    tissue_to_index = {tissue: idx for idx, tissue in enumerate(sorted(unique_tissues))}
    
    # Assign new tissue indices
    df['tissue_index'] = df['tissue'].map(tissue_to_index)
    
    # Reorder columns: filename, sample_name, tissue, tissue_index, aug_version, ...
    columns_order = ['filename', 'sample_name', 'tissue', 'tissue_index', 'aug_version', 
                     'n_regions', 'total_reads', 'seq_length']
    
    # Keep any additional columns that might exist
    other_cols = [col for col in df.columns if col not in columns_order]
    final_columns = columns_order + other_cols
    
    df_updated = df[final_columns]
    
    # Show sample distribution by tissue
    print(f"\n" + "="*70)
    print("SAMPLES PER TISSUE:")
    print("="*70)
    tissue_samples = df_updated.groupby('tissue')['sample_name'].nunique().sort_values(ascending=False)
    for tissue, n_samples in tissue_samples.items():
        tissue_idx = tissue_to_index[tissue]
        print(f"  {tissue:30s} (index {tissue_idx:3d}): {n_samples:3d} samples")
    
    if args.preview:
        print("\n" + "="*70)
        print("PREVIEW MODE - No files written")
        print("="*70)
        print("\nFirst 10 rows of updated data:")
        print(df_updated.head(10).to_string())
    else:
        # Save updated metadata
        output_path = Path(args.output)
        df_updated.to_csv(output_path, index=False)
        print(f"\n" + "="*70)
        print(f"Updated metadata saved to: {output_path}")
        print("="*70)
        print(f"Total rows: {len(df_updated)}")
        print(f"Unique samples: {df_updated['sample_name'].nunique()}")
        print(f"Unique tissues: {df_updated['tissue'].nunique()}")
        print(f"Tissue index range: 0-{df_updated['tissue_index'].max()}")
        
        # Also save the tissue mapping for reference
        mapping_file = output_path.parent / "tissue_index_mapping.csv"
        tissue_mapping = pd.DataFrame([
            {'tissue_index': idx, 'tissue_name': tissue}
            for tissue, idx in sorted(tissue_to_index.items(), key=lambda x: x[1])
        ])
        tissue_mapping.to_csv(mapping_file, index=False)
        print(f"\nTissue index mapping saved to: {mapping_file}")


if __name__ == "__main__":
    main()
