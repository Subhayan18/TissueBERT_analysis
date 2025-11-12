#!/usr/bin/env python3
"""
Quick Data Validation Script
=============================
Run this before 01_inspect_data.py to check your panel_beta_matrix.tsv file.
This will help identify any data format issues.

Usage:
    python validate_input_data.py
"""

import sys

# File to check
INPUT_FILE = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/panel_beta_matrix.tsv"

print("="*80)
print("QUICK DATA VALIDATION")
print("="*80)
print()

# Check 1: File exists
print("Check 1: File exists")
print("-" * 80)
try:
    import os
    if os.path.exists(INPUT_FILE):
        file_size = os.path.getsize(INPUT_FILE) / (1024**2)
        print(f"✓ File found: {INPUT_FILE}")
        print(f"  Size: {file_size:.2f} MB")
    else:
        print(f"✗ File not found: {INPUT_FILE}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error checking file: {e}")
    sys.exit(1)
print()

# Check 2: File structure
print("Check 2: File structure")
print("-" * 80)
try:
    with open(INPUT_FILE, 'r') as f:
        header = f.readline().strip()
        first_data = f.readline().strip()
        
        # Count tabs
        n_header_cols = header.count('\t') + 1
        n_data_cols = first_data.count('\t') + 1
        
        print(f"✓ Header columns: {n_header_cols}")
        print(f"✓ Data columns:   {n_data_cols}")
        
        if n_header_cols != n_data_cols:
            print("✗ WARNING: Header and data column counts don't match!")
        
        # Show first few column names
        header_cols = header.split('\t')
        print(f"\nFirst 3 column names:")
        for i, col in enumerate(header_cols[:3]):
            print(f"  {i+1}. {col[:50]}")
            
except Exception as e:
    print(f"✗ Error reading file: {e}")
    sys.exit(1)
print()

# Check 3: Load with pandas
print("Check 3: Loading with pandas")
print("-" * 80)
try:
    import pandas as pd
    import numpy as np
    
    # Load file
    df = pd.read_csv(INPUT_FILE, sep='\t', index_col=0, low_memory=False, nrows=100)
    print(f"✓ Loaded first 100 rows")
    print(f"  Shape: {df.shape}")
    print()
    
    # Check data types BEFORE conversion
    print("Original data types:")
    dtypes_before = df.dtypes.value_counts()
    for dtype, count in dtypes_before.items():
        print(f"  {dtype}: {count} columns")
    print()
    
    # Try converting to numeric
    print("Converting to numeric...")
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Count NaNs added by conversion
    nans_before = df.isnull().sum().sum()
    nans_after = df_numeric.isnull().sum().sum()
    new_nans = nans_after - nans_before
    
    if new_nans > 0:
        print(f"⚠ Warning: {new_nans} values couldn't be converted to numeric")
        print("  This suggests non-numeric data in the file")
        
        # Find problematic columns
        for col in df.columns:
            col_nans_before = df[col].isnull().sum()
            col_nans_after = df_numeric[col].isnull().sum()
            if col_nans_after > col_nans_before:
                print(f"  Column '{col}': {col_nans_after - col_nans_before} non-numeric values")
                # Show some examples
                non_numeric = df[col][pd.to_numeric(df[col], errors='coerce').isnull() & df[col].notnull()]
                if len(non_numeric) > 0:
                    print(f"    Examples: {list(non_numeric.head(3))}")
    else:
        print("✓ All values are numeric")
    print()
    
    # Check data types AFTER conversion
    print("Data types after conversion:")
    dtypes_after = df_numeric.dtypes.value_counts()
    for dtype, count in dtypes_after.items():
        print(f"  {dtype}: {count} columns")
    print()
    
except ImportError:
    print("✗ pandas not available. Please load LMOD modules:")
    print("  module load SciPy-bundle/2023.07")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Check 4: Validate beta values
print("Check 4: Validate beta value range")
print("-" * 80)
try:
    min_val = df_numeric.min().min()
    max_val = df_numeric.max().max()
    mean_val = df_numeric.mean().mean()
    median_val = df_numeric.median().median()
    
    print(f"Beta value statistics:")
    print(f"  Minimum: {min_val:.6f}")
    print(f"  Maximum: {max_val:.6f}")
    print(f"  Mean:    {mean_val:.6f}")
    print(f"  Median:  {median_val:.6f}")
    print()
    
    # Validate range
    if min_val < 0:
        print(f"✗ ERROR: Found negative values (min={min_val})")
        # Find columns with negative values
        neg_cols = [col for col in df_numeric.columns if df_numeric[col].min() < 0]
        print(f"  Columns with negative values: {neg_cols[:5]}")
    elif max_val > 1:
        print(f"✗ ERROR: Found values > 1 (max={max_val})")
        # Find columns with values > 1
        high_cols = [col for col in df_numeric.columns if df_numeric[col].max() > 1]
        print(f"  Columns with values > 1: {high_cols[:5]}")
    else:
        print("✓ All beta values in valid range [0, 1]")
        
except Exception as e:
    print(f"✗ Error validating values: {e}")
    sys.exit(1)
print()

# Check 5: Sample preview
print("Check 5: Data preview")
print("-" * 80)
try:
    print("First 3 rows, first 3 columns:")
    print(df_numeric.iloc[:3, :3].to_string())
    print()
    print("✓ Data looks reasonable")
except Exception as e:
    print(f"✗ Error showing preview: {e}")
print()

# Summary
print("="*80)
print("VALIDATION SUMMARY")
print("="*80)

# Determine overall status
issues = []
if new_nans > 0:
    issues.append(f"{new_nans} non-numeric values found")
if min_val < 0 or max_val > 1:
    issues.append(f"Values outside [0,1] range")

if issues:
    print("⚠ ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
    print()
    print("Please fix these issues before running 01_inspect_data.py")
    print("See BUGFIX_DATA_TYPES.md for troubleshooting help")
else:
    print("✓ ALL CHECKS PASSED")
    print()
    print("Your data file looks good!")
    print("You can now run: python 01_inspect_data.py")

print("="*80)
