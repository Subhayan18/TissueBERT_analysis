# Phase 1: Data Preparation

## Goal
Prepare a panel-specific training dataset for tissue deconvolution using methylation data from the Loyfer atlas.

## Substeps

### 1.1 Extract Panel Regions
- **Input**: Loyfer WGBS beta values, custom panel BED file, tissue labels.
- **Process**: Intersect panel regions with CpG blocks from Loyfer data.
- **Output**: Panel-specific beta matrix (82 samples × ~45k regions).

### 1.2 Simulate Read-Level Data
- **Why**: DNABERT-S requires read-level methylation patterns.
- **Process**: Simulate 150bp reads per region using beta values, add biological noise and CpG correlation.
- **Output**: Synthetic reads (~10M total), formatted as `[sample_id, read_id, region_id, methylation_pattern]`.

### 1.3 Add DNA Sequence Context
- **Input**: Reference genome (hg38), panel BED file.
- **Process**: Extract DNA sequences and tokenize into 3-mers.
- **Output**: Tokenized sequences in DNABERT format.

### 1.4 Format Training Dataset
- **Structure**: PyTorch-compatible `.npz` files with DNA tokens, methylation patterns, and tissue labels.
- **Metadata**: CSV file with sample IDs, tissue types, region IDs, and coverage.

### 1.5 Data Splitting
- **Strategy**: Split by sample (not region) into train/val/test sets.
- **Considerations**: Stratify by tissue type, handle rare tissues carefully.

## Outputs Summary
| Output Type         | Format         | Purpose                     |
|---------------------|----------------|-----------------------------|
| Panel beta matrix   | `.npz`         | Core training data          |
| Synthetic reads     | `.parquet`     | Read-level methylation      |
| 3-mer sequences     | `.json`        | DNA context for model       |
| Training dataset    | PyTorch        | Model input                 |
| Data splits         | `.csv`         | Train/val/test indices      |
