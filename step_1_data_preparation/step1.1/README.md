# Stage 1.1: Extract Panel Regions from Loyfer Atlas

## Overview

This stage extracts methylation data from the Loyfer et al. WGBS atlas (82 samples, 39 tissue types) for only the genomic regions covered by our custom PDAC detection panel.

**Goal:** Create a panel-specific beta value matrix (82 samples × N blocks) for downstream model training.

---

## Input Files

1. **Panel BED file** (genomic coordinates of capture regions)
   - Format: `chr  start  end`
   - Example: `Probes_merged_ok_Methyl_Lund_MRD_MTE-96334990_hg38_lowfilter.bed`
   - ~45,942 regions

2. **Loyfer WGBS data** (82 tissue samples)
   - Format: `.beta` files (wgbstools binary format)
   - Location: Directory containing all 82 samples
   - Naming: `GSM#######_TissueType-Details.hg38.beta`

3. **Loyfer block index** (CpG block definitions)
   - Format: `chr  start  end  start_cpg  end_cpg  block_idx`
   - Example: `GSE186458_blocks.s207.hg38.index.bed`
   - Used for annotation

---

## Output Files

1. **TWIST_blocks.bed** - Panel regions annotated with CpG block indices
   - Format: `chr  start  end  start_block_idx  end_block_idx`
   - Used as input for beta value extraction

2. **panel_beta_matrix.tsv** - Methylation values for panel regions
   - Rows: Panel blocks/regions
   - Columns: 82 samples (GSM IDs from filenames)
   - Values: Average methylation (0-1 scale)

---

## Pipeline Steps

### Step 1: Annotate Panel Regions with CpG Block Indices

Use `wgbstools convert` to add CpG block indices to panel coordinates.

```bash
# Load required module
module load wgbstools

# Convert panel BED to include CpG indices
wgbstools convert \
  --bed <PANEL_BED_FILE> \
  --genome hg38 \
  --output TWIST_blocks.bed
```

**Input:** Panel BED file with 3 columns (chr, start, end)

**Output:** `TWIST_blocks.bed` with 5 columns:
```
chr    start    end      start_block_idx  end_block_idx
chr1   273383   273777   2778             2782
chr1   273778   273898   2782             2784
chr1   605698   606012   5475             5483
```

---

### Step 2: Extract Beta Values for Panel Regions

Use `wgbstools beta_to_table` to extract methylation data from all 82 Loyfer samples.

```bash
# Extract beta values for panel blocks
wgbstools beta_to_table \
  TWIST_blocks.bed \
  --betas <BETA_FILES_DIR>/*.beta \
  --output panel_beta_matrix.tsv \
  --min_cov 4 \
  --threads 8
```

**Parameters:**
- `TWIST_blocks.bed`: Positional argument (blocks file with ≥5 columns)
- `--betas`: Path to all 82 .beta files (use wildcard: `*.beta`)
- `--min_cov 4`: Only include blocks with ≥4 CpG observations
- `--threads 8`: Use 8 CPU cores for parallel processing

**Output:** Tab-separated matrix
- Rows: Panel regions/blocks
- Columns: 5 annotation columns (chr, start, end, start_cpg, end_cpg) + 82 sample columns
- Sample column names derived from beta filenames (e.g., `GSM5652291_Blood-T-EffMem-CD4-Z00000416.hg38`)

---

## Complete Example

```bash
#!/bin/bash

# Navigate to working directory
cd /path/to/project

# Load module
module load wgbstools

# Step 1: Convert panel coordinates to block indices
wgbstools convert \
  --bed Probes_merged_ok_Methyl_Lund_MRD_MTE-96334990_hg38_lowfilter.bed \
  --genome hg38 \
  --output TWIST_blocks.bed

# Verify output
head TWIST_blocks.bed
wc -l TWIST_blocks.bed

# Step 2: Extract beta values
wgbstools beta_to_table \
  TWIST_blocks.bed \
  --betas /path/to/beta/files/*.beta \
  --output panel_beta_matrix.tsv \
  --min_cov 4 \
  --threads 8

# Verify output
head panel_beta_matrix.tsv
wc -l panel_beta_matrix.tsv
```

---

## Quality Checks

After running the pipeline, verify:

1. **TWIST_blocks.bed dimensions:**
   ```bash
   # Check number of regions
   wc -l TWIST_blocks.bed
   
   # Verify 5 columns present
   head -1 TWIST_blocks.bed | awk '{print NF}'
   ```

2. **panel_beta_matrix.tsv dimensions:**
   ```bash
   # Check number of rows (should match TWIST_blocks.bed)
   wc -l panel_beta_matrix.tsv
   
   # Check number of columns (should be 5 annotation + 82 samples = 87)
   head -1 panel_beta_matrix.tsv | awk '{print NF}'
   ```

3. **Missing values:**
   ```bash
   # Count missing values (represented as 'nan' or empty cells)
   grep -o "nan" panel_beta_matrix.tsv | wc -l
   ```

4. **Methylation value range:**
   ```bash
   # Values should be between 0 and 1
   # Quick check: look at a few data rows
   head -5 panel_beta_matrix.tsv
   ```

---

## Expected Results

**Panel Coverage:**
- Starting regions: ~45,942
- Regions with valid blocks: ~40,000-45,000 (depending on block overlap)
- Average CpGs per region: 5-10

**Data Matrix:**
- Dimensions: ~40K-45K rows × 87 columns (5 annotation + 82 samples)
- File size: ~50-100 MB (text format)
- Missing data: <5% (low coverage regions)

**Tissue Distribution (39 tissue types):**
- Blood cells: ~20-25 samples
- Solid tissues: ~50-55 samples
- Each tissue type: 1-5 replicates

---

## Next Steps

After completing Stage 1.1, proceed to:

**Stage 1.2:** Create read-level training data
- Simulate realistic 150bp cfDNA reads from block beta values
- Add biological noise and CpG correlations
- Generate ~10M synthetic reads for training

**Stage 1.3:** Add DNA sequence context
- Extract reference genome sequences for each panel region
- Convert to 3-mer tokens for DNABERT input

---

## Computational Requirements

- **Time:** ~10-30 minutes (depending on I/O speed)
- **Memory:** ~4-8 GB RAM
- **Storage:** ~100 MB for outputs
- **Dependencies:** wgbstools, Python 3.8+

---

## Troubleshooting

**Issue:** `wgbstools convert` fails with "genome not found"
- **Solution:** Ensure hg38 reference is indexed. Check wgbstools installation.

**Issue:** Beta files not found
- **Solution:** Verify beta file directory path and use full paths with wildcards carefully.

**Issue:** Very high missing data (>20%)
- **Solution:** Lower `--min_cov` threshold or check if panel regions match genome build (hg38).

**Issue:** Sample count mismatch
- **Solution:** Verify all 82 beta files are present in the directory.

---

## References

- Loyfer et al. (2023). A DNA methylation atlas of normal human cell types. *Nature*
- wgbstools: https://github.com/nloyfer/wgbs_tools
- Panel design: Custom PDAC MRD detection panel (45,942 regions)

---

## Contact

For questions about this pipeline stage, refer to project documentation or create an issue in the repository.
