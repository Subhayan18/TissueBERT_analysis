# Step 1.3: Extract DNA Sequences and Create 3-mer Tokenization

## Overview

This step extracts DNA sequences from the reference genome for your custom panel regions and converts them into 3-mer tokenization format required by DNABERT.

**Purpose:** Provide sequence context for methylation patterns to train the DNABERT-S tissue deconvolution model.

---

## What This Step Does

### 1.3.1: Extract DNA Sequences
- Reads your panel BED file containing genomic coordinates
- Extracts the corresponding DNA sequences from the hg38 reference genome
- Handles both forward and reverse strand orientations
- Outputs sequences in FASTA format

### 1.3.2: Create 3-mer Tokenization
- Converts DNA sequences into overlapping 3-letter chunks (k-mers)
- Example: `ATCGATCG` → `ATC TCG CGA GAT ATC TCG`
- This format is required for DNABERT input
- Outputs space-separated 3-mer strings

---

## Requirements

### Input Files
1. **BED file**: Panel regions with genomic coordinates
   - Format: `chr\tstart\tend\t[optional fields]`
   - Location: `/home/chattopa/data_storage/MethAtlas_WGBSanalysis/TWIST_blocks.bed`

2. **Reference genome**: hg38 FASTA file (can be gzipped)
   - Location: `/home/chattopa/data_storage/wgbs_tools/references/hg38/hg38.fa.gz`

### LMOD Modules (HPC)
```bash
module reset
module load GCC/12.3.0
module load Python/3.11.3
module load BEDTools/2.31.0
```

### Python Dependencies
- Standard library only (no external packages required)
- Compatible with Python 3.7+

---

## Usage

### Option 1: Submit SLURM Job (Recommended for HPC)

```bash
# Make script executable
chmod +x run_step1.3.sh

# Submit job
sbatch run_step1.3.sh
```

### Option 2: Run Directly

```bash
# Load modules
module reset
module load GCC/12.3.0
module load Python/3.11.3
module load BEDTools/2.31.0

# Run script
python step1.3_extract_sequences_and_tokenize.py \
    --bed /home/chattopa/data_storage/MethAtlas_WGBSanalysis/TWIST_blocks.bed \
    --fasta /home/chattopa/data_storage/wgbs_tools/references/hg38/hg38.fa.gz \
    --output-dir ./step1.3_output \
    --output-prefix TWIST_panel_sequences
```

---

## Script Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--bed` | Yes | - | Path to BED file with panel regions |
| `--fasta` | Yes | - | Path to reference genome (gzipped or uncompressed) |
| `--output-dir` | No | `./step1.3_output` | Directory for output files |
| `--output-prefix` | No | `panel_sequences` | Prefix for output filenames |

---

## Output Files

### 1. `TWIST_panel_sequences.fa`
FASTA file with extracted DNA sequences:
```
>chr1_12345_12445
ATCGATCGATCGATCG...
>chr1_23456_23556
GCTAGCTAGCTAGCTA...
```

### 2. `TWIST_panel_sequences_3mers.txt`
Tab-separated file with 3-mer tokenization:

| Column | Description |
|--------|-------------|
| region_id | Unique identifier (chr_start_end) |
| chrom | Chromosome name |
| start | Start position (0-based) |
| end | End position (0-based, exclusive) |
| strand | + or - |
| sequence_length | Length of DNA sequence (bp) |
| n_3mers | Number of 3-mers generated |
| 3mer_sequence | Space-separated 3-mer tokens |

Example:
```
region_id           chrom  start   end     strand  sequence_length  n_3mers  3mer_sequence
chr1_12345_12445    chr1   12345   12445   +       100             98       ATC TCG CGA...
```

---

## Expected Performance

- **Input**: ~45,942 panel regions (TWIST panel)
- **Processing time**: ~30-60 minutes (depending on HPC resources)
- **Memory usage**: ~8-16 GB (loading hg38 reference)
- **Output size**: 
  - FASTA: ~10-20 MB
  - 3-mer file: ~50-100 MB

---

## Validation

After running, verify outputs:

```bash
# Check number of regions
wc -l step1.3_output/TWIST_panel_sequences_3mers.txt
# Expected: ~45,943 lines (45,942 regions + 1 header)

# Check FASTA entries
grep -c "^>" step1.3_output/TWIST_panel_sequences.fa
# Expected: ~45,942 entries

# View first few 3-mers
head -n 5 step1.3_output/TWIST_panel_sequences_3mers.txt
```

---

## How It Works

### 1. Load Reference Genome
```python
# Reads hg38.fa.gz into memory
# Handles gzipped files automatically
# Stores each chromosome as a string
```

### 2. Process Each BED Region
```python
# For each line in BED file:
#   1. Parse coordinates (chr, start, end)
#   2. Extract sequence from reference
#   3. Handle strand orientation (reverse complement if needed)
#   4. Convert to 3-mers
#   5. Write to output files
```

### 3. 3-mer Generation Algorithm
```python
def dna_to_3mer(sequence):
    # For sequence "ATCGATCG":
    # Position 0-2: ATC
    # Position 1-3: TCG
    # Position 2-4: CGA
    # Position 3-5: GAT
    # Position 4-6: ATC
    # Position 5-7: TCG
    # Result: "ATC TCG CGA GAT ATC TCG"
```

---

## Troubleshooting

### Error: "BED file not found"
**Solution:** Check the path and ensure file exists
```bash
ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/TWIST_blocks.bed
```

### Error: "FASTA file not found"
**Solution:** Verify reference genome location
```bash
ls -lh /home/chattopa/data_storage/wgbs_tools/references/hg38/hg38.fa.gz
```

### Error: "Out of memory"
**Solution:** Increase SLURM memory allocation
```bash
#SBATCH --mem=32G  # Increase from 16G to 32G
```

### Warning: "Regions skipped"
**Cause:** Region coordinates outside chromosome boundaries or invalid sequences
**Impact:** Usually minimal (<1% of regions)
**Action:** Check log for details, generally safe to proceed

---

## Next Steps

After completing Step 1.3:

1. **Verify outputs** using validation commands above
2. **Proceed to Step 1.4**: Create Training Dataset Structure
   - Combine methylation patterns (from Step 1.2) with sequences
   - Format data for PyTorch DataLoader
   - Create train/validation/test splits

---

## File Structure

```
project_root/
├── step1.3_extract_sequences_and_tokenize.py  # Main Python script
├── run_step1.3.sh                             # SLURM submission script
├── README.md                                  # This file
├── logs/                                      # Job logs (auto-created)
│   ├── step1.3_<job_id>.out
│   └── step1.3_<job_id>.err
└── step1.3_output/                            # Output directory
    ├── TWIST_panel_sequences.fa               # DNA sequences
    └── TWIST_panel_sequences_3mers.txt        # 3-mer tokens
```

---

## Technical Details

### Why 3-mers?
- DNABERT architecture uses k-mer tokenization (k=3 is standard)
- Balances sequence context with vocabulary size
- 3-mers capture dinucleotide dependencies (important for CpG methylation)
- Vocabulary size: 4³ = 64 possible 3-mers

### Strand Handling
- Reverse strand regions are automatically reverse-complemented
- Ensures consistent 5'→3' orientation
- Maintains biological accuracy for methylation analysis

### Memory Optimization
- Loads entire reference genome once (efficient for many regions)
- Processes BED file line-by-line (streaming, low memory)
- Suitable for panels with thousands to millions of regions

---

## Contact & Support

For issues or questions:
1. Check SLURM logs: `logs/step1.3_<job_id>.err`
2. Refer to project roadmap: `PDAC_cfDNA_Deconvolution_Roadmap.md`
3. Validate inputs using commands in Troubleshooting section

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-13 | Initial release |

---

## References

- DNABERT paper: [Ji et al. 2021](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)
- Project roadmap: `PDAC_cfDNA_Deconvolution_Roadmap.md` (Step 1.3)
- BED format specification: [UCSC Genome Browser](https://genome.ucsc.edu/FAQ/FAQformat.html#format1)

---

## License

This script is part of the PDAC cfDNA Deconvolution Project.
