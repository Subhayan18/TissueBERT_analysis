# Step 1.2: Simulate Read-Level Training Data

## Overview

This directory contains scripts to generate synthetic cfDNA reads from the `panel_beta_matrix.tsv` file created in Step 1.1. These reads will be used to train the DNABERT-S model for PDAC deconvolution.

## Scientific Background

- **Input**: Block-level beta values (average methylation across molecules)
- **Output**: Individual synthetic reads with binary methylation patterns
- **Why**: DNABERT-S requires read-level data, but Loyfer atlas provides bulk data
- **Method**: Stochastic simulation based on beta values + biological noise

## Files in This Directory

### Python Scripts (run in order)

1. `01_inspect_data.py` - Load and validate input data (fast, ~1 min)
2. `02_simulate_reads.py` - Generate synthetic reads (slow, ~10-20 hours)
3. `03_verify_output.py` - Quality check outputs (fast, ~5 min)

### SLURM Wrapper

4. `run_step1.2.sh` - Runs all 3 scripts in sequence on SLURM

### Documentation

5. `LMOD_SETUP_INSTRUCTIONS.md` - How to load required LMOD modules
6. `README.md` (this file) - Complete guide
7. `QUICKSTART_CHECKLIST.md` - Step-by-step checklist

## Prerequisites

### 1. Completed Step 1.1

- `panel_beta_matrix.tsv` must exist in:
  ```
  /home/chattopa/data_storage/MethAtlas_WGBSanalysis/
  ```

### 2. LMOD modules loaded

- Follow instructions in `LMOD_SETUP_INSTRUCTIONS.md`
- Required modules: Python/3.11.3, SciPy-bundle/2023.07
- All dependencies managed through LMOD

### 3. Compute resources

- 250 GB RAM
- 48 CPU cores
- 72 hours walltime
- ~50 GB disk space for outputs

## Installation

1. Create a directory for the scripts:

   ```bash
   mkdir -p $HOME/scripts/step1.2
   cd $HOME/scripts/step1.2
   ```

2. Copy all 4 scripts to this directory:

   ```bash
   cp /path/to/01_inspect_data.py .
   cp /path/to/02_simulate_reads.py .
   cp /path/to/03_verify_output.py .
   cp /path/to/run_step1.2.sh .
   ```

3. Make the SLURM script executable:

   ```bash
   chmod +x run_step1.2.sh
   ```

4. Create logs directory:

   ```bash
   mkdir -p logs
   ```

5. Load required LMOD modules:

   Follow `LMOD_SETUP_INSTRUCTIONS.md`

## Usage

### Option A: Run everything automatically via SLURM (RECOMMENDED)

```bash
sbatch run_step1.2.sh
```

This will:
- Load LMOD modules
- Run all 3 scripts in sequence
- Save logs to `logs/step1.2_<jobid>.out`

### Option B: Run scripts individually (for testing/debugging)

```bash
# Load LMOD modules
module reset
module load GCC/12.3.0
module load SciPy-bundle/2023.07
module load Python/3.11.3

# Run each script
python 01_inspect_data.py
python 02_simulate_reads.py
python 03_verify_output.py
```

## What Each Script Does

### Script 01: Inspect Data (`01_inspect_data.py`)

**Duration**: ~1 minute

**What it does**:
- Loads `panel_beta_matrix.tsv`
- Validates data dimensions and value ranges
- Simplifies tissue labels from column headers
  - Example: `GSM5652224_Neuron-Z000000TH.hg38` → `Neuron`
- Saves `tissue_labels.txt` for next script

**Outputs**:
- `tissue_labels.txt`
- `tissue_label_mapping.tsv`

**What to check**:
- Number of regions should be 45,942
- Number of samples should be 82
- Beta values should be between 0 and 1

### Script 02: Simulate Reads (`02_simulate_reads.py`)

**Duration**: ~10-20 hours (depends on system speed)

**What it does**:
- For each of 82 samples:
  - For each of 45,942 regions:
    - Generates 500 synthetic reads
    - Each read has 3-8 CpG sites (stochastic)
    - Methylation state sampled from beta value
    - Adds 5% biological noise
- Saves compressed NPZ files (one per sample)

**Outputs**:
- 82 NPZ files: `sample_000_Neuron.npz`, `sample_001_Cortex-Neuron_1.npz`, ...
- Total reads generated: ~1.88 billion
- Total output size: ~10-20 GB (compressed)

**What to check**:
- All 82 NPZ files created
- File sizes reasonable (100-300 MB each)
- No error messages in output

### Script 03: Verify Output (`03_verify_output.py`)

**Duration**: ~5 minutes

**What it does**:
- Loads and inspects all NPZ files
- Checks data integrity (correct shapes, valid values)
- Calculates summary statistics
- Generates quality report

**Outputs**:
- `simulation_summary.txt` (detailed report)
- `sample_statistics.csv` (per-sample metrics)

**What to check**:
- "All checks passed successfully!" message
- No integrity warnings
- Methylation rates look reasonable (0.3-0.7 typical range)

## Configuration Options

If you want to modify simulation parameters, edit `02_simulate_reads.py`:

| Line | Parameter | Default | Description |
|------|-----------|---------|-------------|
| 27 | `READS_PER_REGION` | 500 | Number of reads per region. Higher = more data but slower |
| 28 | `BIOLOGICAL_NOISE` | 0.05 | Noise level (5% error rate). Typical: 0.01-0.10 |
| 29 | `READ_LENGTH` | 150 | cfDNA fragment length in bp. Usually keep at 150 |

## Output Directory Structure

```
/home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads/
│
├── sample_000_Neuron.npz                  # Synthetic reads for sample 0
├── sample_001_Cortex-Neuron_1.npz         # Synthetic reads for sample 1
├── ...
├── sample_081_<tissue>.npz                # Synthetic reads for sample 81
│
├── tissue_labels.txt                      # Simplified tissue names
├── tissue_label_mapping.tsv               # Original → simplified mapping
├── simulation_summary.txt                 # Quality report
└── sample_statistics.csv                  # Per-sample metrics
```

## NPZ File Format

Each NPZ file contains:

### Arrays

- `methylation_data` - Concatenated methylation patterns (0=U, 1=M)
- `read_boundaries` - Start/end positions of each read
- `read_lengths` - Number of CpGs in each read
- `region_indices` - Which region each read came from
- `region_ids` - Original region identifiers
- `read_ids` - Unique read identifiers

### Metadata

- `tissue_label` - Tissue type for this sample
- `reads_per_region` - Number of reads generated per region (500)
- `biological_noise` - Noise level used (0.05)
- `read_length` - Read length in bp (150)

### To load in Python

```python
import numpy as np
data = np.load('sample_000_Neuron.npz')
methylation = data['methylation_data']
tissue = str(data['tissue_label'])
```

## Troubleshooting

### Problem: "File not found" error for panel_beta_matrix.tsv

**Solution**: Check that Step 1.1 is complete and file exists:
```bash
ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/panel_beta_matrix.tsv
```

### Problem: "module: command not found"

**Solution**: LMOD should be available by default. Contact your system administrator if not.

### Problem: "ModuleNotFoundError" for numpy/pandas/scipy

**Solution**: Load the required LMOD modules following `LMOD_SETUP_INSTRUCTIONS.md`
```bash
module load SciPy-bundle/2023.07
```

### Problem: Job killed or out of memory

**Solution**: 
- Check if 250GB is available: `sinfo -N -l`
- Request more memory in `run_step1.2.sh` line 5:
  ```bash
  #SBATCH --mem=300G
  ```

### Problem: Script 02 taking too long

**Solution**: This is normal! Generating ~2 billion reads takes time
- Expected: 10-20 hours
- Monitor progress in log file:
  ```bash
  tail -f logs/step1.2_<jobid>.out
  ```

### Problem: NPZ files look too small

**Solution**: They're compressed! Check uncompressed size:
```python
import numpy as np
data = np.load('sample_000_Neuron.npz')
uncompressed_mb = sum(arr.nbytes for arr in data.values()) / (1024**2)
print(f"Uncompressed: {uncompressed_mb:.1f} MB")
```

## Checking Job Status

### Submit job
```bash
sbatch run_step1.2.sh
```

### Check queue
```bash
squeue -u $USER
```

### Check job details
```bash
scontrol show job <jobid>
```

### Monitor log in real-time
```bash
tail -f logs/step1.2_<jobid>.out
```

### Cancel job if needed
```bash
scancel <jobid>
```

## Expected Outputs

When everything runs successfully, you should see:

### 1. In terminal/log
```
✓ Script 01 completed successfully
✓ Script 02 completed successfully
✓ Script 03 completed successfully
✓ All checks passed successfully!
```

### 2. In output directory
- 82 NPZ files (total ~10-20 GB)
- 4 text/CSV files with metadata and statistics

### 3. Ready for next step
```
Ready to proceed to Step 1.3: Add DNA Sequence Context
```

## Next Steps

After Step 1.2 completes successfully:

1. Review the `simulation_summary.txt` file
2. Check `sample_statistics.csv` for any anomalies
3. Proceed to **Step 1.3: Add DNA Sequence Context**
   - Extract reference genome sequences for each region
   - Convert sequences to 3-mer tokens (DNABERT format)
   - Combine with methylation data

## Technical Details

### Simulation Algorithm

For each region with beta value β:
```
1. Sample n_cpgs from Poisson(λ=5)  # CpG density
2. For each read:
   a. For each CpG:
      - Sample M ~ Bernoulli(β)
      - Apply noise: flip M with probability 0.05
   b. Save [read_id, region_id, methylation_pattern]
```

### Memory Usage

- Input data: ~500 MB (`panel_beta_matrix.tsv`)
- Working memory: ~50-100 GB (during simulation)
- Output data: ~10-20 GB (compressed NPZ files)

### Computation Time

- Depends on system speed
- Rough estimate: ~100 regions/second
- Total: 45,942 regions × 82 samples = 3,767,244 regions
- Expected time: ~10-20 hours on 48 cores

## Reference

This simulation approach is based on:
- Loyfer et al. (2023) "A DNA methylation atlas of normal human cell types"
- MethylBERT paper (Kim et al. 2025) for read-level simulation
- DNABERT-S architecture for input format requirements

## Version History

**Version 1.0** (2025-11-11):
- Initial release
- Scripts for Step 1.2 of PDAC cfDNA deconvolution pipeline
