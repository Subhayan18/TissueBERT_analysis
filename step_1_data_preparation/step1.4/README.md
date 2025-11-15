# Step 1.4: Create Training Dataset Structure

## Overview

This module creates PyTorch-compatible training data for the DNABERT-S tissue deconvolution model. It takes synthetic reads, beta values, and 3-mer sequences from Steps 1.1-1.3 and creates augmented training files ready for model training.

## 📁 Project Structure

```
Step_1.4/
├── config.py                      # Central configuration
├── data_loader.py                 # Input data loading
├── data_augmentation.py           # Augmentation strategies
├── create_training_dataset.py     # Main orchestration script
└── README.md                      # This file
```

## 🎯 What This Step Does

**Input:**
- Beta matrix (82 samples × 51,089 regions)
- Synthetic reads (119 .npz files)
- 3-mer tokenized sequences
- Panel BED file
- Tissue labels

**Output:**
- **~595 training files** (.npz format) - ONE FILE PER SAMPLE-AUGMENTATION
  - 119 samples × 5 augmentations = 595 files
  - Each file contains ALL 51,089 regions for that sample-augmentation
- metadata.csv with file information
- Log files tracking progress

**Data Augmentation Strategy (5x):**
1. **v0 (original)**: No jitter, 500x coverage, forward strand
2. **v1 (jitter_100x)**: ±5% jitter, 100x coverage, forward strand
3. **v2 (jitter_50x)**: ±5% jitter, 50x coverage, forward strand
4. **v3 (jitter_30x)**: ±5% jitter, 30x coverage, forward strand
5. **v4 (jitter_10x)**: ±5% jitter, 10x coverage, forward strand

## 📦 Required Python Packages

```bash
# Install required packages
pip install numpy pandas tqdm --break-system-packages
```

**Note:** Since you're on a secure server with LMOD, always use `--break-system-packages` flag.

## ⚙️ Configuration

All parameters are in `config.py`. Key settings:

```python
# Paths
BASE_DIR = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "training_dataset")

# Processing
N_CORES = 40           # Use all CPU cores
MAX_MEMORY_GB = 240    # Maximum memory available

# Augmentation
N_AUGMENTATIONS = 5    # 5x augmentation

# Model parameters
N_TISSUES = 39
MAX_SEQUENCE_LENGTH = 150
```

## 🚀 Usage

### Test Run (Recommended First)

Process only 5 samples to verify everything works:

```bash
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis

# Copy scripts to working directory
cp /home/claude/*.py .

# Run test
python3 create_training_dataset.py --test
```

**Expected output:**
- ~25 files (5 samples × 5 augmentations)
- ~5-10 GB storage
- ~30 minutes runtime

### Limited Run

Process first N samples:

```bash
# Process first 10 samples
python3 create_training_dataset.py --n-samples 10
```

### Full Run

Process all 119 samples:

```bash
# Submit as SLURM job (recommended)
sbatch run_step1.4.sh

# Or run interactively (will take 3-5 hours)
python3 create_training_dataset.py
```

## 📊 Output Structure

### Training Files

Each `.npz` file contains one sample-augmentation combination with ALL regions:

```python
{
    'dna_tokens': np.int32[51089, 150],      # All regions × sequence length
    'methylation': np.uint8[51089, 150],     # All regions × sequence length  
    'tissue_label': np.float32[119],         # One-hot tissue label
    'region_ids': np.array[51089],           # Region identifiers
    'n_reads': np.int32[51089],              # Reads per region
    'sample_name': str,                      # Sample identifier
    'tissue_name': str                       # Tissue type name
}
```

**Filename format:**
```
sample_064_Bone_aug0.npz
sample_064_Bone_aug1.npz
sample_064_Bone_aug2.npz
...
```

**File size:** ~500-1000 MB per file

### Metadata File

`metadata.csv` contains:

| Column | Description |
|--------|-------------|
| filename | NPZ filename |
| sample_name | Sample identifier |
| tissue_type | Tissue name |
| tissue_index | Tissue index (0-118) |
| aug_version | Augmentation version (0-4) |
| n_regions | Number of regions (51,089) |
| total_reads | Total reads across all regions |
| seq_length | Sequence length (150) |

## 🔍 Data Augmentation Details

### 1. Jittering (±5%)

Adds biological noise to methylation values:
- For each CpG site, adjust methylation probability by ±5%
- Simulates natural biological variation
- Only applied to v1-v4 (not v0)

**Example:**
```
Original: 70% methylation → Jittered: 68-72% methylation
```

### 2. Coverage Subsampling

Randomly samples reads to achieve target coverage:
- v0: 500x (high coverage)
- v1: 100x (medium-high)
- v2: 50x (medium)
- v3: 30x (medium-low)
- v4: 10x (low coverage)

This prepares the model for variable sequencing depths in real cfDNA samples.

### 3. Strand Flipping

Converts to reverse complement:
- Reverses read order (5'→3' becomes 3'→5')
- Reverse complements 3-mer sequence
- Makes model strand-agnostic
- **Note:** Currently NOT used (all set to `False` in config)

## 💾 Storage Requirements

### Full Dataset (119 samples)

| Component | Size | Files |
|-----------|------|-------|
| Training files | ~300 GB | ~595 |
| Metadata | ~50 KB | 1 |
| Logs | ~100 MB | varies |
| **Total** | **~300 GB** | **~595** |

### Per Sample

- ~2.5 GB per sample (5 augmentations × ~500 MB each)

## ⏱️ Runtime Estimates

With 40 CPU cores:

| Mode | Samples | Runtime | Storage | Files |
|------|---------|---------|---------|-------|
| Test | 5 | ~30 min | ~10 GB | ~25 |
| Limited (10) | 10 | ~1 hour | ~25 GB | ~50 |
| Limited (20) | 20 | ~2 hours | ~50 GB | ~100 |
| **Full (119)** | **119** | **3-5 hours** | **~300 GB** | **~595** |

## 🐛 Troubleshooting

### Issue: Import errors

```bash
# Solution: Make sure all scripts are in the same directory
ls -la config.py data_loader.py data_augmentation.py create_training_dataset.py
```

### Issue: Memory errors

```python
# In config.py, reduce chunk size:
CHUNK_SIZE = 50  # Reduced from 100
```

### Issue: Disk space

```bash
# Check available space
df -h /home/chattopa/data_storage/

# If needed, run on subset first:
python3 create_training_dataset.py --n-samples 20
```

### Issue: Slow performance

```python
# In config.py, adjust cores:
N_CORES = 20  # Use fewer cores if system is overloaded
```

## ✅ Validation

The script automatically validates output files:

1. **File count check**: Expected vs. actual files created
2. **Structure validation**: Each file has required fields
3. **Data type validation**: Correct dtypes (int32, uint8, float32)
4. **Shape validation**: All arrays have correct dimensions
5. **One-hot validation**: Tissue labels sum to 1.0

### Manual Validation

```python
import numpy as np

# Load a training file
data = np.load('training_dataset/all_data/sample_064_Bone_aug0.npz')

# Check contents
print("Keys:", data.files)
print("DNA tokens shape:", data['dna_tokens'].shape)  # Should be (51089, 150)
print("Methylation shape:", data['methylation'].shape)  # Should be (51089, 150)
print("Tissue label shape:", data['tissue_label'].shape)  # Should be (119,)
print("Tissue label sum:", data['tissue_label'].sum())  # Should be 1.0
print("Number of regions:", len(data['region_ids']))  # Should be 51089
```

## 🔧 Customization

### Change Augmentation Strategy

Edit `AUGMENTATION_CONFIG` in `config.py`:

```python
AUGMENTATION_CONFIG = [
    {
        "version": 0,
        "description": "original",
        "jitter": False,
        "jitter_percent": 0.0,
        "coverage": 500,
        "strand_flip": False
    },
    # Add/modify versions here
]
```

### Change Jitter Amount

```python
# In config.py
"jitter_percent": 0.10,  # Change to ±10% instead of ±5%
```

### Add Strand Flipping

```python
# In config.py, set strand_flip to True for any version
"strand_flip": True
```

## 📈 Monitoring Progress

### Real-time Monitoring

```bash
# Watch log file
tail -f training_dataset/logs/training_dataset_*.log

# Check number of files created
watch -n 10 'ls training_dataset/all_data/*.npz | wc -l'
# Expected: ~595 for full run, ~25 for test

# Check disk usage
watch -n 60 'du -sh training_dataset/'
```

### SLURM Job Monitoring

```bash
# Check job status
squeue -u chattopa

# Check job output
tail -f slurm-JOBID.out
```

## 🔄 Restart from Failure

The script creates files atomically, so you can safely restart:

```bash
# Continue from where it left off
python3 create_training_dataset.py
```

Existing files will be overwritten, but this is safe since each file is independent.

## 📝 Log Files

Logs are saved in `training_dataset/logs/`:

```
training_dataset_20241113_143022.log  # Main execution log
```

Contains:
- Progress updates
- Error messages
- Validation results
- Final summary statistics

## 🎓 Understanding the Output

### What Goes into Each Training File?

1. **DNA Tokens**: The genomic sequence converted to 3-mer IDs
   - Example: "ATG TGC GCA" → [0, 21, 42]
   - 64 possible 3-mers (4³) + 5 special tokens

2. **Methylation Pattern**: Binary methylation state at each position
   - 0 = unmethylated CpG
   - 1 = methylated CpG
   - 2 = not a CpG site

3. **Tissue Label**: One-hot encoded vector
   - Length 39 (one per tissue type)
   - Single 1.0 at tissue index, rest 0.0

### How Augmentation Works

For each original sample-region:

```
Original Data (v0)
     ↓
Jitter ±5% (v1, v2, v3, v4)
     ↓
Subsample to target coverage (500x, 100x, 50x, 30x, 10x)
     ↓
Save as separate .npz file
```

## 🚦 Next Steps

After Step 1.4 completes successfully:

1. **Verify output**: Check file counts and metadata
2. **Step 1.5**: Split data into train/validation/test sets
3. **Phase 2**: Build DNABERT-S model architecture
4. **Phase 3**: Train the model

## 📞 Support

If you encounter issues:

1. Check the log files first
2. Try running in test mode (`--test`)
3. Reduce number of samples (`--n-samples 5`)
4. Check available disk space and memory

## 📚 References

- **Roadmap**: `/mnt/project/PDAC_cfDNA_Deconvolution_Roadmap.md`
- **Step 1.4 Details**: Lines 125-148 in Roadmap
- **Data Augmentation**: Lines 143-147 in Roadmap

---

**Author**: Step 1.4 Training Dataset Creator  
**Date**: 2024-11-13  
**Version**: 1.0
