# Step 1.4 Scripts - Execution Summary

## 📦 What You Received

I've created **5 modular Python scripts** + **1 README** + **1 SLURM script** for Step 1.4:

### Core Scripts

1. **config.py** (8.0 KB)
   - Central configuration file
   - All paths, parameters, and settings
   - Easy to modify without touching other code

2. **data_loader.py** (13 KB)
   - Loads beta matrix, synthetic reads, 3-mer sequences
   - Validates data consistency
   - Provides clean interface to access data

3. **data_augmentation.py** (12 KB)
   - Implements jittering (±5% noise)
   - Coverage subsampling (500x, 100x, 50x, 30x, 10x)
   - Strand flipping (reverse complement)

4. **create_training_dataset.py** (21 KB)
   - Main orchestration script
   - Parallelized processing (40 cores)
   - Creates PyTorch-compatible .npz files
   - Progress tracking and logging

5. **README.md** (9.4 KB)
   - Comprehensive documentation
   - Usage instructions
   - Troubleshooting guide
   - Examples

6. **run_step1.4.sh** (SLURM script)
   - Job submission script for HPC
   - Pre-configured for 24 hours, 40 cores, 240GB RAM

## ⏱️ Time Estimate

**Full Run (119 samples):**
- **Wall time**: 3-5 hours with 40 CPU cores
- **Output size**: ~300 GB
- **Files created**: ~30.4 million .npz files

## 🚀 How to Run

### Step 1: Copy Scripts to Working Directory

```bash
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis

# Copy all scripts from outputs
cp computer:///mnt/user-data/outputs/*.py .
cp computer:///mnt/user-data/outputs/README.md .
cp computer:///mnt/user-data/outputs/run_step1.4.sh .
```

### Step 2: Test Run (RECOMMENDED)

Test on just 5 samples first (~30 minutes):

```bash
python3 create_training_dataset.py --test
```

**Expected output:**
- ~1.3 million files
- ~10 GB storage
- Verify everything works before full run

### Step 3: Full Run

**Option A: SLURM Job (Recommended)**
```bash
# Edit partition name in run_step1.4.sh first!
sbatch run_step1.4.sh
```

**Option B: Interactive**
```bash
python3 create_training_dataset.py
```

## 📊 What Gets Created

```
/home/chattopa/data_storage/MethAtlas_WGBSanalysis/
└── training_dataset/
    ├── all_data/                           # ~30.4M .npz files
    │   ├── sample_000_Adipocytes_chr1_273383_273777_aug0.npz
    │   ├── sample_000_Adipocytes_chr1_273383_273777_aug1.npz
    │   ├── sample_000_Adipocytes_chr1_273383_273777_aug2.npz
    │   ├── sample_000_Adipocytes_chr1_273383_273777_aug3.npz
    │   ├── sample_000_Adipocytes_chr1_273383_273777_aug4.npz
    │   └── ... (30.4 million total)
    ├── metadata.csv                        # Summary of all files
    └── logs/
        └── training_dataset_TIMESTAMP.log  # Execution log
```

## 🔍 Each Training File Contains

```python
{
    'dna_tokens': np.int32[150],        # 3-mer token IDs
    'methylation': np.uint8[150],       # Methylation pattern
    'tissue_label': np.float32[39],     # One-hot tissue label
    'n_reads': int,                     # Number of reads
    'avg_coverage': int                 # Average coverage
}
```

## 📈 Monitoring Progress

```bash
# Watch log file
tail -f training_dataset/logs/training_dataset_*.log

# Count files created
watch -n 10 'ls training_dataset/all_data/*.npz | wc -l'

# Check disk usage
watch -n 60 'du -sh training_dataset/'

# Check SLURM job
squeue -u chattopa
```

## ✅ Validation

The script automatically:
- ✓ Validates input files exist
- ✓ Checks data consistency across sources
- ✓ Validates each created file has correct structure
- ✓ Verifies data types and shapes
- ✓ Creates comprehensive metadata file

## 🎯 Data Augmentation (5x)

For **each** of 119 samples × 51,089 regions, creates **5 versions**:

| Version | Description | Jitter | Coverage | Purpose |
|---------|-------------|--------|----------|---------|
| v0 | Original | No | 500x | High-quality baseline |
| v1 | Jitter 100x | ±5% | 100x | Medium-high coverage |
| v2 | Jitter 50x | ±5% | 50x | Medium coverage |
| v3 | Jitter 30x | ±5% | 30x | Medium-low coverage |
| v4 | Jitter 10x | ±5% | 10x | Low coverage (like cfDNA) |

This makes the model robust to:
- Biological variation (jittering)
- Variable sequencing depth (coverage subsampling)
- Real-world cfDNA scenarios

## 🔧 Customization

All parameters are in `config.py`:

```python
# Change augmentation strategy
N_AUGMENTATIONS = 5          # Keep or change
AUGMENTATION_CONFIG = [...]  # Modify versions

# Processing
N_CORES = 40                 # Adjust if needed
CHUNK_SIZE = 100             # Reduce if memory issues

# Paths
BASE_DIR = "..."             # All paths centralized
```

## 🐛 Common Issues

**Issue**: Import errors
```bash
# Solution: Ensure all scripts in same directory
ls -la *.py README.md
```

**Issue**: Memory errors
```python
# In config.py, reduce:
CHUNK_SIZE = 50
N_CORES = 20
```

**Issue**: Disk space
```bash
# Check space first:
df -h /home/chattopa/data_storage/

# Run on subset:
python3 create_training_dataset.py --n-samples 10
```

## 📝 Expected Output (Full Run)

```
================================================================================
STEP 1.4: CREATE TRAINING DATASET STRUCTURE
================================================================================

CONFIGURATION SUMMARY:
  Base directory: /home/chattopa/data_storage/MethAtlas_WGBSanalysis
  Output directory: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset

AUGMENTATION STRATEGY:
  Number of augmentations: 5x
    v0: original - No jitter, 500x coverage
    v1: jitter_100x - Jitter ±5%, 100x coverage
    v2: jitter_50x - Jitter ±5%, 50x coverage
    v3: jitter_30x - Jitter ±5%, 30x coverage
    v4: jitter_10x - Jitter ±5%, 10x coverage

PROCESSING:
  CPU cores: 40
  Max memory: 240 GB
  Random seed: 42

EXPECTED OUTPUT:
  Estimated training examples: ~30,377,955
  Estimated storage: ~300 GB
================================================================================

...processing...

================================================================================
EXECUTION SUMMARY
================================================================================
Total samples processed: 119
Total regions processed: 6,079,591
Total augmentations created: 30,397,955
Total files created: 30,397,955
Failed regions: 0

Execution time: 3:42:15
Output directory: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset
Metadata file: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/metadata.csv
================================================================================
```

## 🎓 What Happens Next

After Step 1.4 completes:

1. **Step 1.5**: Split into train/val/test sets (70%/15%/15%)
2. **Phase 2**: Build DNABERT-S model architecture
3. **Phase 3**: Train the model on this dataset

## 📞 Need Help?

1. Read the comprehensive README.md
2. Check log files in `training_dataset/logs/`
3. Try test mode first: `--test`
4. Start small: `--n-samples 5`

## ✨ Key Features

✓ **Modular design** - Easy to understand and modify
✓ **Fully parallelized** - Uses all 40 cores efficiently
✓ **Progress tracking** - Real-time updates and logs
✓ **Automatic validation** - Catches errors immediately
✓ **Resumable** - Can restart if interrupted
✓ **Memory efficient** - Processes data in chunks
✓ **Well documented** - Comprehensive README and code comments

---

**Ready to run!** Start with `--test` mode to verify everything works.
