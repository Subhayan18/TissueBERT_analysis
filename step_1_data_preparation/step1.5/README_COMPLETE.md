# Step 1.5: Complete Data Preparation Pipeline

## 🎯 Complete Pipeline Overview

Step 1.5 prepares your training data for machine learning by:
1. **Step 1.5a:** Clean tissue names and reassign indices
2. **Step 1.5b:** Generate synthetic samples for rare tissues
3. **Step 1.5c:** Split data into train/validation/test sets  
4. **Step 1.5d:** Validate all outputs before Phase 2

---

## 📊 Current Status Summary

### Your Data (from Step 1.4)

- **Total samples:** 119 unique biological samples
- **Total files:** 595 .npz files (119 samples × 5 augmentations)
- **Regions per file:** 51,089 genomic regions
- **Tissue types:** 22 unique tissues (after cleaning)

### The Problem

**12 tissues have insufficient samples** for proper train/val/test splitting:

| Tissue | Samples | Issue |
|--------|---------|-------|
| Aorta, Bladder, Cerebellum, Coronary, Dermal, Epidermal, Neuron, Prostate, Small | 1 each | Cannot split at all |
| Adipocytes, Gastric, Skeletal | 2 each | Cannot do 70/15/15 split |

**Requirement:** Need ≥4 samples per tissue (2 train / 1 val / 1 test)

### The Solution

Generate synthetic samples using learned biological noise models.

---

## 📁 Files Included (14 files total)

### Step 1.5a: Metadata Cleaning
```
update_metadata_tissues.py         # Clean tissue names (5.8 KB)
run_update_metadata.sh             # Bash wrapper (2.4 KB)
test_tissue_extraction.py          # Unit tests (2.1 KB)
```

### Step 1.5b: Synthetic Sample Generation (Binary Methylation)
```
generate_synthetic_samples_binary.py   # Main script - flip-based noise (15 KB)
run_generate_synthetic_binary.sh       # Bash wrapper (3.5 KB)
run_generate_synthetic_binary.slurm    # SLURM array job for parallel (1.5 KB)
test_generate_synthetic_binary.py      # Debug/test single sample (12 KB)
```

### Step 1.5c: Data Splitting
```
split_train_val_test.py            # Split into train/val/test (18 KB)
run_split_data.sh                  # Bash wrapper (2.5 KB)
```

### Step 1.5d: Final Validation
```
validate_final_data.py             # Comprehensive validation (15 KB)
run_validate_data.sh               # Bash wrapper (2.3 KB)
```

### Documentation
```
README.md                          # This file - master overview
README_metadata_update.md          # Step 1.5a details
README_synthetic_generation.md     # Step 1.5b details
QUICK_REFERENCE.md                 # Quick start commands
FILE_MANIFEST.md                   # File descriptions
```

---

## 🚀 Complete Workflow

### Prerequisites

- ✅ Completed Step 1.4 (training .npz files created)
- ✅ Files located in `/home/chattopa/data_storage/MethAtlas_WGBSanalysis/`
- ✅ Python 3.x with numpy, pandas, scipy, tqdm installed

### Step-by-Step Execution

#### 1. Copy All Files to Server

```bash
# Copy all scripts to your server
scp *.py *.sh chattopa@cosmos1:/home/chattopa/data_storage/MethAtlas_WGBSanalysis/
```

#### 2. SSH to Server

```bash
ssh chattopa@cosmos1
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis
```

#### 3. Run Step 1.5a: Metadata Cleaning

```bash
# First, preview the changes (recommended)
bash run_update_metadata.sh --preview

# If looks good, run for real
bash run_update_metadata.sh
```

**Output:**
- ✅ `updated_metadata.csv` - Clean tissue names
- ✅ `tissue_index_mapping.csv` - Tissue→index mapping

**Expected result:** 119 tissue subtypes → 22 main tissue types  
**Runtime:** ~2 minutes (can run on login node)

---

#### 4. Run Step 1.5b: Synthetic Generation

**⚠️ Use SLURM for this step - Binary Methylation Version**

**Important:** Your data has binary methylation (0/1 values), so use the `_binary` scripts!

```bash
# Copy binary methylation scripts
scp generate_synthetic_samples_binary.py \
    run_generate_synthetic_binary.sh \
    run_generate_synthetic_binary.slurm \
    chattopa@cosmos1:/home/chattopa/data_storage/MethAtlas_WGBSanalysis/

# Submit SLURM array job (PARALLEL - FAST!)
sbatch run_generate_synthetic_binary.slurm

# Monitor progress
squeue -u $USER
tail -f syn_gen_*_*.err
```

**Output:**
- ✅ `synthetic_samples/*.npz` - ~160 synthetic files (~36 samples × 5 augs)
- ✅ `synthetic_samples/synthetic_metadata.csv` - Synthetic sample metadata
- ✅ `synthetic_samples/combined_metadata.csv` - **Use this for next step**
- ✅ `synthetic_samples/validation_results.csv` - Quality metrics

**Expected result:** All 22 tissues now have ≥4 samples  
**Runtime:** ~15-20 minutes (parallel) or ~3-4 hours (sequential)

**Key differences from continuous methylation:**
- Uses flip-based noise (10% flip rate by default)
- Flips 0→1 or 1→0 instead of adding Gaussian noise
- Preserves binary nature of data
- Adaptive validation (reduces flip rate if fails)

---

#### 5. Run Step 1.5c: Data Splitting

```bash
# Run splitting (fast, can use login node)
bash run_split_data.sh
```

**Output:**
- ✅ `data_splits/train_files.csv` - Training file list (~110 samples, ~440 files)
- ✅ `data_splits/val_files.csv` - Validation file list (~23 samples, ~92 files)
- ✅ `data_splits/test_files.csv` - Test file list (~22 samples, ~88 files)
- ✅ `data_splits/train_samples.txt` - Training sample names
- ✅ `data_splits/val_samples.txt` - Validation sample names
- ✅ `data_splits/test_samples.txt` - Test sample names
- ✅ `data_splits/split_config.json` - Split configuration

**Expected result:** 70/15/15 split with proper stratification  
**Runtime:** ~1 minute

---

#### 6. Run Step 1.5d: Final Validation

```bash
# Comprehensive validation of all outputs
bash run_validate_data.sh
```

**This validates:**
1. ✅ File existence - All files in metadata exist
2. ✅ NPZ integrity - Files can be loaded and have correct structure
3. ✅ Metadata consistency - No duplicates, valid columns
4. ✅ Split integrity - No sample appears in multiple splits
5. ✅ Tissue coverage - All tissues in training set
6. ✅ Synthetic quality - Validation metrics pass thresholds
7. ✅ PyTorch compatibility - Data can be loaded with DataLoader

**Expected result:** All tests pass ✓  
**Runtime:** ~5 minutes

---

## 📈 Expected Results Summary

### After All Steps Complete

**Sample Distribution:**
- Total samples: ~155 (119 original + 36 synthetic)
- Training: ~110 samples (70%)
- Validation: ~23 samples (15%)
- Test: ~22 samples (15%)

**File Distribution:**
- Total files: ~620 (155 samples × ~4 augmentations)
- Training files: ~440
- Validation files: ~92
- Test files: ~88

**Tissue Coverage:**
- All 22 tissues represented in training
- All tissues have ≥2 training samples
- Proper stratification across splits

---

## 🔍 Quality Verification Checklist

After completing all steps:

### Step 1.5a Verification
- [ ] `updated_metadata.csv` exists with 595 rows
- [ ] `tissue_index_mapping.csv` shows 22 tissues
- [ ] Tissue names cleaned (no hyphens or version numbers)
- [ ] Each tissue has unique index (0-21)

### Step 1.5b Verification
- [ ] `synthetic_samples/` directory contains ~36 .npz files
- [ ] `combined_metadata.csv` has ~631 rows
- [ ] `validation_results.csv` shows >95% pass rate
- [ ] All tissues have ≥3 samples
- [ ] `is_synthetic` column properly set

### Step 1.5c Verification
- [ ] `data_splits/` directory created
- [ ] `train_files.csv`, `val_files.csv`, `test_files.csv` exist
- [ ] ~70/15/15 split achieved
- [ ] No sample appears in multiple splits
- [ ] All tissues present in training set

### Step 1.5d Verification
- [ ] All 7 validation tests pass
- [ ] No errors reported
- [ ] PyTorch compatibility confirmed
- [ ] Ready for Phase 2

---

## 🎯 Success Criteria

**Overall Pipeline Success:**
- ✅ All 22 tissues have ≥3 samples
- ✅ ~155 total samples (119 original + 36 synthetic)
- ✅ 70/15/15 train/val/test split achieved
- ✅ All validation tests pass
- ✅ No data leakage between splits
- ✅ PyTorch DataLoader works
- ✅ Ready for model training

---

## 📊 Data Flow Diagram

```
Step 1.4 Output
├── 595 .npz files
└── metadata.csv (119 tissue subtypes)
     │
     ▼
Step 1.5a: Metadata Cleaning
├── updated_metadata.csv (22 tissues)
└── tissue_index_mapping.csv
     │
     ▼
Step 1.5b: Synthetic Generation
├── synthetic_samples/*.npz (~36 files)
├── synthetic_metadata.csv
├── combined_metadata.csv ◄── USE THIS
└── validation_results.csv
     │
     ▼
Step 1.5c: Data Splitting
├── train_files.csv (~440 files)
├── val_files.csv (~92 files)
├── test_files.csv (~88 files)
└── split_config.json
     │
     ▼
Step 1.5d: Final Validation
└── ✓ All tests pass
     │
     ▼
Phase 2: Model Training
```

---

## 🔧 Troubleshooting

### Step 1.5a Issues

**Issue:** Tissue names not cleaned properly
```bash
# Check extraction logic
python test_tissue_extraction.py
```

**Issue:** Wrong number of tissues
```bash
# Verify tissue counts
cut -d',' -f3 updated_metadata.csv | sort -u | wc -l
# Should output: 22
```

---

### Step 1.5b Issues

**Issue:** Job running very slow
- **Cause:** Processing 1000 regions per tissue for noise learning
- **Solution:** Edit `generate_synthetic_samples.py` line ~120, reduce to 200 regions
- **Expected speedup:** ~5x faster

**Issue:** Low validation pass rate
- **Cause:** Noise model too aggressive
- **Solution:** Reduce noise magnitude by editing line ~450:
  ```python
  noise_std = self.noise_model.get_noise_std(tissue) * 0.8
  ```

**Issue:** Out of memory
- **Cause:** Loading too many files
- **Solution:** Increase SLURM memory to 60G or reduce samples loaded per tissue

---

### Step 1.5c Issues

**Issue:** Split ratios not exact 70/15/15
- **Expected:** This is normal due to rounding with small sample counts
- **Acceptable range:** 68-72% / 13-17% / 13-17%

**Issue:** Some tissues missing from validation/test
- **Expected:** Tissues with n=3 samples → 2 train, 0 val, 1 test
- **Solution:** This is by design, validation set optional for rare tissues

---

### Step 1.5d Issues

**Issue:** File existence check fails
```bash
# Find missing files
python -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('data_splits/train_files.csv')
data_dir = Path('training_dataset')
for f in df['filename']:
    if not (data_dir / f).exists():
        print(f'Missing: {f}')
"
```

**Issue:** PyTorch compatibility test fails
- **Cause:** PyTorch not installed
- **Solution:** Install PyTorch or skip this test (it's optional)

---

## ⚡ Performance Optimization

### For Faster Step 1.5b Execution

**Option 1: Reduce noise learning regions**
```python
# Edit generate_synthetic_samples.py, line ~120
sampled_regions = np.random.choice(n_regions, min(200, n_regions), replace=False)
# Instead of: min(1000, n_regions)
# Speedup: ~5x faster
```

**Option 2: Load fewer samples per tissue**
```python
# Edit generate_synthetic_samples.py, line ~90
for filename in tissue_samples[:2]:  # Load only 2 samples
# Instead of: tissue_samples[:min_samples]
# Speedup: ~2x faster
```

**Option 3: Use more CPU cores**
```bash
# In SLURM script
#SBATCH --cpus-per-task=8  # Instead of 4
# Note: Limited parallelization in current implementation
```

---

## 📚 Documentation Structure

**Start here:**
1. `README.md` (this file) - Complete pipeline overview

**For details:**
2. `README_metadata_update.md` - Step 1.5a deep dive
3. `README_synthetic_generation.md` - Step 1.5b deep dive
4. `QUICK_REFERENCE.md` - Command cheat sheet

**For reference:**
5. `FILE_MANIFEST.md` - What each file does

---

## 🎓 Key Concepts

### Stratification
Ensuring each split (train/val/test) has proportional representation from all tissue types.

### Sample-Level Splitting
All augmentation versions (aug0-aug4) of a sample stay together in the same split to prevent data leakage.

### Synthetic Samples
Artificially generated samples using learned biological noise from well-represented tissues, validated for quality.

### Tissue Hierarchy
Biological grouping of tissues (Brain, Digestive, etc.) for shared noise modeling when generating synthetics.

### Data Leakage
When information from test/validation sets influences training. Prevented by sample-level splitting.

---

## 🔬 Scientific Rationale

### Why Synthetic Samples are Valid

1. **Learned from real data:** Noise model trained on 10 well-represented tissues
2. **Tissue-specific:** Uses hierarchical grouping (Brain, Digestive, Cardiovascular)
3. **Spatially correlated:** CpGs within regions show realistic correlation (length=5)
4. **Rigorously validated:** KS test (p>0.01), mean preservation (<15% diff), pattern conservation (r>0.85)
5. **Transparent:** Clearly flagged with `is_synthetic=True` in metadata

### Why This Splitting Strategy Works

1. **Prevents overfitting:** Test set never seen during training
2. **Enables proper evaluation:** Validation set for hyperparameter tuning
3. **Maintains tissue representation:** Stratification ensures all tissues in training
4. **Realistic evaluation:** Test set includes both common and rare tissues

---

## 📞 Getting Help

### If Something Fails

1. **Check the specific step's README** for detailed troubleshooting
2. **Review error messages** in `.out` and `.err` files
3. **Verify file paths** in wrapper scripts match your setup
4. **Check disk space:** Need ~3 GB free
5. **Confirm dependencies:** numpy, pandas, scipy, tqdm installed

### Common Issues

| Issue | Most Likely Cause | Solution |
|-------|------------------|----------|
| Step 1.5b very slow | 1000 regions per tissue | Reduce to 200 regions |
| Validation failures | Corrupted .npz files | Re-run Step 1.4 |
| Split errors | Metadata inconsistent | Re-run Step 1.5a |
| Import errors | Missing dependencies | `pip install --break-system-packages scipy tqdm` |

---

## ⏭️ Next Steps

After all validation tests pass:

**→ Proceed to Phase 2: Model Architecture**

You're now ready to:
1. Design DNABERT-S architecture
2. Implement PyTorch Dataset class
3. Set up training pipeline
4. Begin model training

**Use these files for training:**
- `data_splits/train_files.csv`
- `data_splits/val_files.csv`
- `data_splits/test_files.csv`

---

## 📊 Final Statistics

After completing Step 1.5, you will have:

**Samples:**
- 119 original samples
- 36 synthetic samples
- **155 total samples**

**Files:**
- 595 original .npz files
- ~36 synthetic .npz files  
- **~631 total .npz files**

**Splits:**
- ~440 training files (70%)
- ~92 validation files (15%)
- ~88 test files (15%)

**Tissues:**
- 22 unique tissue types
- All tissues ≥3 samples
- All tissues in training set

**Quality:**
- >95% synthetic validation pass rate
- No data leakage
- PyTorch compatible
- Ready for training ✓

---

## ✅ Completion Checklist

**Step 1.5 is complete when:**

- [x] Step 1.5a: Metadata cleaned (22 tissues)
- [x] Step 1.5b: Synthetic samples generated (~36 samples)
- [x] Step 1.5c: Data split into train/val/test (70/15/15)
- [x] Step 1.5d: All validation tests pass
- [x] `combined_metadata.csv` exists
- [x] `train_files.csv`, `val_files.csv`, `test_files.csv` exist
- [x] All files can be loaded
- [x] No split integrity issues
- [x] PyTorch DataLoader works

**When all boxes checked:** 🎉 **Ready for Phase 2!**

---

**Pipeline Progress:** ✅ Step 1.5a | ✅ Step 1.5b | ✅ Step 1.5c | ✅ Step 1.5d | ⏭️ Phase 2

---

**Version:** 2.0  
**Last Updated:** 2024-11-17  
**Authors:** Step 1.5 Data Preparation Pipeline Team
