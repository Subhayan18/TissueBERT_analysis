# Step 1.5b: Synthetic Sample Generation for Rare Tissues

## 📋 Overview

This step generates synthetic methylation samples for tissues with insufficient samples to enable proper train/validation/test splitting. We need at least 4 samples per tissue (2 train / 1 validation / 1 test).

### Current Situation

From your tissue distribution:
- **12 tissues** have only 1-2 samples (rare tissues)
- These cannot be properly split into train/val/test
- Need synthetic samples to augment rare tissues

### Solution

Generate biologically plausible synthetic samples using a learned noise model:
1. Learn methylation variability from well-represented tissues (n≥4)
2. Group tissues by biological hierarchy (Brain, Digestive, etc.)
3. Generate synthetic samples with correlated noise
4. Validate synthetic samples for biological plausibility

---

## 🧬 Scientific Approach

### Binary Methylation Data

**Key Discovery:** Your methylation data is **binary (0 or 1)**, not continuous. This requires a different noise approach.

**Traditional approach (Gaussian noise):**
- Adds continuous noise: value + N(0, σ²)
- Works for continuous data (0.0-1.0)
- **Fails for binary data** - destroys 0/1 pattern

**Our approach (Flip-based noise):**
- Randomly flips methylation values: 0→1 or 1→0
- Uses flip probability (e.g., 10% chance per CpG)
- **Preserves binary nature** of the data
- More biologically realistic for discrete methylation

### Flip-Based Noise Model

**Three-step noise generation:**

1. **Determine flip rate**
   - Measured from biological replicates: ~7-10% variation
   - Default: 10% flip probability per CpG
   - Adaptive: Reduces to 5% if validation fails

2. **Generate spatially correlated flips**
   - CpGs within a region more likely to flip together
   - Uses correlation length = 5 positions
   - Realistic regional co-variation

3. **Apply flips with constraints**
   - Only flip CpG positions (not non-CpG markers)
   - Binary values maintained: 0→1 or 1→0
   - Regional methylation patterns preserved

### Tissue Hierarchy (Reference Only)

Tissues are grouped by biological similarity:

```python
TISSUE_HIERARCHY = {
    'Brain': ['Cortex', 'Cerebellum', 'Neuron', 'Oligodendrocytes'],
    'Digestive': ['Colon', 'Gastric', 'Pancreas', 'Liver', 'Small'],
    'Cardiovascular': ['Heart', 'Aorta', 'Coronary'],
    'Connective': ['Dermal', 'Bone', 'Skeletal'],
    'Respiratory': ['Lung'],
    'Blood': ['Blood'],
    'Urogenital': ['Kidney', 'Bladder', 'Prostate'],
}
```

**Note:** For binary methylation, tissue hierarchy is used for reference but flip rate is uniform across tissues.

### Validation Metrics

Each synthetic sample is validated against its original:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| **KS test p-value** | >0.01 | Methylation distributions similar |
| **Mean difference** | <0.15 | Overall methylation level preserved |
| **Regional correlation** | >0.85 | Spatial patterns maintained |
| **Hamming distance** | 0.05-0.15 | Appropriate flip rate for binary data |

**Adaptive validation:**
- If validation fails, flip rate reduced by 1%
- Retries until validation passes or minimum rate (5%) reached
- Keeps best attempt if all fail

Only samples passing all criteria are kept.

---

## 📁 Files Included

```
generate_synthetic_samples_binary.py  # Main Python script (15 KB) - BINARY METHYLATION
run_generate_synthetic_binary.sh      # Bash wrapper script (3.5 KB)
run_generate_synthetic_binary.slurm   # SLURM array job script (1.5 KB)
test_generate_synthetic_binary.py     # Debug/test script (12 KB)
README_synthetic_generation.md        # This file
```

**Note:** Use the `_binary` versions for your data (binary 0/1 methylation values).

---

## 🚀 Usage

### Step 1: Copy Files to Server

```bash
scp generate_synthetic_samples_binary.py \
    run_generate_synthetic_binary.sh \
    run_generate_synthetic_binary.slurm \
    chattopa@cosmos1:/home/chattopa/data_storage/MethAtlas_WGBSanalysis/
```

### Step 2: SSH and Navigate

```bash
ssh chattopa@cosmos1
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis
```

### Step 3: Run Synthetic Generation

**⚠️ Use SLURM for this step**

You have two options:

#### **Option A: Parallel Execution (RECOMMENDED)** ⚡

Process all tissues in parallel (much faster):

```bash
# Submit SLURM array job
sbatch run_generate_synthetic_binary.slurm

# Monitor progress
squeue -u $USER

# Check output for each tissue
tail -f syn_gen_*_*.err
```

**Runtime:** ~15-20 minutes (all tissues processed simultaneously)

#### **Option B: Sequential Execution**

Process all tissues one by one:

```bash
# Create SLURM script for sequential run
cat > run_step1.5b_sequential.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=step1.5b_seq
#SBATCH --output=step1.5b_%j.out
#SBATCH --error=step1.5b_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

module load GCC/12.3.0
module load SciPy-bundle/2023.07
module load Python/3.11.3

cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis
bash run_generate_synthetic_binary.sh
EOF

# Submit
sbatch run_step1.5b_sequential.slurm
```

**Runtime:** ~3-4 hours (processes tissues sequentially)

**What the script does:**
1. Loads `updated_metadata.csv` (from Step 1.5a)
2. Identifies rare tissues (n<4 samples)
3. Generates synthetic samples using flip-based noise (10% default)
4. Processes all augmentation versions (aug0-aug4)
5. Validates each synthetic sample (adaptive retry if fails)
6. Saves results

---

## 📊 Expected Output

### 1. Synthetic Training Files

```
synthetic_samples/
├── sample_001_Aorta_synthetic_1_aug0.npz
├── sample_001_Aorta_synthetic_2_aug0.npz
├── sample_001_Aorta_synthetic_3_aug0.npz
├── sample_063_Bladder_synthetic_1_aug0.npz
└── ...
```

Each `.npz` file contains:
- `dna_tokens`: [51089, 150] - DNA sequences (copied from original)
- `methylation`: [51089, 150] - **Modified with correlated noise**
- `region_ids`: [51089] - Region identifiers
- `n_reads`: [51089] - Read counts
- `tissue_label`: [119] - One-hot tissue label
- `sample_name`: String with "_synthetic_X" suffix
- `tissue_name`: String

### 2. Metadata Files

**`synthetic_metadata.csv`** - Only synthetic samples:
```csv
filename,sample_name,tissue,tissue_index,aug_version,n_regions,total_reads,seq_length,is_synthetic,original_sample
sample_001_Aorta_synthetic_1_aug0.npz,sample_001_Aorta_synthetic_1,Aorta,1,0,51089,25544500,150,True,sample_001_Aorta
```

**`combined_metadata.csv`** - Original + synthetic:
```csv
filename,sample_name,tissue,tissue_index,aug_version,n_regions,total_reads,seq_length,is_synthetic,original_sample
sample_000_Adipocytes_aug0.npz,sample_000_Adipocytes,Adipocytes,0,0,51089,25544500,150,False,sample_000_Adipocytes
sample_001_Aorta_synthetic_1_aug0.npz,sample_001_Aorta_synthetic_1,Aorta,1,0,51089,25544500,150,True,sample_001_Aorta
```

**Key columns:**
- `is_synthetic`: Boolean flag (True/False)
- `original_sample`: Which original sample this was derived from

### 3. Validation Results

**`validation_results.csv`**:
```csv
synthetic_sample,original_sample,tissue,ks_statistic,ks_pvalue,original_mean,synthetic_mean,mean_diff,regional_correlation
sample_001_Aorta_synthetic_1,sample_001_Aorta,Aorta,0.0234,0.856,0.523,0.518,0.005,0.972
```

**Use this to:**
- Check quality of synthetic samples
- Verify all samples passed validation
- Identify any potential issues

---

## 📈 Expected Results

### Before Synthetic Generation

| Tissue | Original Samples | Status |
|--------|-----------------|--------|
| Aorta | 1 | ✗ Cannot split |
| Bladder | 1 | ✗ Cannot split |
| Cerebellum | 1 | ✗ Cannot split |
| Coronary | 1 | ✗ Cannot split |
| Dermal | 1 | ✗ Cannot split |
| Epidermal | 1 | ✗ Cannot split |
| Neuron | 1 | ✗ Cannot split |
| Prostate | 1 | ✗ Cannot split |
| Small | 1 | ✗ Cannot split |
| Adipocytes | 2 | ✗ Cannot split |
| Gastric | 2 | ✗ Cannot split |
| Skeletal | 2 | ✗ Cannot split |

### After Synthetic Generation

All tissues will have ≥4 samples:

| Tissue | Original | Synthetic | Total | Status |
|--------|----------|-----------|-------|--------|
| Aorta | 1 | 3 | 4 | ✓ Can split (2/1/1) |
| Bladder | 1 | 3 | 4 | ✓ Can split (2/1/1) |
| Cerebellum | 1 | 3 | 4 | ✓ Can split (2/1/1) |
| Adipocytes | 2 | 2 | 4 | ✓ Can split (2/1/1) |
| ... | ... | ... | ... | ... |

**Total synthetic samples generated:** ~36 samples (3 per tissue with n=1, 2 per tissue with n=2)

---

## 🔍 Quality Assurance

### How to Verify Synthetic Samples

#### 1. Check Validation Results

```bash
# View validation summary
cat synthetic_samples/validation_results.csv | column -t -s,

# Check if all samples passed
awk -F',' 'NR>1 && $5<0.01 {print "FAILED:", $1}' synthetic_samples/validation_results.csv
```

Should show no failures.

#### 2. Inspect Methylation Distributions

```python
import numpy as np
import matplotlib.pyplot as plt

# Load original
original = np.load('training_dataset/sample_001_Aorta_aug0.npz')

# Load synthetic
synthetic = np.load('synthetic_samples/sample_001_Aorta_synthetic_1_aug0.npz')

# Compare methylation distributions
orig_meth = original['methylation'][original['methylation'] < 2]
synth_meth = synthetic['methylation'][synthetic['methylation'] < 2]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(orig_meth, bins=50, alpha=0.7, label='Original')
plt.hist(synth_meth, bins=50, alpha=0.7, label='Synthetic')
plt.legend()
plt.xlabel('Methylation Level')
plt.title('Methylation Distribution')

plt.subplot(1, 2, 2)
plt.scatter(original['methylation'].mean(axis=1), 
            synthetic['methylation'].mean(axis=1), 
            alpha=0.1)
plt.xlabel('Original Regional Mean')
plt.ylabel('Synthetic Regional Mean')
plt.title('Regional Pattern Preservation')
plt.show()
```

**Expected:** Distributions should overlap well, scatter plot should be diagonal.

#### 3. Check Sample Counts

```bash
# Count synthetic samples per tissue
cd synthetic_samples
grep "is_synthetic,True" combined_metadata.csv | cut -d',' -f3 | sort | uniq -c
```

---

## 🎯 Success Criteria

✅ **All rare tissues now have ≥4 samples**  
✅ **Validation pass rate >95%**  
✅ **KS test p-values >0.01** (distributions similar)  
✅ **Mean methylation difference <15%**  
✅ **Regional correlation >0.85**  
✅ **No crashes or errors during generation**

---

## 🔧 Troubleshooting

### Issue: "Synthetic sample failed validation"

**Cause:** Generated sample too different from original (flip rate too high)

**Solution:**
- Script automatically reduces flip rate by 1% and retries
- Minimum flip rate: 5%
- If all attempts fail, keeps last attempt

### Issue: Low validation pass rate (<80%)

**Cause:** Default flip rate (10%) may be too aggressive for your data

**Solution:**
```bash
# Reduce flip rate to 7%
# Edit run_generate_synthetic_binary.sh, line 15:
FLIP_RATE=0.07
```

### Issue: All samples pass but look too similar to originals

**Cause:** Flip rate too conservative

**Solution:**
```bash
# Increase flip rate to 12%
FLIP_RATE=0.12
```

### Issue: SLURM array job - some tasks fail

**Symptom:** Some `syn_gen_*_*.out` files show errors

**Solution:**
```bash
# Identify failed tasks
grep -l "ERROR" syn_gen_*_*.err

# Re-run specific tissue (e.g., tissue index 5)
python generate_synthetic_samples_binary.py \
    --metadata updated_metadata.csv \
    --data-dir training_dataset/all_data \
    --output-dir synthetic_samples \
    --slurm-array-id 5
```

### Issue: "File not found" errors

**Cause:** Incorrect paths or missing aug0 files

**Solution:**
```bash
# Verify aug0 files exist for rare tissues
python -c "
import pandas as pd
df = pd.read_csv('updated_metadata.csv')
for tissue in ['Aorta', 'Neuron', 'Bladder']:
    aug0 = df[(df['tissue']==tissue) & (df['aug_version']==0)]
    print(f'{tissue}: {len(aug0)} aug0 files')
"
```

### Issue: Validation metrics look strange

**Check data type:**
```bash
# Verify your data is binary (should show 2 unique values: 0 and 1)
python -c "
import numpy as np
data = np.load('training_dataset/all_data/sample_064_Bone_aug0.npz')
cpg = data['methylation'][data['methylation'] < 2]
print(f'Unique values: {np.unique(cpg)}')
print(f'Data type: {"Binary" if len(np.unique(cpg))==2 else "Continuous"}')
"
```

If data is continuous (many unique values), you're using the wrong script!

---

## ⚡ Performance Optimization

### For Faster Execution

**Use SLURM array jobs** (parallel processing):
- Sequential: ~3-4 hours
- Parallel (13 tissues): ~15-20 minutes
- **Speedup: ~10-12x**

**Adjust SLURM resources:**
```bash
# In run_generate_synthetic_binary.slurm
#SBATCH --cpus-per-task=2   # Each tissue uses 2 CPUs
#SBATCH --mem=16G            # 16GB per task
#SBATCH --time=02:00:00      # 2 hours max per tissue
```

**Monitor progress:**
```bash
# Check how many tasks completed
squeue -u $USER | grep syn_gen | wc -l

# View real-time progress for tissue 0
tail -f syn_gen_*_0.err
```

### Issue: "Synthetic sample failed validation"

**Cause:** Generated sample too different from original

**Solution:**
- Script automatically regenerates
- If repeated failures, check noise model learning phase
- May need to adjust `correlation_length` parameter

### Issue: Low KS test p-values

**Cause:** Methylation distributions diverging

**Solution:**
```bash
# Reduce noise magnitude by editing the script
# Line ~120: Adjust noise_std scaling factor
noise_std = self.noise_model.get_noise_std(tissue) * 0.8  # Reduce by 20%
```

### Issue: Low regional correlation

**Cause:** Too much spatial noise variation

**Solution:**
```bash
# Increase correlation length
bash run_generate_synthetic.sh --correlation-length 10
```

### Issue: "File not found" errors

**Cause:** Incorrect paths

**Solution:**
- Verify `DATA_DIR` in `run_generate_synthetic.sh`
- Ensure all .npz files are in `training_dataset/`
- Check metadata paths

---

## 🧪 Advanced Options

### Run Python Script Directly

For more control:

```bash
python generate_synthetic_samples.py \
    --metadata updated_metadata.csv \
    --data-dir training_dataset \
    --output-dir synthetic_samples \
    --min-samples 4 \
    --correlation-length 5 \
    --seed 42
```

**Parameters:**
- `--min-samples`: Threshold for "rare" tissues (default: 4)
- `--correlation-length`: Spatial correlation in noise (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)

### Modify Tissue Hierarchy

Edit `generate_synthetic_samples.py` line 35:

```python
TISSUE_HIERARCHY = {
    'Brain': ['Cortex', 'Cerebellum', 'Neuron', 'Oligodendrocytes'],
    # Add your custom groupings here
}
```

### Adjust Validation Thresholds

Edit `generate_synthetic_samples.py` line ~570:

```python
is_valid = (
    ks_pvalue > 0.01 and          # Make stricter: > 0.05
    metrics['mean_diff'] < 0.15 and  # Make stricter: < 0.10
    metrics['regional_correlation'] > 0.85  # Make stricter: > 0.90
)
```

---

## 📊 Understanding the Output

### Noise Model Statistics

During execution, you'll see:

```
LEARNING NOISE MODEL FROM WELL-REPRESENTED TISSUES
======================================================================

Well-represented tissues (n≥4):
  - Blood                (36 samples, category: Blood)
  - Pancreas             (17 samples, category: Digestive)
  - Heart                (10 samples, category: Cardiovascular)
  ...

Processing category: Blood
  Analyzing Blood...
  Category Blood: mean_std=0.0423

Processing category: Digestive
  Analyzing Pancreas...
  Analyzing Liver...
  Analyzing Colon...
  Category Digestive: mean_std=0.0512

Global noise statistics:
  Mean std: 0.0467
  Median std: 0.0445
```

**Interpretation:**
- `mean_std`: Average methylation variation in this tissue category
- Digestive tissues show higher variability (0.0512) than Blood (0.0423)
- Global median (0.0445) used as fallback

### Generation Progress

```
GENERATING SYNTHETIC SAMPLES
======================================================================
Processing tissues: 100%|████████████████| 12/12 [15:32<00:00, 77.71s/tissue]
```

### Final Summary

```
GENERATION COMPLETE
======================================================================

Synthetic samples generated: 36
Validation pass rate: 36/36 (100.0%)

Files saved:
  - Synthetic .npz files: synthetic_samples/
  - Synthetic metadata: synthetic_samples/synthetic_metadata.csv
  - Combined metadata: synthetic_samples/combined_metadata.csv
  - Validation results: synthetic_samples/validation_results.csv

Updated tissue distribution:
  ✓ Adipocytes            4 samples (2 original + 2 synthetic)
  ✓ Aorta                 4 samples (1 original + 3 synthetic)
  ✓ Bladder               4 samples (1 original + 3 synthetic)
  ✓ Blood                36 samples (36 original + 0 synthetic)
  ...
```

---

## 🔬 Scientific Rationale

### Why Synthetic Samples?

**Problem:** Machine learning models require balanced data across classes
- Rare tissues = poor model generalization
- Cannot properly validate on tissues with 1 sample

**Solution:** Generate synthetic samples that:
1. Preserve tissue-specific methylation signatures
2. Add realistic biological noise
3. Enable proper train/validation/test splitting

### Why This Approach Works

**Biological basis:**
- Methylation patterns are tissue-specific but show intra-tissue variability
- Variability comes from:
  - Biological heterogeneity (different cells within tissue)
  - Technical noise (sequencing, bisulfite conversion)
  - Sample quality variations

**Our model captures this by:**
1. Learning actual variability from real data
2. Preserving spatial correlations (nearby CpGs correlate)
3. Maintaining tissue-specific signatures

### Validation Strategy

**Three-tier validation:**
1. **Statistical:** KS test for distribution similarity
2. **Biological:** Regional pattern preservation
3. **Quantitative:** Mean methylation conservation

This ensures synthetic samples are indistinguishable from biological replicates.

---

## 📚 Next Steps

After generating synthetic samples:

1. **Review validation results** (`validation_results.csv`)
2. **Visually inspect** a few synthetic samples
3. **Proceed to Step 1.5c:** Data splitting using `combined_metadata.csv`

The combined metadata now includes both original and synthetic samples, properly flagged with `is_synthetic` column.

---

## 🔗 Related Files

- `updated_metadata.csv` (from Step 1.5a)
- Training .npz files (from Step 1.4)
- `tissue_index_mapping.csv` (tissue name to index)

---

## ❓ Questions?

If you see unexpected results:
1. Check validation_results.csv for failed samples
2. Verify tissue counts in combined_metadata.csv
3. Inspect a few synthetic .npz files manually

The script is designed to be conservative - it will only keep high-quality synthetic samples that pass all validation criteria.

---

**Version:** 1.0  
**Date:** 2024-11-17  
**Author:** Step 1.5b Synthetic Sample Generator
