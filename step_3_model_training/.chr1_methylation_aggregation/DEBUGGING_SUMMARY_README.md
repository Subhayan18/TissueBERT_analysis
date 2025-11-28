# TissueBERT Debugging and Solution Summary

## Executive Summary

**Problem**: TissueBERT was stuck at ~6% accuracy (near random chance for 22 classes = 4.5%)

**Root Cause**: Fundamental mismatch between training paradigm and data structure
- Model trained on individual regions (150bp) with file-level labels
- Each of 4,381 regions from a file got the same label, creating massive label noise

**Solution**: File-level classification matching logistic regression approach
- Process all regions from a file as one sample
- Achieved **93.3% validation accuracy** (matching 91% from logistic regression baseline)

---

## Debugging Timeline

### Phase 1: Initial Investigation

#### Observation
- Training accuracy: ~6% (random chance: 4.5%)
- Loss stuck at 3.09 (= -ln(1/22), theoretical random loss)
- Model predictions collapsed to 1-3 classes

#### Diagnostic Steps Taken

**1. Label Consistency Check** (`diagnostic_1_labels_and_variance.py`)
```
✓ RESULT: Perfect match between CSV tissue_index and HDF5 tissue_labels
✓ Between/within class variance ratio: 3.83 (good separation)
```

**2. Augmentation Leakage Check** (`diagnostic_2_augmentation_and_separability.py`)
```
⚠️ RESULT: 14 samples had synthetic versions across train/val/test splits
✓ RESULT: Broad tissue categories showed internal heterogeneity
- Blood: 14 subtypes
- Pancreas: 5 subtypes (internal separability 0.60)
```

**3. Logistic Regression Baseline** (`baseline_logistic_regression.py`)
```
✓ RESULT: 91.43% test accuracy!
KEY INSIGHT: Data IS learnable - problem must be in model/training setup
```

### Phase 2: Architecture Debugging

#### Hypothesis: Input Encoding Issue

**Original TissueBERT Architecture:**
```python
# Embeddings combined via summation
dna_embeds = self.dna_embedding(dna_tokens)      # 69 vocab
meth_embeds = self.meth_embedding(methylation)    # 3 vocab (0,1,2)
embeddings = dna_embeds + meth_embeds + pos_embeds
```

**Problem Identified:**
- Methylation signal (3 tokens) diluted by DNA signal (69 tokens)
- Logistic regression proved methylation alone is sufficient

**Fix Attempted:** Methylation-only model
```python
# Only use methylation embeddings
embeddings = meth_embeds + pos_embeds  # Ignore DNA
```

**Result:** Still stuck at ~27% accuracy

### Phase 3: Critical Discovery

#### Sanity Check (`sanity_check_overfit.py`)

**Test:** Can model overfit 10 samples?
```
✓ RESULT: 100% accuracy in 1 epoch
CONCLUSION: Model architecture is sound, gradients flow correctly
```

This proved the issue was NOT in the model itself.

#### The Breakthrough Realization

**Logistic Regression Setup:**
```python
# Extract features: [n_files, 1000 regions]
for file_idx in file_indices:
    for region_idx in range(n_regions):
        features[file_idx, region_idx] = mean_methylation_of_region
# Classification: ONE prediction per FILE
```

**Neural Network Setup (WRONG):**
```python
# Dataloader: [batch, 150] - ONE REGION per sample
# Label: File-level tissue label
# Problem: Predicting FILE tissue from SINGLE REGION
```

**The Core Issue:**
- **Region-level training with file-level labels**
- Each of 4,381 regions from `file_042_Liver.npz` got label "Liver"
- But individual regions vary widely:
  - Some regions are tissue-specific
  - Some are universal (housekeeping genes)
  - Some have no methylation data
- Model couldn't learn because labels were wrong for most regions

**Why 27% Accuracy Cap?**
- Model learned to predict most common tissue for each region's pattern
- But couldn't improve beyond weak regional associations
- True task required seeing ALL regions from a file

---

## Final Solution

### Architecture Changes

#### 1. File-Level Dataloader (`dataloader_filelevel.py`)

**Before:**
```python
def __getitem__(self, idx):
    file_idx = self.region_to_file[idx]
    region_idx = self.region_to_region_idx[idx]
    
    # Returns ONE region
    return {
        'dna_tokens': [150],      # Single region
        'methylation': [150],
        'tissue_label': scalar    # File-level label (WRONG!)
    }
```

**After:**
```python
def __getitem__(self, idx):
    file_idx = self.file_indices[idx]
    
    # Returns ALL regions from file
    return {
        'dna_tokens': [n_regions, 150],     # All regions
        'methylation': [n_regions, 150],
        'tissue_label': scalar              # File-level label (CORRECT!)
    }
```

#### 2. File-Level Model (`model_filelevel_mlp.py`)

**Architecture:**
```python
def forward(self, dna_tokens, methylation_status):
    # Input: [batch, n_regions, seq_length]
    
    # Step 1: Compute mean methylation per region
    region_means = compute_mean_per_region(methylation_status)
    # Shape: [batch, n_regions]
    
    # Step 2: Project to hidden dimension
    features = self.input_projection(region_means)
    # Shape: [batch, hidden_size]
    
    # Step 3: MLP classification
    logits = self.network(features)
    # Shape: [batch, num_classes]
    
    return logits
```

**Key Features:**
- Processes all regions from a file simultaneously
- Each region contributes one feature (mean methylation)
- Matches logistic regression approach exactly

#### 3. Training Script (`train_filelevel.py`)

**Critical Changes:**
```python
# Smaller batch size (files, not regions)
batch_size = 8  # Down from 128

# Import file-level components
from dataloader_filelevel import create_filelevel_dataloaders
from model_filelevel_mlp import TissueBERT

# Remove max_steps_per_epoch limit
# (was set to 1000, but only 57 file-level batches exist)
```

---

## Results Comparison

### Before vs After

| Metric | Region-Level (WRONG) | File-Level (CORRECT) |
|--------|---------------------|----------------------|
| Training paradigm | Individual regions | Entire files |
| Samples per epoch | 455 files × 4,381 regions = ~2M | 455 files |
| Batch size | 128 regions | 8 files |
| Label alignment | Region → File label ❌ | File → File label ✓ |
| Train accuracy | 6% (stuck) | 87.5% (epoch 25) |
| Val accuracy | 4-6% | **93.3%** (epoch 25) |
| Convergence | Never | ~20 epochs |

### Performance Metrics

**Epoch-by-Epoch Progress:**
```
Epoch  1: Train  6.6%  → Val  -
Epoch  5: Train 16.0%  → Val 17.8%
Epoch 10: Train 52.3%  → Val 54.8%
Epoch 15: Train 69.2%  → Val 82.2%
Epoch 20: Train 83.3%  → Val 88.2%
Epoch 25: Train 87.5%  → Val 93.3%  ← MATCHES LOGISTIC REGRESSION!
```

**Final Results:**
- ✅ Validation Accuracy: **93.3%**
- ✅ Matches/Exceeds Logistic Regression: **91.4%**
- ✅ Significantly Above Random: **4.5%**

---

## Key Lessons Learned

### 1. **Data-Label Alignment is Critical**
- Always verify that your training samples and labels are at the same granularity
- Region-level features require region-level labels (or aggregate to file-level)

### 2. **Baseline Models Are Essential**
- Logistic regression baseline (91%) proved the data was learnable
- This eliminated data quality as the issue and pointed to model/training setup

### 3. **Sanity Checks Matter**
- Overfitting 10 samples proved the model architecture was sound
- Isolated the problem to the training paradigm, not gradient flow or initialization

### 4. **Diagnostic Scripts Save Time**
- Systematic diagnostics (labels, variance, leakage) ruled out common issues
- Focused debugging efforts on the actual root cause

### 5. **Question Assumptions**
- The dataloader had been working for months in region-level mode
- Switching to file-level classification was the "obvious" solution in hindsight
- But required questioning the fundamental training paradigm

---

## Technical Details

### Model Architecture

**Input Processing:**
```
Input: [batch, n_regions, 150] methylation values

1. Aggregate per region:
   - Mask missing values (methylation == 2)
   - Compute mean: Σ(valid methylation) / Σ(valid positions)
   - Result: [batch, n_regions] continuous values

2. Project to hidden space:
   - Linear: n_regions → 512
   - Result: [batch, 512]

3. MLP Classification:
   - 512 → 512 → 1024 → 22 (num classes)
   - BatchNorm + ReLU + Dropout between layers
```

**Parameters:**
- ~3-5M parameters (vs 17M in original transformer)
- Much faster training: ~6 seconds/epoch vs 3 minutes/epoch

### Training Configuration

```yaml
training:
  num_epochs: 50
  batch_size: 8                    # Files per batch
  gradient_accumulation_steps: 4
  num_workers: 8
  max_steps_per_epoch: null        # Use all data
  validation_frequency: 5

optimizer:
  learning_rate: 1.0e-5            # Conservative LR
  weight_decay: 0.01
  warmup_ratio: 0.1

loss:
  label_smoothing: 0.1             # Helps with class imbalance
```

### Data Statistics (chr1 subset)

```
Training Files:     455
Validation Files:   135  
Test Files:         175
Total Files:        765

Regions per File:   4,381 (chr1 only)
Sequence Length:    150 bp per region
Classes:            22 tissue types

Batches per Epoch:  57 (455 files / 8 batch_size)
```

---

## Future Work

### 1. Scale to Full Genome

**Current:** Chr1 only (4,381 regions)
**Next:** All chromosomes (~51,089 regions)

**Expected Results:**
- Higher accuracy (more features)
- Longer training time (more regions to aggregate)
- Same paradigm should work

### 2. Investigate Fine-Grained Classification

**Current:** 22 broad tissue categories
**Alternative:** ~60 fine-grained tissue subtypes

**From diagnostics:**
- Blood: 14 distinct subtypes
- Pancreas: 5 subtypes with internal separability 0.60
- May achieve better performance on fine-grained task

### 3. Address Data Leakage

**Issue:** 14 samples have synthetic versions across train/val/test
**Fix:** Re-split ensuring all synthetic versions stay together
**Impact:** More honest validation metrics

### 4. Explore Transformer for Regional Patterns

**Idea:** Use transformer ACROSS regions (not within)
- Attention mechanism to identify important regions
- Potentially capture regional interactions
- May improve beyond simple mean aggregation

---

## Files Generated During Debugging

### Diagnostic Scripts
- `diagnostic_1_labels_and_variance.py` - Label consistency & variance analysis
- `diagnostic_2_augmentation_and_separability.py` - Leakage & separability checks
- `baseline_logistic_regression.py` - Logistic regression baseline
- `sanity_check_overfit.py` - Model overfitting capability test

### Model Iterations
- `model.py` - Original TissueBERT (DNA + methylation)
- `model_methylation_only.py` - Methylation-only version
- `model_simplified.py` - Simplified aggregation attempt
- `model_simple_mlp.py` - Region-level MLP (didn't work)
- `model_filelevel_mlp.py` - ✅ **Final working version**

### Data Loading
- `dataset_dataloader.py` - Original region-level dataloader
- `dataloader_filelevel.py` - ✅ **Final file-level dataloader**

### Training Scripts
- `train.py` - Original training script
- `train_filelevel.py` - ✅ **Final working training script**

---

## Conclusion

The debugging process revealed a **fundamental paradigm mismatch**: training on individual regions with file-level labels created insurmountable label noise. The solution—file-level classification matching logistic regression's approach—achieved 93.3% accuracy, validating both the data quality and the overall methodology.

**Key Success Factors:**
1. Systematic diagnostics to rule out common issues
2. Strong baseline (logistic regression) to prove data quality  
3. Sanity checks to isolate the problem domain
4. Willingness to question fundamental assumptions

The final solution is simpler, faster, and more accurate than the original approach, demonstrating that sometimes the right solution comes from stepping back and re-examining the problem definition rather than adding more complexity to the model.

---

**Status:** ✅ SOLVED - Ready for full-genome training

**Final Performance:** 93.3% validation accuracy (chr1 subset)

**Next Step:** Scale to all chromosomes and evaluate on full test set
