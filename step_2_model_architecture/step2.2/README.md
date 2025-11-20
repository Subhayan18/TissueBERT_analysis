# Methylation Data Processing Pipeline

Three scripts for converting NPZ methylation data to HDF5 and loading it efficiently for training.

---

## 📁 Scripts Overview

### 1. `convert_to_hdf5.py`
Converts NPZ training files to a single HDF5 file with 22 tissue classes.

**Key Features:**
- Validates all NPZ files before conversion
- Uses tissue_index from CSV metadata (22 classes: 0-21)
- Creates efficient HDF5 with gzip compression
- Automatically maps tissue names to indices if needed

**Usage:**
```bash
python convert_to_hdf5.py
```

**Output:**
- Creates `methylation_dataset.h5` (~2.4 GB)
- Contains: DNA tokens, methylation status, read counts, tissue labels
- 22 tissue types (0-21) present in all splits

**Run this:** Once, when you first set up your data or if you need to regenerate the HDF5 file.

---

### 2. `check_hdf5.py`
Quick diagnostic to verify HDF5 file contents and tissue label setup.

**Checks:**
- Number of tissue classes (should be 22, not 119)
- Tissue label range (should be 0-21)
- File size and structure
- Whether you're using the correct version

**Usage:**
```bash
python check_hdf5.py
```

**Output:**
```
✅ CORRECT: Using 22-class system (0-21)
```
or
```
❌ WRONG: Using 119-class system (0-118)
   Run: python convert_to_hdf5.py
```

**Run this:** Anytime you want to verify your HDF5 file is correct.

---

### 3. `dataset_dataloader.py`
PyTorch DataLoader with tissue-balanced sampling for training.

**Key Features:**
- Efficient HDF5 random access
- Tissue-balanced sampling (avoids 2^24 limit with custom batch sampler)
- CpG dropout augmentation for training
- Supports train/validation/test splits

**Usage:**

**As a script (for testing):**
```bash
python dataset_dataloader.py
```
Tests dataloaders with 10 batches and shows tissue distribution.

**In your training code:**
```python
from dataset_dataloader import create_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    hdf5_path='path/to/methylation_dataset.h5',
    train_csv='train_files.csv',
    val_csv='val_files.csv',
    test_csv='test_files.csv',
    batch_size=32,
    num_workers=4,
    cpg_dropout_rate=0.05,      # Augmentation: randomly mask 5% of CpGs
    tissue_balanced=True,        # Balance tissues in training batches
    pin_memory=True
)

# Use in training
for batch in train_loader:
    dna_tokens = batch['dna_tokens']       # Shape: [batch_size, 150]
    methylation = batch['methylation']     # Shape: [batch_size, 150]
    n_reads = batch['n_reads']             # Shape: [batch_size]
    tissue_labels = batch['tissue_label']  # Shape: [batch_size], values 0-21
    
    # Your training code here
```

**Run this:** Import in your training script, or run standalone to test.

---

## 🚀 Quick Start

### First-Time Setup
```bash
# 1. Convert NPZ files to HDF5
python convert_to_hdf5.py

# 2. Verify it worked
python check_hdf5.py

# 3. Test the dataloader
python dataset_dataloader.py
```

### Expected Output (Step 3)
```
Creating DataLoaders
...
Testing Training DataLoader...
   Testing 10 batches...
   
   Tissue distribution (top 10):
     Tissue 1: 23 samples
     Tissue 15: 21 samples
     Tissue 2: 21 samples
     ...
     Tissue 0: 15 samples  ✓ Good distribution!
     
✓ All dataloaders tested successfully!
```

---

## 📊 Data Format

### Input (NPZ files)
- `dna_tokens`: [51089, 150] uint8 - DNA sequence as tokens
- `methylation`: [51089, 150] uint8 - Methylation status (0=unmethylated, 1=methylated, 2=no CpG)
- `n_reads`: [51089] int32 - Read coverage per region
- `region_ids`: [51089] - Region identifiers

### Output (HDF5)
- **Dimensions:** [765 files, 51089 regions, 150 bp]
- **Tissue labels:** 22 classes (0-21)
- **Total size:** ~2.4 GB (compressed)

### CSV Metadata Required
- `train_files.csv`, `val_files.csv`, `test_files.csv`
- Must contain columns: `filename`, `tissue_index`

---

## 🔧 Configuration

### File Paths (in convert_to_hdf5.py)
```python
NPZ_DATA_DIR = Path(".../training_dataset/all_data")
METADATA_FILE = Path(".../combined_metadata.csv")
TRAIN_CSV = Path(".../train_files.csv")
VAL_CSV = Path(".../val_files.csv")
TEST_CSV = Path(".../test_files.csv")
OUTPUT_HDF5 = Path(".../methylation_dataset.h5")
```

### Dataloader Parameters
```python
batch_size=32              # Samples per batch
num_workers=4              # Parallel data loading threads
cpg_dropout_rate=0.05      # Training augmentation (5% CpG dropout)
tissue_balanced=True       # Balance tissue representation in batches
pin_memory=True            # Faster GPU transfer (if using GPU)
```

---

## ⚠️ Important Notes

### Tissue Balancing
- Uses **file-level sampling** (samples from 455 files, not 23M regions)
- Avoids PyTorch's 2^24 category limit
- Ensures each tissue appears equally often in training batches

### CpG Dropout Augmentation
- Only applied during training (not val/test)
- Randomly converts methylated/unmethylated CpGs to "missing" (value 2)
- Helps model handle variable coverage

### Missing CpG Rate (~97%)
- This is **normal and expected**
- CpGs are sparse in the genome (~3% of positions)
- The other 97% of positions don't have CpGs (value=2)

---

## 🐛 Troubleshooting

### "number of categories cannot exceed 2^24"
- **Cause:** Old version of dataloader with region-level sampling
- **Fix:** Use the provided `dataset_dataloader.py` (uses file-level sampling)

### "Tissue index out of range"
- **Cause:** HDF5 has 119 classes but model expects 22
- **Fix:** Run `python check_hdf5.py` then regenerate if needed

### All validation samples are tissue 0
- **Cause:** Small sample size (only testing 320 out of 6.9M regions)
- **Fix:** Not a bug - this is normal when testing with small batches

### Model expects 119 classes
- **Fix:** Update your model output layer to 22 classes:
  ```python
  self.classifier = nn.Linear(hidden_dim, 22)  # Not 119!
  ```

---

## 📈 Performance Tips

- **num_workers:** Set to 4-8 for faster loading (or number of CPU cores)
- **pin_memory:** Set to True when using GPU
- **batch_size:** Increase if you have enough GPU memory (try 64, 128)
- **persistent_workers:** Already enabled for faster epoch transitions

---

## 🔍 Validation

The dataloader includes built-in validation:
```python
from dataset_dataloader import validate_dataloaders

validate_dataloaders(train_loader, val_loader, test_loader, n_samples=10)
```

Checks:
- Batch shapes are correct
- No NaN or Inf values
- Tissue distribution across batches
- Data loading speed

---

## 📝 Summary

1. **convert_to_hdf5.py** → Convert NPZ to HDF5 (run once)
2. **check_hdf5.py** → Verify HDF5 is correct (run to check)
3. **dataset_dataloader.py** → Load data for training (import in your code)

All three scripts work together to provide efficient, tissue-balanced training data.

---

## Questions?

- Check that your CSV files have `tissue_index` column (0-21)
- Verify HDF5 file is using 22 classes with `check_hdf5.py`
- Test dataloaders with `python dataset_dataloader.py`
- Ensure your model expects 22 output classes
