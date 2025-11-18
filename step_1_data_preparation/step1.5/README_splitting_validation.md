# Step 1.5c & 1.5d: Data Splitting and Validation

## Quick Guide

### Step 1.5c: Split Data (Train/Val/Test)
```bash
# Run on login node (~1 minute)
bash run_split_data.sh
```

**Output:**
- `data_splits/train_files.csv` (~440 files, 70%)
- `data_splits/val_files.csv` (~92 files, 15%)
- `data_splits/test_files.csv` (~88 files, 15%)

**What it does:**
- Splits by sample (all aug0-aug4 stay together)
- Stratifies by tissue (proportional representation)
- Handles rare tissues (n<4): 2 train / 0-1 val / 1 test

---

### Step 1.5d: Validate Everything
```bash
# Run on login node (~5 minutes)
bash run_validate_data.sh
```

**Tests performed:**
1. ✓ File existence (all files in metadata exist)
2. ✓ NPZ integrity (files load correctly)
3. ✓ Metadata consistency (no duplicates)
4. ✓ Split integrity (no sample in multiple splits)
5. ✓ Tissue coverage (all tissues in training)
6. ✓ Synthetic quality (validation metrics)
7. ✓ PyTorch compatibility (DataLoader works)

**Success:** All 7 tests pass → Ready for Phase 2!

---

## Expected Results

**After splitting:**
- Train: ~110 samples (~440 files)
- Val: ~23 samples (~92 files)
- Test: ~22 samples (~88 files)

**Validation:**
```
✓✓✓ ALL VALIDATION TESTS PASSED ✓✓✓
🎉 Your data is ready for Phase 2: Model Training!
```

---

## Troubleshooting

**Split ratios not exact 70/15/15?**
- Normal with small sample sizes
- Acceptable: 68-72% / 13-17% / 13-17%

**Validation test fails?**
- Check which test failed
- Most common: File paths in wrapper scripts
- Fix: Edit paths in `run_split_data.sh` and `run_validate_data.sh`

**Some tissues missing from val/test?**
- Expected for tissues with n=3 samples
- They go: 2 train / 0 val / 1 test

---

## Quick Check
```bash
# Count samples per split
wc -l data_splits/*.csv

# Check tissue distribution
python -c "
import pandas as pd
df = pd.read_csv('data_splits/train_files.csv')
print(df.groupby('tissue')['sample_name'].nunique())
"
```

---

**Next:** Use `train_files.csv`, `val_files.csv`, `test_files.csv` for Phase 2 training!
