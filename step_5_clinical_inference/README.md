# PDAC Clinical Prediction Pipeline

Two-step pipeline to convert PDAC beta value data and run tissue deconvolution inference using **HDF5 format**.

## Overview

This pipeline converts PDAC methylation beta values (TSV format) into HDF5 format and runs inference to predict tissue proportions for clinical applications.

**Pipeline Steps:**
1. **Data Conversion**: Convert beta values TSV → HDF5 file (single file with all samples)
2. **Inference**: Run trained model → Tissue proportion predictions

---

## Files

### Main Scripts (Use these with HDF5):
- `convert_pdac_beta_to_hdf5.py` - **[RECOMMENDED]** Convert TSV to HDF5 format  
- `run_pdac_inference_hdf5.py` - **[RECOMMENDED]** Run inference on HDF5 data

### Legacy Scripts (PyTorch .pt files):
- `convert_pdac_beta_to_model_input.py` - Convert TSV to .pt files (one per sample)
- `run_pdac_inference.py` - Run inference on .pt files

### Supporting Files:
- `tissue_names.txt` - Names of 22 tissue types
- `model_deconvolution.py` - Model architecture (required)

---

## Quick Start (Recommended: HDF5 Format)

### Step 1: Convert TSV to HDF5

```bash
python convert_pdac_beta_to_hdf5.py \
  --input_tsv BETA_all.tsv \
  --output_h5 pdac_samples.h5 \
  --simulation_method probabilistic \
  --verbose
```

**Output:**
- `pdac_samples.h5` - HDF5 file with methylation data for ALL samples
- `pdac_samples.csv` - Metadata CSV with sample information
- `region_info.csv` - Genomic region coordinates

### Step 2: Run Inference

```bash
python run_pdac_inference_hdf5.py \
  --input_h5 pdac_samples.h5 \
  --input_metadata pdac_samples.csv \
  --checkpoint /path/to/checkpoint_best.pt \
  --output_dir ./predictions \
  --tissue_names tissue_names.txt \
  --apply_renormalization \
  --batch_size 8 \
  --verbose
```

**Output:**
- `predictions/tissue_proportions_all_samples.csv` - All predictions
- `predictions/clinical_reports/*.txt` - Individual reports per sample
- `predictions/predictions.json` - JSON format for programmatic access
- `predictions/summary_statistics.json` - Summary statistics

---

## Why HDF5 Format?

✅ **Single file** for all samples (easier to manage)  
✅ **Compressed storage** (smaller file size)  
✅ **Fast random access** (efficient batch processing)  
✅ **Same format as training data** (no conversion issues)  
✅ **Scalable** to thousands of samples  

---

## Input Data Format

### TSV File Structure (Input)

Your `BETA_all.tsv` should have:

```
chr    start    end    startCpG    endCpG    Sample1    Sample2    Sample3    ...
chr1   273778   273898   2782       2784      0.51       0.66       0.82
chr1   642988   643382   6127       6130      0.85       0.85       0.81
...
```

- **First 5 columns**: Region coordinates (chr, start, end, startCpG, endCpG)
- **Remaining columns**: Beta values per sample (0-1 or NA)

### HDF5 File Structure (After Conversion)

```
pdac_samples.h5:
  └── methylation: [n_samples, 51089, 150]
      ├── dtype: int8
      ├── values: 0 (unmethylated), 1 (methylated), 2 (missing)
      └── compression: gzip level 4
```

### Metadata CSV (After Conversion)

```csv
sample_idx,sample_name,n_regions,seq_length,methylation_pct,missing_pct
0,L01_191024_S1,51089,150,45.23,8.12
1,L01_191121_S2,51089,150,47.89,6.45
```

---

## Command Line Arguments

### Conversion Script (`convert_pdac_beta_to_hdf5.py`)

```
Required:
  --input_tsv           Input TSV file with beta values
  --output_h5           Output HDF5 file path

Optional:
  --output_metadata     Output metadata CSV (default: same as h5 with .csv)
  --simulation_method   binary|probabilistic|replicate (default: probabilistic)
  --n_simulated_reads   Number of reads for probabilistic (default: 10)
  --seq_length          Sequence length per region (default: 150)
  --n_regions           Expected number of regions (default: 51089)
  --verbose             Print detailed progress
```

### Inference Script (`run_pdac_inference_hdf5.py`)

```
Required:
  --input_h5            HDF5 file with methylation data
  --input_metadata      Metadata CSV file
  --checkpoint          Path to trained model checkpoint
  --output_dir          Output directory for predictions

Optional:
  --tissue_names        File with tissue names (default: generic names)
  --batch_size          Batch size for inference (default: 8)
  --device              cuda|cpu (default: cuda)
  --apply_renormalization   Enable post-processing (recommended)
  --renorm_strategy     threshold|soft_threshold|bayesian (default: threshold)
  --renorm_threshold    Minimum proportion to keep (default: 0.05)
  --min_proportion      Minimum for clinical reports (default: 0.01)
  --verbose             Print detailed progress
```

---

## Simulation Methods

When converting beta values to 150bp read-level data:

### 1. Probabilistic (Recommended)
```bash
--simulation_method probabilistic --n_simulated_reads 10
```
- Simulates multiple reads per region
- Aggregates using majority vote
- Most realistic representation
- **Use this for best results**

### 2. Binary
```bash
--simulation_method binary --beta_to_binary_threshold 0.5
```
- Simple threshold-based conversion
- Fast but less realistic
- Use for quick testing

### 3. Replicate
```bash
--simulation_method replicate
```
- Distributes methylation uniformly
- Intermediate complexity
- Good balance of speed and realism

---

## Model Input/Output

### Input to Model
```
Shape: [batch_size, 51089, 150]
Values: 0 (unmeth), 1 (meth), 2 (missing)
```

### Model Processing
1. Compute mean methylation per region → `[batch, 51089]`
2. Linear projection → `[batch, 512]`
3. MLP network → `[batch, 22]`
4. Sigmoid + L1 normalize → `[batch, 22]` proportions (sum=1.0)

### Output Format
```csv
sample_name,Adipocytes,Aorta,Bladder,Blood,Bone,...
L01_S1,0.0123,0.0045,0.0012,0.7834,0.0091,...
```

---

## Clinical Interpretation

### Expected Tissue Proportions

**Healthy cfDNA:**
- Blood cells: 70-90%
- Organ tissues: <5% each

**PDAC with Metastasis:**
- Elevated organ-specific cfDNA indicates tissue damage
- Threshold: >2x baseline or >10% absolute
- Example: Liver 12% → possible liver micrometastasis

### Clinical Report Example

```
======================================================================
TISSUE DECONVOLUTION REPORT
======================================================================
Sample ID: L01_191024_S1
Analysis Date: 2026-01-08 14:23:45

Sample Quality Metrics:
  Methylation rate: 91.88%
  Missing data: 8.12%

Predicted Tissue Proportions:
----------------------------------------------------------------------
Tissue                         Proportion      Percentage
----------------------------------------------------------------------
Blood                          0.7834           78.34%
Liver                          0.1234           12.34%
Lung                           0.0512            5.12%
----------------------------------------------------------------------

Clinical Alerts:
----------------------------------------------------------------------
  ⚠️  Elevated Liver: 12.34%
      → Consider enhanced surveillance for liver metastasis
======================================================================
```

---

## Tissue Types (22 Total)

1. Adipocytes
2. Aorta
3. Bladder
4. Blood
5. Bone
6. Cerebellum
7. Colon
8. Coronary
9. Cortex
10. Dermal
11. Epidermal
12. Gastric
13. Heart
14. Kidney
15. Liver
16. Lung
17. Neuron
18. Oligodendrocytes
19. Pancreas
20. Prostate
21. Skeletal
22. Small

---

## Complete Example Workflow

```bash
# 1. Convert TSV to HDF5
python convert_pdac_beta_to_hdf5.py \
  --input_tsv /path/to/BETA_all.tsv \
  --output_h5 pdac_samples.h5 \
  --simulation_method probabilistic \
  --n_simulated_reads 10 \
  --verbose

# 2. Run inference
python run_pdac_inference_hdf5.py \
  --input_h5 pdac_samples.h5 \
  --input_metadata pdac_samples.csv \
  --checkpoint /path/to/checkpoint_best.pt \
  --output_dir ./predictions \
  --tissue_names tissue_names.txt \
  --apply_renormalization \
  --renorm_strategy threshold \
  --renorm_threshold 0.05 \
  --batch_size 8 \
  --device cuda \
  --verbose

# 3. View results
cat predictions/clinical_reports/L01_191024_S1_report.txt
head predictions/tissue_proportions_all_samples.csv
```

---

## Troubleshooting

### Region count mismatch
**Error**: `Expected 51089 regions, but TSV has 45942`  
**Solution**: Script automatically adjusts to actual number

### CUDA out of memory
**Error**: `RuntimeError: CUDA out of memory`  
**Solutions**:
- Reduce batch size: `--batch_size 4` or `--batch_size 1`
- Use CPU: `--device cpu` (slower)

### HDF5 compression
**Info**: File sizes are ~20-50% of raw data due to gzip compression
