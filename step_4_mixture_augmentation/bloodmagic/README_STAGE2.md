# Stage 2: Blood-Masked Tissue Deconvolution Training

## Overview

Stage 2 implements a **two-stage deconvolution strategy** to overcome blood signal dominance in cfDNA:

### Problem
- cfDNA from cancer patients is 60-100% blood-derived
- Blood signals mask trace tissue signals (pancreas, liver, lung)
- Phase 3 model struggles to detect low-abundance tissues

### Solution: Two-Stage Deconvolution

**Stage 1 (Phase 3):** Predict ALL 22 tissues including blood
- Output: Blood proportion (e.g., 85%)

**Stage 2 (This):** Predict ONLY 21 non-blood tissues
- Input: Same methylation data (WITH blood signal)
- Output: Non-blood tissue proportions (renormalized)
- Training: Model learns to "ignore" blood and focus on trace signals

**Final Inference:**
```python
# Stage 1: Get blood proportion
blood_prop = model_phase3.predict(cfDNA)['Blood']  # 0.85

# Stage 2: Get non-blood proportions
tissue_props = model_stage2.predict(cfDNA)  # [Liver:0.5, Pancreas:0.3, Lung:0.2]

# Combine: Scale non-blood by (1 - blood_prop)
final = {
    'Blood': 0.85,
    'Liver': 0.5 * 0.15 = 0.075,
    'Pancreas': 0.3 * 0.15 = 0.045,
    'Lung': 0.2 * 0.15 = 0.030
}
```

---

## Files Included

### Core Scripts
1. **`dataloader_mixture_stage2.py`**
   - Blood-masked mixture generator
   - Generates mixtures with blood in input, excluded from labels

2. **`generate_stage2_mixtures.py`**
   - Creates validation/test datasets
   - Generates 1500 + 1500 blood-masked mixtures

3. **`train_stage2_bloodmasked.py`**
   - Training script for Stage 2
   - Loads Phase 3 checkpoint, adapts for 21-class output

### Configuration
4. **`config_stage2_bloodmasked.yaml`**
   - Stage 2 training parameters
   - num_classes: 21 (Blood removed)
   - Fine-tuning from Phase 3

### SLURM Scripts
5. **`prepare_stage2_datasets.sh`**
   - Generates validation/test mixtures
   - CPU job, ~2-3 hours

6. **`submit_stage2.sh`**
   - Main training job
   - GPU job, ~16-20 hours

---

## Usage Instructions

### Step 1: Prepare Datasets (CPU Job)

First, generate blood-masked validation and test datasets:

```bash
# Submit dataset generation job
sbatch prepare_stage2_datasets.sh

# Monitor progress
tail -f /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/logs/stage2_prep_*.out

# Verify datasets were created
ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/stage2_*.h5
```

**Expected output:**
```
stage2_validation_mixtures.h5  (~300 MB)
stage2_test_mixtures.h5        (~300 MB)
```

---

### Step 2: Train Stage 2 Model (GPU Job)

Once datasets are ready, launch training:

```bash
# Submit training job
sbatch submit_stage2.sh

# Monitor training
tail -f /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked/logs/slurm_*.out

# Check TensorBoard (optional)
tensorboard --logdir=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked/logs
```

**Training time:** ~16-20 hours on A100

---

## Model Architecture

### Input
- **Shape:** [batch, 51089, 150]
- **Content:** Mixed methylation WITH blood signal (60-100% blood)

### Output
- **Shape:** [batch, 21]
- **Content:** Non-blood tissue proportions (Blood excluded)
- **Normalization:** Softmax → sum to 1.0

### Architecture Changes from Phase 3
```
Phase 3 Model:
  Input: [51089, 150]
  Encoder: [51089] → [512] (region means → hidden)
  MLP: [512] → [1024] → [512]
  Classifier: [512] → [22] → Softmax
  
Stage 2 Model:
  Input: [51089, 150]  (same)
  Encoder: [51089] → [512]  (loaded from Phase 3)
  MLP: [512] → [1024] → [512]  (loaded from Phase 3)
  Classifier: [512] → [21] → Softmax  (REINITIALIZED)
```

### Transfer Learning Strategy
1. Load Phase 3 checkpoint
2. Copy encoder + MLP weights
3. Reinitialize classifier for 21 classes
4. Fine-tune entire model with lower LR

---

## Training Details

### Hyperparameters
- **Epochs:** 25 (fewer than Phase 3 due to fine-tuning)
- **Batch size:** 4 (effective 32 with gradient accumulation)
- **Learning rate:** 1e-6 (10x lower than Phase 3)
- **Mixtures per epoch:** 7500 (same as Phase 3)
- **Pure sample ratio:** 0.1 (10% pure non-blood samples)

### Loss Function
- MSE between predicted and true proportions
- Weighted equally across all 21 tissues

### Expected Performance
- **Validation MAE:** <0.05 for major tissues (>10%)
- **Validation MAE:** <0.10 for minor tissues (<10%)
- **Convergence:** ~15-20 epochs

---

## Output Files

After training completes:

```
/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked/
├── checkpoints/
│   ├── checkpoint_best.pt      # Best validation loss
│   ├── checkpoint_last.pt      # Last epoch
│   └── checkpoint_epoch_*.pt   # Periodic saves
├── logs/
│   ├── slurm_*.out             # Training logs
│   ├── training.log            # Detailed logs
│   └── tensorboard/            # TensorBoard events
└── results/
    ├── validation_predictions.csv
    ├── test_predictions.csv
    └── per_tissue_performance.json
```

---

## Two-Stage Inference Pipeline

### Example Code

```python
import torch
from model_deconvolution import TissueBERTDeconvolution

# Load models
model_phase3 = TissueBERTDeconvolution(num_classes=22)
model_phase3.load_state_dict(torch.load('phase3_best.pt')['model_state_dict'])

model_stage2 = TissueBERTDeconvolution(num_classes=21)
model_stage2.load_state_dict(torch.load('stage2_best.pt')['model_state_dict'])

# Inference on cfDNA sample
methylation = load_cfdna_sample()  # [51089, 150]

# Stage 1: Get blood proportion
with torch.no_grad():
    pred_phase3 = model_phase3(methylation.unsqueeze(0))  # [1, 22]
    blood_idx = 3  # 'Blood' index
    blood_prop = pred_phase3[0, blood_idx].item()

# Stage 2: Get non-blood proportions
with torch.no_grad():
    pred_stage2 = model_stage2(methylation.unsqueeze(0))  # [1, 21]
    tissue_props = pred_stage2[0].cpu().numpy()

# Combine results
tissue_names_nonblood = [t for t in all_tissues if t != 'Blood']
final_proportions = {'Blood': blood_prop}

for i, tissue in enumerate(tissue_names_nonblood):
    final_proportions[tissue] = tissue_props[i] * (1 - blood_prop)

# Verify sum = 1.0
assert abs(sum(final_proportions.values()) - 1.0) < 1e-5

print(final_proportions)
```

---

## Troubleshooting

### Issue: Datasets not found
**Error:** `Stage 2 mixture datasets not found`
**Solution:** Run `sbatch prepare_stage2_datasets.sh` first

### Issue: Phase 3 checkpoint not found
**Error:** `Phase 3 checkpoint not found`
**Solution:** Complete Phase 3 training before Stage 2

### Issue: Out of memory during training
**Error:** CUDA OOM
**Solution:** Reduce batch_size from 4 to 2 in config

### Issue: Model not learning (loss plateau)
**Check:**
1. Is learning rate too low? Try 2e-6
2. Is warmup too long? Reduce warmup_ratio to 0.05
3. Check validation loss trend in TensorBoard

---

## Expected Results

### Training Convergence
```
Epoch  Train Loss  Val Loss  Val MAE
  1      0.089      0.092     0.145
  5      0.042      0.048     0.089
 10      0.028      0.036     0.067
 15      0.021      0.031     0.058
 20      0.018      0.029     0.054
 25      0.017      0.028     0.052
```

### Per-Tissue Performance (Expected)
| Tissue      | Avg Proportion | MAE   |
|-------------|---------------|-------|
| Liver       | 0.15          | 0.042 |
| Pancreas    | 0.12          | 0.048 |
| Lung        | 0.10          | 0.051 |
| Colon       | 0.08          | 0.058 |
| Others      | <0.05         | 0.065 |

---

## Next Steps After Training

1. **Evaluate on test set:**
   ```bash
   python evaluate_stage2.py --checkpoint checkpoint_best.pt
   ```

2. **Apply to PDAC samples:**
   - Use two-stage inference pipeline
   - Track tissue proportions over time
   - Detect metastasis-related signals

3. **Compare Stage 1 vs Stage 2:**
   - Does Stage 2 improve low-abundance tissue detection?
   - Visualize tissue proportion changes

---

## Citation

If you use this code, please cite:

```
Stage 2 Blood-Masked Tissue Deconvolution
PDAC cfDNA Monitoring Project, 2024
```

---

## Contact

For questions or issues:
- Check logs in `/logs/` directory
- Review TensorBoard for training curves
- Ensure Phase 3 training completed successfully
