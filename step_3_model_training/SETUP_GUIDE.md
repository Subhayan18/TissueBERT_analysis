# Step 2.3 Complete: Training Setup Summary

## ✅ What We've Created

Complete training infrastructure for TissueBERT tissue classification model.

---

## 📦 Deliverables

### 1. Core Training Scripts (6 files)

| File | Lines | Purpose |
|------|-------|---------|
| `train.py` | ~700 | Main training loop with checkpointing, logging, metrics |
| `model.py` | ~400 | DNABERT-S transformer architecture (6 layers, 512 hidden) |
| `utils.py` | ~550 | Metrics, visualization, checkpointing utilities |
| `submit_training.sh` | ~120 | SLURM job submission with auto-resume |
| `analyze_results.py` | ~350 | Comprehensive results analysis |
| `README_TRAINING.md` | ~450 | Complete documentation |

### 2. Configuration Files (3 files)

- `config_20epoch.yaml` - Initial 20 epoch run (~25-40 hours)
- `config_50epoch.yaml` - Medium 50 epoch run (~60-100 hours)  
- `config_100epoch.yaml` - Full 100 epoch run (~120-200 hours)

---

## 🚀 Installation Instructions

### Step 1: Copy Files to HPC

```bash
# SSH to your HPC cluster
ssh user@hpc-cluster

# Navigate to training directory
cd /home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture

# Create step_3_model_training directory
mkdir -p step_3_model_training
cd step_3_model_training

# Copy all files from outputs (adjust path as needed)
cp /path/to/outputs/step_3_training_setup/* .

# Make scripts executable
chmod +x submit_training.sh
chmod +x analyze_results.py
```

### Step 2: Verify Setup

```bash
# Test model architecture
python model.py
# Expected output: "✓ Model test passed!"

# Check data paths in config
cat config_20epoch.yaml | grep -A 4 "data:"
# Verify HDF5 path is correct

# Test module loading
source /home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/LMOD.sourceme
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Update Email in SLURM Script (Optional)

```bash
# Edit submit_training.sh
nano submit_training.sh

# Change this line:
#SBATCH --mail-user=your_email@example.com
# To:
#SBATCH --mail-user=your.actual.email@domain.com

# Save and exit (Ctrl+X, Y, Enter)
```

### Step 4: Create Output Directories

```bash
# These will be created automatically, but you can pre-create them
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/{checkpoints,logs,results}
```

---

## 🎯 Quick Start: Launch Training

### Option A: Start with 20 Epochs (Recommended)

```bash
# Submit job
sbatch submit_training.sh config_20epoch.yaml

# Check job status
squeue -u $USER

# Monitor progress (wait a few seconds after submission)
tail -f /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs/slurm_*.out
```

### Option B: Start with 50 Epochs

```bash
sbatch submit_training.sh config_50epoch.yaml
```

### Option C: Full 100 Epoch Training

```bash
sbatch submit_training.sh config_100epoch.yaml
```

---

## 📊 What to Expect

### Timeline (20 Epochs)

```
Epoch 1-5:   ~30-45 min each    → 2-4 hours total
  - Loss: 3.0 → 1.5
  - Accuracy: 10% → 50%

Epoch 5-10:  ~30-45 min each    → 2-4 hours total
  - Loss: 1.5 → 1.0
  - Accuracy: 50% → 65%

Epoch 10-20: ~30-45 min each    → 5-8 hours total
  - Loss: 1.0 → 0.6
  - Accuracy: 65% → 75-80%

Total: ~10-16 hours for 20 epochs
```

### Success Criteria (20 Epochs)

- ✓ **Validation accuracy > 75%**
- ✓ **Train-val gap < 10%**
- ✓ **All 22 tissues learning** (F1 > 0.5)
- ✓ **No NaN or Inf values**
- ✓ **GPU memory < 30 GB**

### When to Continue to 50 Epochs

✅ If after 20 epochs:
- Validation accuracy > 75%
- Validation loss still decreasing
- No severe overfitting (train-val gap < 15%)
- Per-tissue metrics improving

---

## 📈 Monitoring Training

### Real-Time Monitoring

```bash
# Watch SLURM output
tail -f /path/to/slurm_*.out

# Watch training metrics (after first epoch)
watch -n 10 "tail -5 /path/to/logs/training_log.csv"

# Check GPU usage
watch -n 5 nvidia-smi
```

### TensorBoard (Optional)

```bash
# On HPC, start tensorboard
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs
tensorboard --logdir tensorboard --port 6006 --host 0.0.0.0

# On local machine, create SSH tunnel
ssh -L 6006:localhost:6006 user@hpc-cluster

# Open browser
http://localhost:6006
```

### Check Progress Anytime

```bash
# View last 10 epochs
tail -10 /path/to/logs/training_log.csv | column -t -s,

# Check best metrics so far
python -c "
import pandas as pd
df = pd.read_csv('/path/to/logs/training_log.csv')
print(f'Best val loss: {df[\"val_loss\"].min():.4f} (epoch {df.loc[df[\"val_loss\"].idxmin(), \"epoch\"]})')
print(f'Best val acc: {df[\"val_acc\"].max():.4f} (epoch {df.loc[df[\"val_acc\"].idxmax(), \"epoch\"]})')
"
```

---

## 🔍 After Training Completes

### Step 1: Analyze Results

```bash
cd /home/chattopa/data_storage/TissueBERT_analysis/step_3_model_training

python analyze_results.py \
    --log_dir /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs \
    --results_dir /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/results \
    --output analysis_report_20epoch.txt

# View report
cat analysis_report_20epoch.txt
```

### Step 2: Check Output Files

```bash
# View all outputs
ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/

# Checkpoints
ls -lh checkpoints/
# Expected: checkpoint_last.pt, checkpoint_best_loss.pt, checkpoint_best_acc.pt, checkpoint_epoch_*.pt

# Logs
ls -lh logs/
# Expected: training_log.csv, per_tissue_metrics.csv, training_summary.txt, slurm_*.out/err

# Results
ls -lh results/
# Expected: final_results.json, training_curves.png, confusion_test_final.png
```

### Step 3: Decide Next Steps

**If validation accuracy ≥ 80%:**
- ✅ Great! Continue to 50 epochs
- ✅ Start working on inference pipeline
- ✅ Test on validation samples

**If validation accuracy 75-80%:**
- ⚠️  Check per-tissue performance
- ⚠️  Continue to 50 epochs if still improving
- ⚠️  Analyze confusion matrix for problem tissues

**If validation accuracy < 75%:**
- ❌ Check data loading (verify HDF5 file)
- ❌ Review hyperparameters
- ❌ Check for data issues
- ❌ Analyze per-tissue metrics

---

## 🎓 Key Features Implemented

### 1. Robust Checkpointing
- ✅ Auto-saves every N epochs
- ✅ Saves best validation loss model
- ✅ Saves best validation accuracy model
- ✅ Auto-resume from last checkpoint
- ✅ Max 5 resumptions (prevents infinite loops)

### 2. Comprehensive Logging
- ✅ TensorBoard real-time visualization
- ✅ CSV logs (epoch-level metrics)
- ✅ Per-tissue performance tracking
- ✅ GPU memory monitoring
- ✅ Auto-generated training summary

### 3. Smart Training
- ✅ OneCycleLR scheduler (cosine annealing)
- ✅ Gradient clipping (prevents explosions)
- ✅ Gradient accumulation (effective batch = 512)
- ✅ Label smoothing (prevents overconfidence)
- ✅ CpG dropout augmentation

### 4. Tissue-Balanced Sampling
- ✅ File-level balancing (22 tissues)
- ✅ Avoids PyTorch 2^24 limit
- ✅ Ensures equal representation in batches

### 5. Comprehensive Evaluation
- ✅ Overall accuracy
- ✅ Per-tissue accuracy, precision, recall, F1
- ✅ Top-3 accuracy
- ✅ Confusion matrix visualization
- ✅ Calibration analysis

---

## 📋 Configuration Summary

### Model Architecture
```
Type: DNABERT-S (Transformer)
Layers: 6
Hidden Size: 512
Attention Heads: 8
Parameters: ~20M
Output: 22 tissue classes
```

### Training Configuration
```
Batch Size: 128
Gradient Accumulation: 4x (effective = 512)
Learning Rate: 4e-4 (OneCycleLR)
Weight Decay: 0.01
Label Smoothing: 0.1
CpG Dropout: 0.05 (training only)
Max Gradient Norm: 1.0
```

### Hardware Requirements
```
GPU: A100 (1x)
GPU Memory: ~20-30 GB
RAM: 64 GB
CPU Cores: 12
Storage: ~10 GB (checkpoints + logs)
Time Limit: 120 hours
```

### Data Configuration
```
Train Files: 455 (23.2M regions)
Val Files: 135 (6.9M regions)
Test Files: 175 (8.9M regions)
Total Regions: 39.0M
Sequence Length: 150 bp
Tissue Classes: 22
```

---

## 🔧 Troubleshooting

### Job Fails Immediately
```bash
# Check SLURM error log
cat /path/to/slurm_*.err

# Common causes:
# 1. Wrong HDF5 path → Edit config_*.yaml
# 2. Module loading failed → Check LMOD.sourceme
# 3. Python import error → Test with: python train.py --help
```

### Out of Memory
```bash
# Reduce batch size in config
# Edit config_20epoch.yaml:
batch_size: 64  # Instead of 128
gradient_accumulation_steps: 8  # Keep effective = 512
```

### Job Timeout
- ✅ Training auto-resumes (up to 5 times)
- ✅ Loads from checkpoint_last.pt
- ✅ Continues from last completed epoch
- ⚠️  If you hit 5 resume limit, manually resubmit

### Slow Training
```bash
# Increase data loading workers in config
num_workers: 12  # Up to CPU core count

# Check GPU utilization
nvidia-smi
# Should show >80% GPU utilization
```

---

## 📞 Support & Next Steps

### Current Status
✅ **Step 2.1-2.2 Complete**: Data pipeline + Model architecture
✅ **Step 2.3 Complete**: Loss functions + Training setup
⏳ **Step 3.1-3.7**: Execute training (you'll do this now!)

### Next Milestones

**After 20 Epochs:**
1. Analyze results
2. Validate performance (target: >75% accuracy)
3. Decision: Continue to 50 epochs?

**After 50 Epochs:**
1. Comprehensive evaluation
2. Per-tissue analysis
3. Begin inference pipeline development

**After 100 Epochs (if needed):**
1. Final model selection
2. Test set evaluation
3. Start deconvolution experiments

---

## 📚 Documentation Reference

- **Roadmap**: `/mnt/project/PDAC_cfDNA_Deconvolution_Roadmap.md`
- **Data Structure**: `/mnt/project/TRAINING_DATA_STRUCTURE.md`
- **Training README**: `README_TRAINING.md` (in this directory)

---

## ✅ Pre-Flight Checklist

Before submitting your first job, verify:

- [ ] All files copied to `/home/chattopa/.../ step_3_model_training/`
- [ ] Files are executable (`chmod +x *.sh *.py`)
- [ ] HDF5 path is correct in config files
- [ ] CSV paths point to /mnt/project/*.csv
- [ ] Output directories will be created automatically
- [ ] LMOD.sourceme exists and loads modules correctly
- [ ] Email address updated in submit_training.sh (optional)

---

## 🎉 You're Ready to Train!

Submit your first job:
```bash
cd /home/chattopa/data_storage/TissueBERT_analysis/step_3_model_training
sbatch submit_training.sh config_20epoch.yaml
```

Monitor it:
```bash
tail -f /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs/slurm_*.out
```

Good luck! 🚀

---

**Document Version**: 1.0
**Date**: 2024-11-20
**Status**: Ready for deployment
