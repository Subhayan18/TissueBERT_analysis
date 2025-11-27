# TissueBERT Training Setup - Step 3

Complete training pipeline for TissueBERT tissue classification model.

---

## 📁 Files Overview

### Core Training Files

1. **`train.py`** - Main training script
   - Implements complete training loop
   - Handles checkpointing and resumption
   - Comprehensive logging (TensorBoard + CSV)
   - Auto-evaluation on test set

2. **`model.py`** - TissueBERT model architecture
   - DNABERT-S implementation (6-layer transformer)
   - DNA sequence + methylation embedding
   - Multi-head attention mechanism
   - Classification head for 22 tissues

3. **`utils.py`** - Utility functions
   - Metrics computation (accuracy, F1, confusion matrix)
   - Checkpointing functions
   - Training visualization
   - Comprehensive summary generation

4. **`config_*.yaml`** - Training configurations
   - `config_20epoch.yaml` - Initial 20 epoch run
   - `config_50epoch.yaml` - Medium 50 epoch run
   - `config_100epoch.yaml` - Full 100 epoch run

5. **`submit_training.sh`** - SLURM job submission script
   - Auto-loads modules
   - GPU allocation (A100)
   - Auto-resubmission (up to 5 times)
   - Comprehensive logging

6. **`analyze_results.py`** - Results analysis script
   - Analyzes training logs
   - Per-tissue performance analysis
   - Generates comprehensive reports

---

## 🚀 Quick Start

### 1. Setup (First Time Only)

```bash
# Navigate to training directory
cd /home/chattopa/data_storage/TissueBERT_analysis/step_3_model_training

# Copy all training files here
cp /path/to/train.py .
cp /path/to/model.py .
cp /path/to/utils.py .
cp /path/to/config_*.yaml .
cp /path/to/submit_training.sh .
cp /path/to/analyze_results.py .

# Make scripts executable
chmod +x submit_training.sh
chmod +x analyze_results.py

# Test model architecture
python model.py
```

### 2. Start Training (20 Epochs)

```bash
# Submit training job
sbatch submit_training.sh config_20epoch.yaml

# Check job status
squeue -u $USER

# Monitor training (real-time)
tail -f /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs/slurm_*.out
```

### 3. Monitor Progress

```bash
# View TensorBoard (from local machine)
# First, setup SSH tunnel:
ssh -L 6006:localhost:6006 user@hpc-cluster

# Then on HPC:
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs
tensorboard --logdir tensorboard --port 6006

# Open browser: http://localhost:6006
```

### 4. Analyze Results

```bash
# After training completes
python analyze_results.py \
    --log_dir /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs \
    --results_dir /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/results \
    --output analysis_report.txt

# View report
cat analysis_report.txt
```

### 5. Continue Training (50 or 100 Epochs)

If 20 epochs show good convergence:

```bash
# Submit 50 epoch run
sbatch submit_training.sh config_50epoch.yaml

# Or full 100 epoch run
sbatch submit_training.sh config_100epoch.yaml
```

---

## 📊 Output Structure

```
/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/
├── checkpoints/
│   ├── checkpoint_last.pt          # Latest checkpoint (auto-resume)
│   ├── checkpoint_best_loss.pt     # Best validation loss
│   ├── checkpoint_best_acc.pt      # Best validation accuracy
│   ├── checkpoint_epoch_005.pt     # Periodic checkpoints
│   ├── checkpoint_epoch_010.pt
│   └── ...
│
├── logs/
│   ├── tensorboard/                # TensorBoard logs
│   ├── training_log.csv            # Epoch-level metrics
│   ├── per_tissue_metrics.csv      # Per-tissue performance
│   ├── training_summary.txt        # Auto-generated summary
│   ├── slurm_*.out                 # SLURM stdout
│   └── slurm_*.err                 # SLURM stderr
│
└── results/
    ├── final_results.json          # Test set results (JSON)
    ├── training_curves.png         # Loss/accuracy plots
    ├── per_tissue_curves.png       # Per-tissue F1 curves
    ├── confusion_epoch_*.png       # Confusion matrices
    └── confusion_test_final.png    # Final test confusion matrix
```

---

## ⚙️ Configuration Details

### Model Architecture
```yaml
vocab_size: 69              # DNA 3-mers + special tokens
hidden_size: 512            # Transformer hidden dimension
num_layers: 6               # Transformer layers
num_attention_heads: 8      # Attention heads
num_classes: 22             # Tissue types
```

### Training Parameters
```yaml
batch_size: 128             # Samples per batch
gradient_accumulation: 4    # Effective batch = 512
learning_rate: 4e-4         # Base LR
weight_decay: 0.01          # L2 regularization
label_smoothing: 0.1        # Regularization
cpg_dropout: 0.05           # Augmentation
```

### Hardware Utilization
- **GPU**: A100 (1x)
- **Memory**: ~20-30 GB GPU, 64 GB RAM
- **Batch size**: 128 (can handle up to ~200)
- **Workers**: 8 parallel data loaders

---

## 📈 Expected Performance

### Training Dynamics

**First 5 Epochs:**
- Loss: 3.0 → 1.5 (rapid decrease)
- Accuracy: 10% → 50%
- Time: ~30-45 min per epoch

**Epochs 5-20:**
- Loss: 1.5 → 0.5
- Accuracy: 50% → 75-80%
- Gradual improvement

**Epochs 20-50:**
- Loss: 0.5 → 0.3
- Accuracy: 75-80% → 85%+
- Slower convergence

**Success Criteria (20 Epochs):**
- ✓ Validation accuracy > 75%
- ✓ Train-val gap < 10%
- ✓ All tissues learning (F1 > 0.5)

**Success Criteria (50 Epochs):**
- ✓ Validation accuracy > 85%
- ✓ Per-tissue F1 > 0.7
- ✓ Top-3 accuracy > 95%

---

## 🔍 Monitoring Training

### TensorBoard Metrics

**Scalar Metrics:**
- `epoch/train_loss` - Training loss
- `epoch/val_loss` - Validation loss
- `epoch/train_acc` - Training accuracy
- `epoch/val_acc` - Validation accuracy
- `tissue/f1_tissue_*` - Per-tissue F1 scores
- `train/lr` - Learning rate schedule

**View in TensorBoard:**
```bash
tensorboard --logdir /path/to/logs/tensorboard
```

### CSV Logs

**training_log.csv:**
```
epoch,step,train_loss,train_acc,val_loss,val_acc,learning_rate,time_elapsed,gpu_memory
1,22680,0.8234,0.6543,0.7821,0.6789,0.0004,1823.45,18.34
...
```

**per_tissue_metrics.csv:**
```
epoch,tissue_id,accuracy,precision,recall,f1
1,0,0.7234,0.7123,0.7345,0.7234
1,1,0.8123,0.8234,0.8012,0.8123
...
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size in config
batch_size: 64  # Instead of 128
gradient_accumulation: 8  # Keep effective batch = 512
```

### Slow Training
```bash
# Increase workers
num_workers: 12  # Up to number of CPU cores

# Enable mixed precision (add to train.py if needed)
# Use torch.cuda.amp.autocast()
```

### Job Timeout
- Training auto-resumes from last checkpoint
- Max 5 resumptions (configurable in config.yaml)
- Each resumption continues from where it left off

### Poor Performance
1. Check data loading:
   ```bash
   python -c "from dataset_dataloader import create_dataloaders; print('OK')"
   ```

2. Check model:
   ```bash
   python model.py
   ```

3. Analyze logs:
   ```bash
   python analyze_results.py --log_dir logs --results_dir results
   ```

---

## 📝 Checkpointing & Resumption

### Automatic Resumption

Training automatically resumes from `checkpoint_last.pt` if it exists:
- Restores model, optimizer, scheduler states
- Continues from last completed epoch
- Tracks number of resumptions (max 5)

### Manual Checkpoint Loading

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_best_acc.pt')

# Extract information
epoch = checkpoint['epoch']
best_val_acc = checkpoint['best_val_acc']
model_state = checkpoint['model_state_dict']

# Load into model
model.load_state_dict(model_state)
```

---

## 🎯 Next Steps After Training

### 1. Validate Performance
```bash
# Analyze results
python analyze_results.py --log_dir logs --results_dir results

# Check test accuracy > 85%
# Verify per-tissue F1 > 0.7
```

### 2. Test on Validation Samples
```python
# Create inference script (inference.py)
# Load best model
# Test on individual regions
# Aggregate to sample-level predictions
```

### 3. Create Mixtures for Deconvolution
```python
# Simulate tissue mixtures
# Test model on mixed samples
# Validate deconvolution accuracy
```

### 4. Apply to Real cfDNA
```python
# Process PDAC cfDNA samples
# Generate tissue proportion estimates
# Track over time (T0, T1, T2, T3)
```

---

## 📚 Key Metrics to Watch

### During Training
1. **Validation Loss** - Should decrease steadily
2. **Validation Accuracy** - Target: >85% by epoch 50
3. **Train-Val Gap** - Should be <10% (no overfitting)
4. **Per-Tissue F1** - All tissues should improve
5. **Learning Rate** - Follows OneCycleLR schedule

### Final Evaluation
1. **Test Accuracy** - Overall performance
2. **Top-3 Accuracy** - Captures uncertainty
3. **Macro F1** - Average across all tissues
4. **Weighted F1** - Accounts for class imbalance
5. **Confusion Matrix** - Shows which tissues are confused

---

## ⚠️ Important Notes

1. **Auto-resubmission**: Job auto-resubmits up to 5 times if it times out
2. **Checkpointing**: Always saves `checkpoint_last.pt` for resumption
3. **Best models**: Saves separate checkpoints for best loss and best accuracy
4. **Data location**: HDF5 file must be at correct path in config
5. **Module loading**: LMOD.sourceme must load Python, PyTorch, CUDA

---

## 🆘 Getting Help

If training fails:

1. **Check SLURM logs**:
   ```bash
   cat /path/to/slurm_*.err
   ```

2. **Check training logs**:
   ```bash
   tail -100 /path/to/logs/training_log.csv
   ```

3. **Verify data**:
   ```bash
   python -c "import h5py; f = h5py.File('path/to/data.h5', 'r'); print(f.keys())"
   ```

4. **Test model**:
   ```bash
   python model.py
   ```

---

## 📧 Questions?

Check the roadmap for context:
- `/mnt/project/PDAC_cfDNA_Deconvolution_Roadmap.md`

Review training data structure:
- `/mnt/project/TRAINING_DATA_STRUCTURE.md`

---

**Good luck with training! 🚀**
