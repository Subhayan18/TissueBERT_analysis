# Quick Start Guide - Step 1.4

## 📁 Directory Setup

**Scripts location (where you execute from):**
```
/home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4/
```

**Data location (where outputs are written):**
```
/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/
```

## 🚀 Get Started in 4 Steps

```bash
# 1. Create and navigate to script directory
mkdir -p /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4
cd /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4

# 2. Copy the scripts here (download from Claude first!)
# Place all .py files, .sh file, and README.md in this directory

# 3. Verify input files exist
ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/panel_beta_matrix.tsv
ls /home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads/ | head

# 4. Run test mode (takes ~30 minutes)
python3 create_training_dataset.py --test
```

## ✅ If Test Works, Run Full Dataset

```bash
# Make sure you're in the script directory
cd /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4

# Option A: Submit SLURM job (RECOMMENDED)
# First, edit run_step1.4.sh and set your partition name
sbatch run_step1.4.sh

# Option B: Run interactively (takes 3-5 hours)
python3 create_training_dataset.py
```

## 📋 Files You Need

Download these from Claude's outputs:
- config.py
- data_loader.py
- data_augmentation.py  
- create_training_dataset.py
- README.md
- run_step1.4.sh

## ⏱️ Time & Storage

| Mode | Time | Storage | Files |
|------|------|---------|-------|
| Test (5 samples) | ~30 min | ~10 GB | ~25 |
| Full (119 samples) | ~4 hours | ~300 GB | ~595 |

## 📊 Monitor Progress

```bash
# Watch the log
tail -f training_dataset/logs/training_dataset_*.log

# Count files created
watch -n 10 'ls training_dataset/all_data/*.npz | wc -l'
```

## 🎯 What You Get

**Output location:**
```
/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/
├── all_data/          # ~595 .npz files (PyTorch format)
│                      # Each file = 1 sample × 1 augmentation × all 51,089 regions
├── metadata.csv       # Summary of all training data
└── logs/             # Execution logs
```

## ❓ Problems?

1. **ImportError**: Make sure all .py files are in `/home/.../TissueBERT_analysis/.../step1.4/`
2. **No space**: Check `df -h /home/chattopa/data_storage/MethAtlas_WGBSanalysis`
3. **Input files not found**: Verify paths in config.py point to MethAtlas_WGBSanalysis
4. **Memory error**: Edit config.py, reduce N_CORES or CHUNK_SIZE
5. **Test mode fails**: Check the log file in MethAtlas_WGBSanalysis/training_dataset/logs/

## 📊 Monitor Progress

```bash
# Watch the log (from any directory)
tail -f /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/logs/training_dataset_*.log

# Count files created
watch -n 10 'ls /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/all_data/*.npz | wc -l'
```
