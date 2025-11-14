# Setup Instructions - Updated Directory Structure

## 📁 Directory Structure

```
/home/chattopa/data_storage/
├── TissueBERT_analysis/
│   └── step_1_data_preparation/
│       └── step1.4/                          ← Scripts go here (YOU EXECUTE FROM HERE)
│           ├── config.py
│           ├── data_loader.py
│           ├── data_augmentation.py
│           ├── create_training_dataset.py
│           ├── run_step1.4.sh
│           └── README.md
│
└── MethAtlas_WGBSanalysis/                   ← Data stays here
    ├── panel_beta_matrix.tsv                 ← INPUT: Beta values
    ├── synthetic_reads/                      ← INPUT: Synthetic reads
    │   ├── sample_000_Adipocytes.npz
    │   └── ...
    ├── TWIST_panel_sequences_3mers.txt       ← INPUT: 3-mer sequences
    ├── TWIST_blocks.bed                      ← INPUT: Panel regions
    └── training_dataset/                     ← OUTPUT: Created by scripts
        ├── all_data/                         ← 30M .npz files go here
        ├── metadata.csv
        └── logs/
```

## 🚀 Setup Steps

### Step 1: Create Script Directory

```bash
# Create the directory structure
mkdir -p /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4

# Navigate to it
cd /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4
```

### Step 2: Copy Scripts Here

Download all files from Claude's outputs, then copy to the script directory:

```bash
# Copy all Python scripts
cp /path/to/downloaded/config.py .
cp /path/to/downloaded/data_loader.py .
cp /path/to/downloaded/data_augmentation.py .
cp /path/to/downloaded/create_training_dataset.py .
cp /path/to/downloaded/run_step1.4.sh .
cp /path/to/downloaded/README.md .

# Or use scp/sftp to upload directly to this directory
```

### Step 3: Verify Input Files Exist

```bash
# Check that all input files are present in MethAtlas_WGBSanalysis
ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/panel_beta_matrix.tsv
ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads/ | head
ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/TWIST_panel_sequences_3mers.txt
ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/TWIST_blocks.bed
```

### Step 4: Run Test Mode

```bash
# Make sure you're in the script directory
cd /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4

# Run test (processes 5 samples)
python3 create_training_dataset.py --test
```

**Output will be created in:**
`/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/`

### Step 5: Run Full Dataset

```bash
# Option A: Submit SLURM job (RECOMMENDED)
# First edit run_step1.4.sh and set your partition name
sbatch run_step1.4.sh

# Option B: Run interactively
python3 create_training_dataset.py
```

## ✅ What This Setup Does

### Scripts Location (Where You Work)
- **Location**: `/home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4`
- **Purpose**: Keep all your analysis code organized
- **You execute from here**: All commands run from this directory

### Data Location (Where Data Lives)
- **Input Location**: `/home/chattopa/data_storage/MethAtlas_WGBSanalysis/`
  - `panel_beta_matrix.tsv`
  - `synthetic_reads/`
  - `TWIST_panel_sequences_3mers.txt`
  - `TWIST_blocks.bed`
  
- **Output Location**: `/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/`
  - `all_data/` (30M .npz files)
  - `metadata.csv`
  - `logs/`

## 🔍 Verification Commands

```bash
# From the script directory, check the configuration
cd /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4
python3 -c "import config; config.print_config_summary(); config.validate_input_files()"

# This will print:
#   Script directory: /home/chattopa/.../step1.4
#   Data directory: /home/chattopa/.../MethAtlas_WGBSanalysis
#   Output directory: /home/chattopa/.../MethAtlas_WGBSanalysis/training_dataset
```

## 📝 After Running

### Monitor Progress
```bash
# Watch log file (from any directory)
tail -f /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/logs/training_dataset_*.log

# Count output files
watch -n 10 'ls /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/all_data/*.npz | wc -l'

# Check disk usage
du -sh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/
```

## 🎯 Key Points

1. **Scripts are in**: `TissueBERT_analysis/step_1_data_preparation/step1.4/`
2. **Data stays in**: `MethAtlas_WGBSanalysis/`
3. **Execute from**: Script directory
4. **Output goes to**: Data directory

This keeps your code organized while keeping data in its original location!

## ⚠️ Important Notes

- Always run commands from: `/home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4`
- The scripts will automatically read from and write to: `/home/chattopa/data_storage/MethAtlas_WGBSanalysis/`
- No need to change directories between script location and data location
- All paths are absolute, so it works regardless of where you submit the SLURM job

## 🐛 Troubleshooting

**Problem**: "No such file or directory" errors for input files

**Solution**: Check that paths in `config.py` are correct:
```python
DATA_BASE_DIR = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis"
```

**Problem**: Output not being created

**Solution**: Verify you have write permissions:
```bash
ls -ld /home/chattopa/data_storage/MethAtlas_WGBSanalysis/
```

**Problem**: Scripts not found

**Solution**: Make sure you're in the script directory:
```bash
cd /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4
ls -la *.py
```
