#!/bin/bash
#SBATCH --job-name=step2_2_setup
#SBATCH --account=lu2025-7-54
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --output=step2_2_setup_%j.log
#SBATCH --error=step2_2_setup_%j.err

# Step 2.2 Setup Script
# This script will:
# 1. Convert NPZ files to HDF5
# 2. Validate dataloaders with 10 samples

echo "============================================================"
echo "Step 2.2: Dataset and DataLoader Setup"
echo "============================================================"
echo ""
echo "Start time: $(date)"
echo ""

# Load modules
source /home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/LMOD.sourceme

# Set working directory
WORK_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/step2.2/"
cd $WORK_DIR || exit 1

echo "Working directory: $WORK_DIR"
echo ""

# Step 1: Convert to HDF5
echo "============================================================"
echo "Step 1: Converting NPZ files to HDF5"
echo "============================================================"
echo ""

python3 convert_to_hdf5.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ HDF5 conversion failed!"
    exit 1
fi

echo ""
echo "✓ HDF5 conversion completed successfully!"
echo ""

# Step 2: Validate dataloaders
echo "============================================================"
echo "Step 2: Validating DataLoaders (10 samples)"
echo "============================================================"
echo ""

python3 dataset_dataloader.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ DataLoader validation failed!"
    exit 1
fi

echo ""
echo "✓ DataLoader validation completed successfully!"
echo ""

# Summary
echo "============================================================"
echo "Step 2.2 Setup Complete!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  - HDF5 dataset: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5"
echo ""
echo "Next steps:"
echo "  1. Review the validation output above"
echo "  2. If everything looks good, proceed to Step 2.3 (training script)"
echo "  3. Import dataloaders in training: from dataset_dataloader import create_dataloaders"
echo ""
echo "End time: $(date)"
echo "============================================================"
