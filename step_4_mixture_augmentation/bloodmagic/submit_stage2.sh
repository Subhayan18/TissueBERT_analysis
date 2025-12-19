#!/bin/bash
#SBATCH --job-name=stage2_bloodmasked
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250G
#SBATCH --time=40:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked/logs/slurm_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked/logs/slurm_%j.err

################################################################################
# Stage 2: Blood-Masked Tissue Deconvolution Training
# 
# Trains model to predict NON-BLOOD tissue proportions from blood-dominant cfDNA
# Fine-tunes from Phase 3 checkpoint with modified output (21 vs 22 classes)
# 
# Purpose: Enable detection of trace tissue signals masked by blood
################################################################################

echo "============================================================"
echo "Stage 2: Blood-Masked Tissue Deconvolution Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration
WORK_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation/bloodmagic"
CONFIG_FILE="config_stage2_bloodmasked.yaml"

echo ""
echo "Configuration:"
echo "  Working directory: $WORK_DIR"
echo "  Config file: $CONFIG_FILE"
echo ""

# Change to working directory FIRST
cd $WORK_DIR || exit 1
echo "✓ Changed to working directory"

# Verify required Python files exist
echo ""
echo "Checking required files..."
REQUIRED_FILES=(
    "train_deconvolution.py"
    "model_deconvolution.py"
    "dataloader_mixture_stage2.py"
    "train_stage2_bloodmasked.py"
    "config_stage2_bloodmasked.yaml"
)

MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file MISSING!"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_FILES required file(s) missing from $WORK_DIR"
    echo "  Please copy all Stage 2 files to the working directory"
    exit 1
fi
echo "✓ All required files present"

# Verify Phase 3 checkpoint exists
PHASE3_CHECKPOINT="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/checkpoints/checkpoint_best.pt"
if [ ! -f "$PHASE3_CHECKPOINT" ]; then
    echo "ERROR: Phase 3 checkpoint not found at $PHASE3_CHECKPOINT"
    echo "  You must complete Phase 3 training first!"
    exit 1
fi
echo "✓ Phase 3 checkpoint found"

# Verify Stage 2 mixture datasets exist
VAL_MIXTURES="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/stage2_validation_mixtures.h5"
TEST_MIXTURES="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/stage2_test_mixtures.h5"

if [ ! -f "$VAL_MIXTURES" ] || [ ! -f "$TEST_MIXTURES" ]; then
    echo "ERROR: Stage 2 mixture datasets not found"
    echo "  Expected:"
    echo "    $VAL_MIXTURES"
    echo "    $TEST_MIXTURES"
    echo ""
    echo "  Please run: python generate_stage2_mixtures.py first"
    exit 1
fi
echo "✓ Stage 2 mixture datasets found"

# Create output directories
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked/logs
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked/checkpoints
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked/results

# Change to working directory (after verifying files)
cd $WORK_DIR || exit 1

# Load modules
echo "Loading modules..."
source /home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/LMOD.sourceme

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Python info
echo "Python environment:"
echo "  Python: $(which python3)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo ""

# Run training
echo "============================================================"
echo "Starting Stage 2 Training (Blood-Masked Deconvolution)"
echo "============================================================"
echo ""
echo "Training strategy:"
echo "  - Input: Mixed methylation WITH blood (60-100%)"
echo "  - Output: Non-blood tissue proportions (21 classes)"
echo "  - Fine-tuning: From Phase 3 checkpoint"
echo "  - Goal: Detect trace tissue signals masked by blood"
echo ""

python3 train_stage2_bloodmasked.py --config $CONFIG_FILE

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Stage 2 Training Complete"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓✓✓ Stage 2 completed successfully! ✓✓✓"
    echo ""
    echo "Stage 2 model saved at:"
    echo "  /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked/checkpoints/checkpoint_best.pt"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate Stage 2 model on test set"
    echo "  2. Combine Stage 1 (Phase 3) + Stage 2 predictions"
    echo "  3. Apply two-stage deconvolution to PDAC samples"
    echo ""
    echo "Two-stage inference pipeline:"
    echo "  Stage 1: Predict all 22 tissues (including blood)"
    echo "  Stage 2: Predict 21 non-blood tissues"
    echo "  Combine: Blood from Stage 1, tissues from Stage 2 scaled by (1-blood)"
    echo ""
else
    echo ""
    echo "✗ Stage 2 failed with exit code $EXIT_CODE"
    echo "  Check logs for details"
fi

exit $EXIT_CODE
