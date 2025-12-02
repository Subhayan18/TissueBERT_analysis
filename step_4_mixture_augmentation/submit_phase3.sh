#!/bin/bash
#SBATCH --job-name=phase3_deconv
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/logs/slurm_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/logs/slurm_%j.err

################################################################################
# Phase 3: Realistic cfDNA Mixture Deconvolution Training
# 
# Teaches model to handle blood-dominant mixtures simulating cancer patient cfDNA
# Fine-tunes from Phase 2 checkpoint
# FINAL PRODUCTION MODEL
################################################################################

echo "============================================================"
echo "Phase 3: Realistic cfDNA Mixture Deconvolution Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration
WORK_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation"
CONFIG_FILE="config_phase3_realistic.yaml"

echo ""
echo "Configuration:"
echo "  Working directory: $WORK_DIR"
echo "  Config file: $CONFIG_FILE"
echo ""

# Verify Phase 2 checkpoint exists
PHASE2_CHECKPOINT="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_multitissue/checkpoints/checkpoint_best.pt"
if [ ! -f "$PHASE2_CHECKPOINT" ]; then
    echo "ERROR: Phase 2 checkpoint not found at $PHASE2_CHECKPOINT"
    echo "  You must complete Phase 2 training first!"
    exit 1
fi
echo "✓ Phase 2 checkpoint found"

# Create output directories
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/logs
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/checkpoints
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/results

# Change to working directory
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
echo "Starting Phase 3 Training (FINAL MODEL)"
echo "============================================================"
echo ""

python3 train_deconvolution.py --config $CONFIG_FILE

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Phase 3 Training Complete"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓✓✓ Phase 3 completed successfully! ✓✓✓"
    echo ""
    echo "ALL PHASES COMPLETE!"
    echo ""
    echo "Final production model saved at:"
    echo "  /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/checkpoints/checkpoint_best.pt"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate model on test set"
    echo "  2. Apply to PDAC patient samples"
    echo "  3. Analyze tissue proportion changes over time"
    echo ""
else
    echo ""
    echo "✗ Phase 3 failed with exit code $EXIT_CODE"
    echo "  Check logs for details"
fi

exit $EXIT_CODE
