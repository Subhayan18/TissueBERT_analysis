#!/bin/bash
#SBATCH --job-name=phase2_deconv
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_multitissue/logs/slurm_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_multitissue/logs/slurm_%j.err

################################################################################
# Phase 2: 3-5 Tissue Mixture Deconvolution Training
# 
# Teaches model to handle moderate complexity mixtures
# Fine-tunes from Phase 1 checkpoint
################################################################################

echo "============================================================"
echo "Phase 2: 3-5 Tissue Mixture Deconvolution Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration
WORK_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation"
CONFIG_FILE="config_phase2_multitissue.yaml"

echo ""
echo "Configuration:"
echo "  Working directory: $WORK_DIR"
echo "  Config file: $CONFIG_FILE"
echo ""

# Verify Phase 1 checkpoint exists
PHASE1_CHECKPOINT="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/checkpoints/checkpoint_best.pt"
if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    echo "ERROR: Phase 1 checkpoint not found at $PHASE1_CHECKPOINT"
    echo "  You must complete Phase 1 training first!"
    exit 1
fi
echo "✓ Phase 1 checkpoint found"

# Create output directories
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_multitissue/logs
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_multitissue/checkpoints
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_multitissue/results

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
echo "Starting Phase 2 Training"
echo "============================================================"
echo ""

python3 train_deconvolution.py --config $CONFIG_FILE

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Phase 2 Training Complete"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Phase 2 completed successfully!"
    echo "  Next: Run Phase 3 (realistic cfDNA mixtures)"
    echo "  Command: sbatch submit_phase3.sh"
else
    echo ""
    echo "✗ Phase 2 failed with exit code $EXIT_CODE"
    echo "  Check logs for details"
fi

exit $EXIT_CODE
