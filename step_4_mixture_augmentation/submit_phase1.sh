#!/bin/bash
#SBATCH --job-name=phase1_deconv
#SBATCH --partition=gpua100i
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/logs/slurm_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/logs/slurm_%j.err

################################################################################
# Phase 1: 2-Tissue Mixture Deconvolution Training
# 
# Teaches model to decompose simple binary mixtures
# Fine-tunes from single-tissue classification checkpoint
################################################################################

echo "============================================================"
echo "Phase 1: 2-Tissue Mixture Deconvolution Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration
WORK_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation"
CONFIG_FILE="config_phase1_2tissue.yaml"

echo ""
echo "Configuration:"
echo "  Working directory: $WORK_DIR"
echo "  Config file: $CONFIG_FILE"
echo ""

# Create output directories
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/logs
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/checkpoints
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/results

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
echo "Starting Phase 1 Training"
echo "============================================================"
echo ""

python3 train_deconvolution.py --config $CONFIG_FILE

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Phase 1 Training Complete"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Phase 1 completed successfully!"
    echo "  Next: Run Phase 2 (3-5 tissue mixtures)"
    echo "  Command: sbatch submit_phase2.sh"
else
    echo ""
    echo "✗ Phase 1 failed with exit code $EXIT_CODE"
    echo "  Check logs for details"
fi

exit $EXIT_CODE
