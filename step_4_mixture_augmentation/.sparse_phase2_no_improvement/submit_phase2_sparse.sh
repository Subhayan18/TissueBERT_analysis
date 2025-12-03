#!/bin/bash
#SBATCH --job-name=phase2_sparse
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_sparse/logs/slurm_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_sparse/logs/slurm_%j.err

################################################################################
# Phase 2 SPARSE: 3-5 Tissue Mixture Deconvolution with Sparsity Regularization
# 
# Two-stage architecture with presence detection and sparsity constraints
# Addresses baseline noise problem (spurious 1-5% predictions for absent tissues)
# Fine-tunes from Phase 1 checkpoint
################################################################################

echo "============================================================"
echo "Phase 2 SPARSE: Mixture Deconvolution with Sparsity"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration
WORK_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation"
CONFIG_FILE="config_phase2_sparse.yaml"

echo ""
echo "Configuration:"
echo "  Working directory: $WORK_DIR"
echo "  Config file: $CONFIG_FILE"
echo "  Architecture: Two-Stage (Presence Detection + Proportion Estimation)"
echo "  Sparsity: ENABLED"
echo ""

# Verify Phase 1 checkpoint exists
PHASE1_CHECKPOINT="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/checkpoints/checkpoint_best.pt"
if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    echo "ERROR: Phase 1 checkpoint not found at $PHASE1_CHECKPOINT"
    echo "  You must complete Phase 1 training first!"
    exit 1
fi
echo "✓ Phase 1 checkpoint found"

# Verify sparse model and config files exist
if [ ! -f "$WORK_DIR/model_deconvolution_sparse.py" ]; then
    echo "ERROR: model_deconvolution_sparse.py not found in $WORK_DIR"
    echo "  Please copy from: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/"
    exit 1
fi
echo "✓ Sparse model file found"

if [ ! -f "$WORK_DIR/train_deconvolution_sparse.py" ]; then
    echo "ERROR: train_deconvolution_sparse.py not found in $WORK_DIR"
    echo "  Please copy from: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/"
    exit 1
fi
echo "✓ Sparse training script found"

if [ ! -f "$WORK_DIR/$CONFIG_FILE" ]; then
    echo "ERROR: $CONFIG_FILE not found in $WORK_DIR"
    echo "  Please copy from: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/"
    exit 1
fi
echo "✓ Sparse config file found"

# Create output directories
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_sparse/logs
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_sparse/checkpoints
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_sparse/results

# Change to working directory
cd $WORK_DIR || exit 1

# Load modules
echo ""
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

# Test sparse model before training (quick sanity check)
echo "============================================================"
echo "Testing Sparse Model Architecture"
echo "============================================================"
echo ""

python3 -c "
from model_deconvolution_sparse import TissueBERTDeconvolution
import torch

# Quick test
model = TissueBERTDeconvolution(
    hidden_size=512,
    num_classes=22,
    dropout=0.1,
    intermediate_size=2048,
    n_regions=51089,
    use_two_stage=True
)
print('✓ Sparse model initialized successfully')
print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'  Two-stage: {model.use_two_stage}')
print(f'  Sparsity regularization: {model.sparsity_regularization}')
" || { echo "ERROR: Sparse model test failed"; exit 1; }

echo ""

# Run training
echo "============================================================"
echo "Starting Phase 2 SPARSE Training"
echo "============================================================"
echo ""
echo "Expected improvements over original Phase 2:"
echo "  - Eliminate spurious predictions (1-5% for absent tissues)"
echo "  - Improve true tissue accuracy (50% → 100% prediction ratio)"
echo "  - Better sparsity (only present tissues predicted)"
echo ""
echo "Key metrics to monitor:"
echo "  - Presence Accuracy: Should reach >90%"
echo "  - Validation MAE: Should be <5% (similar to original)"
echo "  - Sparsity Loss: Should decrease over time"
echo ""

python3 train_deconvolution_sparse.py --config $CONFIG_FILE

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Phase 2 SPARSE Training Complete"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Phase 2 SPARSE completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate results with Miami plot:"
    echo "     python visualize_mixture_miami.py \\"
    echo "       --checkpoint /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_sparse/checkpoints/checkpoint_best.pt \\"
    echo "       --test_h5 /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/phase2_test_mixtures.h5 \\"
    echo "       --output miami_plot_sparse.png \\"
    echo "       --summary summary_stats_sparse.csv"
    echo ""
    echo "  2. Compare to original Phase 2 (non-sparse) results"
    echo ""
    echo "  3. If successful, proceed to Phase 3:"
    echo "     sbatch submit_phase3_sparse.sh"
    echo ""
else
    echo ""
    echo "✗ Phase 2 SPARSE failed with exit code $EXIT_CODE"
    echo "  Check logs for details:"
    echo "    - SLURM log: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_sparse/logs/slurm_$SLURM_JOB_ID.{out,err}"
    echo "    - Training log: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_sparse/logs/training_log.csv"
    echo "    - TensorBoard: tensorboard --logdir=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_sparse/logs/tensorboard"
fi

exit $EXIT_CODE
