#!/bin/bash
#SBATCH --job-name=TB_fullgenome
#SBATCH --partition=gpua100i
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/logs/slurm_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/logs/slurm_%j.err

################################################################################
# TissueBERT Full Genome Training
# 
# Trains on ALL chromosomes (51,089 regions per file)
# Optimized for 40GB GPU memory
################################################################################

echo "============================================================"
echo "TissueBERT Full Genome Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration
CONFIG_FILE=${1:-"config_fullgenome.yaml"}
WORK_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_3_model_training/full_methylation_aggregation"

echo ""
echo "Configuration:"
echo "  Config: $CONFIG_FILE"
echo "  Working directory: $WORK_DIR"
echo "  Chromosomes: ALL (51,089 regions)"
echo ""

# Create output directories
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/logs
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/checkpoints

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
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Run training
echo "============================================================"
echo "Starting Training"
echo "============================================================"
echo ""

python train.py --config $CONFIG_FILE

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Training Complete"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "============================================================"

# Auto-resubmit if not finished
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    
    CHECKPOINT_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/checkpoints"
    LAST_CHECKPOINT="$CHECKPOINT_DIR/checkpoint_last.pt"
    
    if [ -f "$LAST_CHECKPOINT" ]; then
        RESUME_COUNT=$(python -c "
import torch
ckpt = torch.load('$LAST_CHECKPOINT', map_location='cpu')
print(ckpt.get('resume_count', 0))
")
        
        echo "  Resume count: $RESUME_COUNT"
        
        if [ "$RESUME_COUNT" -lt 5 ]; then
            echo "  Auto-resubmitting..."
            sbatch $0 $CONFIG_FILE
        else
            echo "  Resume limit reached."
        fi
    fi
else
    echo ""
    echo "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
