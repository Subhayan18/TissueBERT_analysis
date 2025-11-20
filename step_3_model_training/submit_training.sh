#!/bin/bash
#SBATCH --job-name=tissuebert_train
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs/slurm_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=subhayan.chattopadhyay@med.lu.se

################################################################################
# TissueBERT Training Job
# 
# This script trains the TissueBERT model for tissue classification
# from DNA methylation patterns
#
# Usage:
#   sbatch submit_training.sh <config_file>
#
# Example:
#   sbatch submit_training.sh config_20epoch.yaml
################################################################################

echo "============================================================"
echo "TissueBERT Training Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration
CONFIG_FILE=${1:-"config_20epoch.yaml"}
WORK_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_3_model_training"

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Working directory: $WORK_DIR"
echo ""

# Change to working directory
cd $WORK_DIR || exit 1

# Load modules
echo "Loading modules..."
source /home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/LMOD.sourceme

# Verify GPU is available
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Print Python environment info
echo "Python environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
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

# If training completed successfully, check if we should continue
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    
    # Check if checkpoint exists and we haven't exceeded resume limit
    CHECKPOINT_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/checkpoints"
    LAST_CHECKPOINT="$CHECKPOINT_DIR/checkpoint_last.pt"
    
    if [ -f "$LAST_CHECKPOINT" ]; then
        # Extract resume count from checkpoint
        RESUME_COUNT=$(python -c "
import torch
ckpt = torch.load('$LAST_CHECKPOINT', map_location='cpu')
print(ckpt.get('resume_count', 0))
")
        
        echo "  Resume count: $RESUME_COUNT"
        
        if [ "$RESUME_COUNT" -lt 5 ]; then
            echo "  Auto-resubmission enabled (under resume limit)"
            echo "  Resubmitting job..."
            
            # Resubmit the job
            sbatch $0 $CONFIG_FILE
        else
            echo "  Resume limit reached. No auto-resubmission."
        fi
    fi
else
    echo ""
    echo "Training failed with exit code $EXIT_CODE"
    echo "Check logs for details:"
    echo "  stdout: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs/slurm_${SLURM_JOB_ID}.out"
    echo "  stderr: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_results/logs/slurm_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
