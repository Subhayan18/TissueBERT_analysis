#!/bin/bash
#SBATCH --job-name=cnn_attention_train
#SBATCH --partition=gpua100i
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/chr1_cnn_attention_results/logs/slurm_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/chr1_cnn_attention_results/logs/slurm_%j.err

################################################################################
# CNN + Attention Model Training Job
# 
# This script trains the CNN+Attention model for tissue classification
# from DNA methylation patterns (per-base encoding with file-level aggregation)
#
# Usage:
#   sbatch submit_cnn_attention_training.sh [config_file]
#
# Example:
#   sbatch submit_cnn_attention_training.sh config_chr1_cnn_attention.yaml
################################################################################

echo "============================================================"
echo "CNN + Attention Model Training Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration
CONFIG_FILE=${1:-"config_chr1_cnn_attention.yaml"}
#WORK_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis"

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Working directory: $WORK_DIR"
echo "  Model: CNN + Attention (per-base encoding)"
echo ""

# Change to working directory
#cd $WORK_DIR || exit 1

# Create log directory if it doesn't exist
mkdir -p /home/chattopa/data_storage/MethAtlas_WGBSanalysis/chr1_cnn_attention_results/logs

# Load modules
echo "Loading modules..."
source /home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/LMOD.sourceme

# Verify GPU is available
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# CRITICAL: Clean up any leftover GPU memory from previous jobs
echo "Cleaning GPU memory..."
echo "Before cleanup:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# Kill any stuck Python processes on this GPU (if any exist)
STUCK_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
if [ ! -z "$STUCK_PIDS" ]; then
    echo "Found stuck processes: $STUCK_PIDS"
    for PID in $STUCK_PIDS; do
        echo "  Killing PID $PID..."
        kill -9 $PID 2>/dev/null || true
    done
    sleep 2
fi

echo "After cleanup:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Prevent fragmentation

# Clear any stuck processes using GPU
echo "Checking for stuck GPU processes..."
nvidia-smi | grep 'python' || echo "No stuck processes found"

# Print Python environment info
echo "Python environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Verify required files exist
echo "Checking required files..."
REQUIRED_FILES=(
    "train_cnn_attention.py"
    "model_cnn_attention.py"
    "dataloader_filelevel.py"
    "$CONFIG_FILE"
)

ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ✗ Missing: $file"
        ALL_FILES_EXIST=false
    else
        echo "  ✓ Found: $file"
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    echo ""
    echo "ERROR: Required files are missing!"
    echo "Please ensure all files are in: $WORK_DIR"
    exit 1
fi

echo ""
echo "All required files found!"
echo ""

# Run training
echo "============================================================"
echo "Starting Training"
echo "============================================================"
echo ""

python train_cnn_attention.py --config $CONFIG_FILE

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
    CHECKPOINT_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/chr1_cnn_attention_results/checkpoints"
    LAST_CHECKPOINT="$CHECKPOINT_DIR/checkpoint_last.pt"
    
    if [ -f "$LAST_CHECKPOINT" ]; then
        # Extract resume count from checkpoint
        RESUME_COUNT=$(python -c "
import torch
ckpt = torch.load('$LAST_CHECKPOINT', map_location='cpu')
print(ckpt.get('resume_count', 0))
" 2>/dev/null)
        
        if [ ! -z "$RESUME_COUNT" ]; then
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
    fi
    
    echo ""
    echo "Results saved to:"
    echo "  Checkpoints: $CHECKPOINT_DIR"
    echo "  Logs: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/chr1_cnn_attention_results/logs"
    
else
    echo ""
    echo "Training failed with exit code $EXIT_CODE"
    echo "Check logs for details:"
    echo "  stdout: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/chr1_cnn_attention_results/logs/slurm_${SLURM_JOB_ID}.out"
    echo "  stderr: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/chr1_cnn_attention_results/logs/slurm_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
