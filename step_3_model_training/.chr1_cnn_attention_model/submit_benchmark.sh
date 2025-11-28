#!/bin/bash
#SBATCH --job-name=benchmark_models
#SBATCH --partition=gpua100i
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/chr1_cnn_attention_results/logs/slurm_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/chr1_cnn_attention_results/logs/slurm_%j.err

################################################################################
# Model Benchmarking Job
# 
# This script compares the Aggregation model vs CNN+Attention model
# on the test set to determine which performs better.
#
# Usage:
#   sbatch submit_benchmark.sh
#
# Note: Edit the checkpoint paths below before running!
################################################################################

echo "============================================================"
echo "Model Benchmarking Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration - EDIT THESE PATHS!
WORK_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis"
AGG_CHECKPOINT="${WORK_DIR}/chr1_fast_results/checkpoints/best_model.pt"
CNN_CHECKPOINT="${WORK_DIR}/chr1_cnn_attention_results/checkpoints/best_model.pt"
AGG_CONFIG="/home/chattopa/data_storage/TissueBERT_analysis/step_3_model_training/chr1_methylation_aggregation/config_chr1_debug.yaml"
CNN_CONFIG="/home/chattopa/data_storage/TissueBERT_analysis/step_3_model_training/chr1_cnn_attention_model/config_chr1_cnn_attention.yaml"
OUTPUT_DIR="${WORK_DIR}/chr1_cnn_attention_results/benchmark_results"

echo ""
echo "Configuration:"
echo "  Working directory: $WORK_DIR"
echo "  Aggregation checkpoint: $AGG_CHECKPOINT"
echo "  CNN+Attention checkpoint: $CNN_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Change to working directory
cd $WORK_DIR || exit 1

# Create output directory
mkdir -p $OUTPUT_DIR

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
echo ""

# Verify checkpoints exist
echo "Checking checkpoints..."
if [ ! -f "$AGG_CHECKPOINT" ]; then
    echo "  ✗ Aggregation checkpoint not found: $AGG_CHECKPOINT"
    echo ""
    echo "ERROR: Please train the aggregation model first or update the path!"
    exit 1
fi
echo "  ✓ Aggregation checkpoint found"

if [ ! -f "$CNN_CHECKPOINT" ]; then
    echo "  ✗ CNN+Attention checkpoint not found: $CNN_CHECKPOINT"
    echo ""
    echo "ERROR: Please train the CNN+Attention model first or update the path!"
    exit 1
fi
echo "  ✓ CNN+Attention checkpoint found"

echo ""
echo "All checkpoints found!"
echo ""

# Run benchmark
echo "============================================================"
echo "Starting Benchmark"
echo "============================================================"
echo ""

python benchmark_models.py \
  --aggregation_checkpoint "$AGG_CHECKPOINT" \
  --cnn_checkpoint "$CNN_CHECKPOINT" \
  --aggregation_config "$AGG_CONFIG" \
  --cnn_config "$CNN_CONFIG" \
  --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Benchmark Complete"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Benchmark completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  $OUTPUT_DIR/benchmark_results.json"
    echo ""
    echo "View results:"
    echo "  cat $OUTPUT_DIR/benchmark_results.json"
else
    echo ""
    echo "Benchmark failed with exit code $EXIT_CODE"
    echo "Check logs for details:"
    echo "  stdout: $OUTPUT_DIR/slurm_${SLURM_JOB_ID}.out"
    echo "  stderr: $OUTPUT_DIR/slurm_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
