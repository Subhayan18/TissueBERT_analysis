#!/bin/bash
#
# Step 1.5b - Generate Synthetic Samples (SLURM Array Job)
# ==========================================================
#
# This wrapper supports both single-job and parallel execution:
# - Single job: Processes all tissues sequentially
# - Array job: Processes one tissue per SLURM task (parallel)
#
# Usage:
#   # Sequential (single job):
#   bash run_generate_synthetic_binary.sh
#
#   # Parallel (submit as SLURM array):
#   sbatch run_generate_synthetic_binary.slurm
#

# Set paths
BASE_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis"
METADATA="${BASE_DIR}/training_dataset/updated_metadata.csv"
DATA_DIR="${BASE_DIR}/training_dataset/all_data"
OUTPUT_DIR="${BASE_DIR}/synthetic_samples"
PYTHON_SCRIPT="generate_synthetic_samples_binary.py"

# Parameters
MIN_SAMPLES=4
FLIP_RATE=0.10
CORRELATION_LENGTH=5
RANDOM_SEED=42

echo "========================================================================"
echo "Step 1.5b: Generate Synthetic Samples (Binary Methylation)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Metadata:           $METADATA"
echo "  Data directory:     $DATA_DIR"
echo "  Output directory:   $OUTPUT_DIR"
echo "  Min samples:        $MIN_SAMPLES"
echo "  Flip rate:          $FLIP_RATE"
echo "  Correlation length: $CORRELATION_LENGTH"
echo "  Random seed:        $RANDOM_SEED"
echo ""

# Check if metadata exists
if [ ! -f "$METADATA" ]; then
    echo "Error: Metadata file not found at $METADATA"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at $DATA_DIR"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if running as SLURM array job
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Running as SLURM array job: Task ID = $SLURM_ARRAY_TASK_ID"
    echo ""
    
    # Run for specific tissue assigned to this array task
    python "$PYTHON_SCRIPT" \
        --metadata "$METADATA" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --min-samples $MIN_SAMPLES \
        --flip-rate $FLIP_RATE \
        --correlation-length $CORRELATION_LENGTH \
        --seed $RANDOM_SEED \
        --slurm-array-id $SLURM_ARRAY_TASK_ID
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "Task $SLURM_ARRAY_TASK_ID completed successfully"
    else
        echo ""
        echo "Task $SLURM_ARRAY_TASK_ID failed with exit code $exit_code"
    fi
    
    exit $exit_code
else
    echo "Running as single sequential job (processing all tissues)"
    echo ""
    
    # Run for all tissues
    python "$PYTHON_SCRIPT" \
        --metadata "$METADATA" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --min-samples $MIN_SAMPLES \
        --flip-rate $FLIP_RATE \
        --correlation-length $CORRELATION_LENGTH \
        --seed $RANDOM_SEED
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "========================================================================"
        echo "SUCCESS!"
        echo "========================================================================"
        echo ""
        echo "Synthetic samples saved to: $OUTPUT_DIR"
        echo ""
        echo "Generated files:"
        echo "  - synthetic_samples/*.npz      (synthetic training files)"
        echo "  - synthetic_metadata.csv       (metadata for synthetic samples)"
        echo "  - combined_metadata.csv        (original + synthetic combined)"
        echo "  - validation_results.csv       (quality metrics)"
        echo ""
        echo "Next steps:"
        echo "  1. Review validation_results.csv"
        echo "  2. Proceed to Step 1.5c (data splitting)"
    else
        echo ""
        echo "========================================================================"
        echo "ERROR: Script failed"
        echo "========================================================================"
        exit 1
    fi
fi
